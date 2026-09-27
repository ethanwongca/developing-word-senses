import argparse
import json
import multiprocessing as mp
import os
import re
import unicodedata
from functools import cached_property
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
from nltk.tokenize import sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

# Any mask token a sentences.jsonl may contain (older files were written with MPNet's <mask>)
MASK_RE = re.compile(r"<mask>|\[MASK\]")


def _extract_year(args) -> list[str]:
    """Worker: JSON rows for every sentence in one year file containing the word (module-level so it pickles)."""
    path, year, word, mask_token, region_col, batch_rows = args
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    rows = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["article", region_col]):
        for article, region in zip(batch.column("article").to_pylist(), batch.column(region_col).to_pylist()):
            # Cheap article-level check first: few articles contain the word, and sent_tokenize is the bottleneck
            if not isinstance(article, str) or not pattern.search(article):
                continue
            for sentence in sent_tokenize(article):
                if pattern.search(sentence):
                    rows.append(json.dumps({"word": word, "year": year, "region": region,
                                            "sentence": pattern.sub(mask_token, sentence)}))
    return rows


class DataPipeline:
    def __init__(self, data_dir: str, result_dir: str, model: str, start_year: int, end_year: int):
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.mask_token = self.tokenizer.mask_token
        if self.mask_token is None:
            raise ValueError(f"Model '{model}' has no mask token.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_year = start_year
        self.end_year = end_year

    @cached_property
    def model(self):
        """Loaded on first use, so plotting on a CPU node doesn't pay for it."""
        model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        return model.half() if self.device == "cuda" else model

    def word_dir(self, word: str) -> Path:
        return self.result_dir / "words" / word

    def sentences_path(self, word: str) -> Path:
        return self.word_dir(word) / "sentences.jsonl"

    def embeddings_path(self, word: str) -> Path:
        # New name: embeddings.parquet holds the old whole-sentence embeddings, which must not be reused
        return self.word_dir(word) / "context_vectors.parquet"

    @staticmethod
    def _clean_ocr_text(text: str, basic: bool = False) -> str:
        """Removes hyphenation, collapses whitespace, optionally normalizes/strips accents."""
        if not isinstance(text, str):
            return text
        text = re.sub(r"-[ \t]*\r?\n[ \t]*", "", text)
        if not basic:
            text = unicodedata.normalize("NFKC", text)
            text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
        return re.sub(r"\s+", " ", text).strip()

    def clean_pipeline(self, batch_rows: int = 10_000):
        """Clean each year file in row batches, so a large year never sits fully in memory."""
        self.result_dir.mkdir(parents=True, exist_ok=True)
        for file in sorted(self.data_dir.glob("*.parquet")):
            out_path = self.result_dir / file.name
            if not (self.start_year <= int(file.stem) <= self.end_year) or out_path.exists():
                continue
            writer = None
            for batch in pq.ParquetFile(file).iter_batches(batch_size=batch_rows):
                df = batch.to_pandas()
                df["article"] = df["article"].map(self._clean_ocr_text)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table.cast(writer.schema))
            if writer:
                writer.close()

    def extract_sentences(self, word: str, region_col: str = "State", batch_rows: int = 10_000, workers: int = 1):
        """Write {word, year, region, sentence} per matching sentence, one year file per worker process."""
        jobs = [(str(self.result_dir / f"{year}.parquet"), year, word, self.mask_token, region_col, batch_rows)
                for year in range(self.start_year, self.end_year + 1)
                if (self.result_dir / f"{year}.parquet").exists()]
        if not jobs:
            raise FileNotFoundError(f"No year files for {self.start_year}-{self.end_year} in {self.result_dir}")
        # Fail now, not hours into the job, if a year file lacks a column (as happened with "region" vs "State")
        for path, year, *_ in jobs:
            missing = {"article", region_col} - set(pq.read_schema(path).names)
            if missing:
                raise KeyError(f"{path} lacks column(s) {sorted(missing)}; has {pq.read_schema(path).names}")
        out_path = self.sentences_path(word)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".jsonl.tmp")  # renamed on success, so a crash can't leave a partial file
        # spawn, not fork: the model may already hold a CUDA context, which forked children can't share
        with open(tmp_path, "w") as out, mp.get_context("spawn").Pool(workers) as pool:
            for rows in pool.imap(_extract_year, jobs):  # imap keeps year order, so output is deterministic
                out.writelines(r + "\n" for r in rows)
        tmp_path.replace(out_path)

    def _window(self, sentence: str, width: int = 50) -> str:
        """Keep `width` words either side of the first mask, so long OCR run-ons don't truncate it away."""
        words = MASK_RE.sub(f" {self.mask_token} ", sentence).split()
        i = next((j for j, w in enumerate(words) if w == self.mask_token), 0)
        return " ".join(words[max(0, i - width): i + width + 1])

    def _context_vectors(self, sentences: list[str], n_layers: int = 6) -> tuple[np.ndarray, np.ndarray]:
        """
        Li et al.: the hidden state at the masked target, averaged over the last n_layers layers.
        It is the model's representation of what fits the slot given the context, i.e. the sense.
        Returns (vectors, keep); keep is False where the mask didn't survive tokenization.
        """
        enc = self.tokenizer([self._window(s) for s in sentences], padding=True, truncation=True,
                             max_length=256, return_tensors="pt").to(self.device)
        is_mask = enc["input_ids"] == self.tokenizer.mask_token_id
        rows = torch.arange(len(sentences), device=self.device)
        first = is_mask.int().argmax(dim=1)  # first mask if the word occurs more than once
        with torch.inference_mode():
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[-n_layers:]
            vecs = torch.stack([h[rows, first] for h in hidden]).float().mean(dim=0)
        return vecs.cpu().numpy(), is_mask.any(dim=1).cpu().numpy()

    def embed_sentences(self, word: str, batch_size: int = 256, chunk_size: int = 16_384):
        """Stream sentences.jsonl in chunks, write one context vector per use (sentence kept for sense labels)."""
        dim = self.model.config.hidden_size
        schema = pa.schema([
            ("word", pa.string()),
            ("year", pa.int32()),
            ("region", pa.string()),
            ("sentence", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
        ])
        with open(self.sentences_path(word)) as f:
            first = json.loads(f.readline() or "{}")
        if first and set(first) != {"word", "year", "region", "sentence"}:
            raise KeyError(f"{self.sentences_path(word)} rows have keys {sorted(first)}; "
                           "expected word, year, region, sentence (delete it to re-extract)")
        buffer = []
        out_path = self.embeddings_path(word)
        tmp_path = out_path.with_suffix(".parquet.tmp")

        with pq.ParquetWriter(tmp_path, schema) as writer:
            def flush():
                if not buffer:
                    return
                # Length-sorted batches waste far less compute on padding
                order = np.argsort([len(r["sentence"]) for r in buffer])
                vecs, keep = np.zeros((len(buffer), dim), np.float32), np.zeros(len(buffer), bool)
                for i in range(0, len(order), batch_size):
                    idx = order[i:i + batch_size]
                    vecs[idx], keep[idx] = self._context_vectors([buffer[j]["sentence"] for j in idx])
                rows = [r for r, k in zip(buffer, keep) if k]
                table = pa.table({
                    "word": [r["word"] for r in rows],
                    "year": [r["year"] for r in rows],
                    "region": [r["region"] for r in rows],
                    "sentence": [r["sentence"] for r in rows],
                    "embedding": pa.FixedSizeListArray.from_arrays(pa.array(vecs[keep].ravel()), dim),
                }, schema=schema)
                writer.write_table(table)
                buffer.clear()

            with open(self.sentences_path(word)) as f:
                for line in f:
                    buffer.append(json.loads(line))
                    if len(buffer) == chunk_size:
                        flush()
            flush()
        tmp_path.replace(out_path)

    def _read(self, word, start_year, end_year, region, columns) -> pa.Table:
        """Read only rows matching word/years/region (predicate pushdown)."""
        filters = [("word", "==", word), ("year", ">=", start_year), ("year", "<=", end_year)]
        if region is not None:
            filters.append(("region", "==", region))
        return pq.read_table(self.embeddings_path(word), columns=columns, filters=filters)

    @staticmethod
    def _to_matrix(table) -> np.ndarray:
        col = table.column("embedding")
        col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        return col.flatten().to_numpy().reshape(len(col), col.type.list_size)

    def _load_embeddings(self, word, start_year, end_year, region=None) -> Optional[np.ndarray]:
        table = self._read(word, start_year, end_year, region, ["embedding"])
        return None if table.num_rows == 0 else self._to_matrix(table)

    def get_embeddings(self, word, start_year, end_year, region=None) -> Optional[np.ndarray]:
        """Pooled: mean context vector over the word's uses in the period (and region)."""
        emb = self._load_embeddings(word, start_year, end_year, region)
        return None if emb is None else emb.mean(axis=0)

    @staticmethod
    def load_concreteness(path: str) -> dict:
        """Brysbaert, Warriner & Kuperman (2014) norms: tab-separated, columns 'Word' and 'Conc.M'."""
        df = pd.read_csv(path, sep="\t")
        return dict(zip(df["Word"].astype(str).str.lower(), df["Conc.M"]))

    @staticmethod
    def _choose_k(X, k_max=8):
        """Elbow: the k where the inertia curve lies farthest below the straight line from k=1 to k_max."""
        ks = np.arange(1, k_max + 1)
        inertia = np.array([KMeans(k, n_init=10, random_state=0).fit(X).inertia_ for k in ks])
        x = (ks - 1) / (k_max - 1)
        y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
        return max(2, int(ks[np.argmax((1 - x) - y)]))

    @staticmethod
    def _sense_names(sents, labels, k, n_terms=2):
        """Label each sense with its most distinctive context words (class-based TF-IDF)."""
        docs = [" ".join(s for s, l in zip(sents, labels) if l == c) for c in range(k)]
        vec = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | {"mask"}),
                              token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        X = vec.fit_transform(docs).toarray()
        terms = vec.get_feature_names_out()
        return [", ".join(terms[X[c].argsort()[::-1][:n_terms]]) for c in range(k)]

    @staticmethod
    def _sense_concreteness(sents, concreteness):
        """Mean concreteness of the content words around the target word."""
        tokens = re.findall(r"[a-z]+", " ".join(sents).lower())
        vals = [concreteness[t] for t in tokens if t in concreteness and t not in ENGLISH_STOP_WORDS]
        return np.mean(vals) if vals else np.nan

    def _scan(self, word, periods, states=None, max_points=1000, state_points=500,
              min_period_count=500, min_state_count=200, batch_rows=65_536) -> Optional[dict]:
        """
        One streamed pass over the word's context vectors, returning
          nat:       a balanced sample of up to max_points uses per period (fits PCA and the senses)
          samples:   {state: sample of up to state_points uses per period in that state}
          means:     {(period, state): mean vector}; overall: {period: mean vector}
          eligible:  states with >= min_state_count uses in every kept period, so their diagrams cover
                     the same decades (states=None samples exactly these; pass a list to choose, [] for none)
        Periods with < min_period_count uses nationally are dropped: a few hundred mostly-OCR-noise uses
        would otherwise date senses.
        """
        path = self.embeddings_path(word)
        meta = pq.read_table(path, columns=["year", "region"])
        years = meta.column("year").to_numpy()
        codes, names = pd.factorize(pd.Series(meta.column("region").to_pylist()))
        names = list(names)
        starts = np.array([s for s, _ in periods])
        n_p, n_s = len(periods), len(names)

        p_idx = np.full(len(years), -1)
        for i, (s, e) in enumerate(periods):
            p_idx[(years >= s) & (years <= e)] = i
        sparse = np.flatnonzero(np.bincount(p_idx[p_idx >= 0], minlength=n_p) < min_period_count)
        p_idx[np.isin(p_idx, sparse)] = -1
        live = np.unique(p_idx[p_idx >= 0])
        if len(live) < 2:
            return None

        has_state = (p_idx >= 0) & (codes >= 0)
        counts = np.zeros((n_p, n_s), int)
        np.add.at(counts, (p_idx[has_state], codes[has_state]), 1)
        # "Georgia, Virginia"-style values are papers filed under several states, not a state
        eligible = [st for j, st in enumerate(names) if "," not in st and (counts[live, j] >= min_state_count).all()]
        states = eligible if states is None else [st for st in states if st in names]

        rng = np.random.default_rng(0)
        in_nat, in_state = np.zeros(len(years), bool), np.zeros(len(years), bool)
        for i in live:
            idx = np.flatnonzero(p_idx == i)
            in_nat[rng.choice(idx, min(max_points, len(idx)), replace=False)] = True
            for st in states:
                idx = np.flatnonzero((p_idx == i) & (codes == names.index(st)))
                in_state[rng.choice(idx, min(state_points, len(idx)), replace=False)] = True

        # Group sums via a sparse one-hot matrix: key = period * n_states + state; last n_p keys = all states
        key = np.where(has_state, p_idx * n_s + codes, -1)
        nat_key = np.where(p_idx >= 0, n_p * n_s + p_idx, -1)
        dim = pq.ParquetFile(path).schema_arrow.field("embedding").type.list_size
        sums, sizes = np.zeros((n_p * n_s + n_p, dim)), np.zeros(n_p * n_s + n_p)
        kept_X, kept_rows, sents, offset = [], [], [], 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["embedding", "sentence"]):
            n = batch.num_rows
            X = self._to_matrix(batch)
            for keys in (key[offset:offset + n], nat_key[offset:offset + n]):
                ok = keys >= 0
                onehot = sp.csr_matrix((np.ones(ok.sum()), (keys[ok], np.flatnonzero(ok))), shape=(len(sums), n))
                sums += onehot @ X
                sizes += np.bincount(keys[ok], minlength=len(sums))
            take = np.flatnonzero(in_nat[offset:offset + n] | in_state[offset:offset + n])
            if len(take):
                kept_X.append(X[take])
                kept_rows.append(offset + take)
                sents += batch.column("sentence").take(pa.array(take)).to_pylist()
            offset += n

        rows, KX, sents = np.concatenate(kept_rows), np.vstack(kept_X), np.array(sents, dtype=object)

        def group(mask):
            return KX[mask], list(sents[mask]), starts[p_idx[rows[mask]]]

        means = sums / np.maximum(sizes, 1)[:, None]
        return {
            "word": word,
            "nat": group(in_nat[rows]),
            "samples": {st: group(in_state[rows] & (codes[rows] == names.index(st))) for st in states},
            "means": {(starts[i], names[j]): means[i * n_s + j]
                      for i in live for j in range(n_s) if sizes[i * n_s + j] >= min_state_count},
            "overall": {starts[i]: means[n_p * n_s + i] for i in live},
            "eligible": eligible,
            "totals": {st: int(counts[live, names.index(st)].sum()) for st in eligible},
        }

    @staticmethod
    def _shares(labels, times, k, min_share):
        """
        Share of each sense per period, and its emergence: the first period where it holds >= min_share of
        uses and still does in the next (k-means hands every sense a few stray early uses, so a blip isn't it).
        """
        ts = np.unique(times)
        share = np.array([[np.mean(labels[times == t] == s) for t in ts] for s in range(k)])
        above = share >= min_share
        sustained = above & np.hstack([above[:, 1:], above[:, -1:]])  # last period only needs itself
        emerged = np.array([ts[np.argmax(row)] if row.any() else ts[share[s].argmax()]
                            for s, row in enumerate(sustained)])
        return ts, share, emerged

    def analyze_word(self, word, periods, states=None, n_senses=None, k_max=8, min_share=0.1,
                     concreteness=None, **scan_kwargs) -> Optional[dict]:
        """
        Following Li et al.: context vectors -> top 2 PCs -> k-means sense clusters in that 2-D space.
        PCA and senses are fitted once on the national sample, then applied to every state, so a sense means
        the same thing (same colour, same number) in every state's diagram. Senses are numbered by emergence.
        """
        res = self._scan(word, periods, states, **scan_kwargs)
        if res is None:
            return None
        emb, sents, times = res["nat"]
        pca = PCA(n_components=2).fit(emb)
        X = pca.transform(emb)
        k = n_senses or self._choose_k(X, k_max)
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        _, _, emerged = self._shares(km.labels_, times, k, min_share)
        mean_time = np.array([times[km.labels_ == s].mean() for s in range(k)])
        order = np.lexsort((mean_time, emerged))
        rank = np.empty(k, int)
        rank[order] = np.arange(k)
        names = self._sense_names(sents, rank[km.labels_], k)
        if concreteness:
            names = [f"{n}, conc {self._sense_concreteness([x for x, l in zip(sents, rank[km.labels_]) if l == s], concreteness):.2f}"
                     for s, n in enumerate(names)]
        res.update(pca=pca, km=km, rank=rank, k=k, names=names, min_share=min_share)
        return res

    @staticmethod
    def _assign(res, emb):
        """PC coordinates and (emergence-ordered) sense of each vector, in the word's national sense space."""
        X = res["pca"].transform(emb)
        return X, res["rank"][res["km"].predict(X)]

    def _path(self, res, state=None):
        """The word's mean vector per period (nationally, or in one state) in PC space."""
        pts = [(t, v) for t, v in sorted(res["overall"].items())] if state is None else \
              [(t, res["means"][(t, state)]) for t in sorted(res["overall"]) if (t, state) in res["means"]]
        return [t for t, _ in pts], res["pca"].transform(np.array([v for _, v in pts]))

    def _plot_space(self, ax, res, emb, times, emerged, state=None):
        """Uses coloured by sense, with the mean-vector path (state solid, national dashed)."""
        X, labels = self._assign(res, emb)
        for s in range(res["k"]):
            m = labels == s
            ax.scatter(X[m, 0], X[m, 1], s=3, alpha=0.25, color=COLORS[s % 10],
                       label=f"{s + 1}. {res['names'][s]} (from {emerged[s]}s)")
        for st, style in ([(state, "-o"), (None, "--")] if state else [(None, "-o")]):
            ts, P = self._path(res, st)
            ax.plot(P[:, 0], P[:, 1], style, color="k", lw=2 if st == state else 1.2, ms=4,
                    label=None if st == state else "national mean")
            if st == state:
                for t, (x, y) in zip(ts, P):
                    ax.annotate(f"{t}s", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        pca = res["pca"]
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.legend(fontsize=7, markerscale=4, loc="best")

    @staticmethod
    def _plot_shares(ax, res, ts, share, emerged, legend=False):
        ax.stackplot(ts, share, colors=[COLORS[s % 10] for s in range(res["k"])], alpha=0.8,
                     labels=[f"{s + 1}. {n}" for s, n in enumerate(res["names"])])
        for s in range(res["k"]):
            ax.axvline(emerged[s], color=COLORS[s % 10], ls=":", lw=1.5)
        ax.set_xticks(ts)
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(0, 1)
        if legend:
            ax.legend(fontsize=7, loc="upper left")

    def plot_overview(self, res, n_labeled=6, axes=None):
        """
        (a) National sense space: every sampled use coloured by sense; black line = mean vector per decade.
        (b) National sense shares per decade (dotted = emergence).
        (c) Mean vector per decade for every eligible state in the same space; the n_labeled states with
            the most uses are coloured, the rest grey.
        """
        if axes is None:
            axes = plt.figure(figsize=(19, 5.5)).subplots(1, 3)
        emb, _, times = res["nat"]
        ts, share, emerged = self._shares(self._assign(res, emb)[1], times, res["k"], res["min_share"])
        self._plot_space(axes[0], res, emb, times, emerged)
        axes[0].set_title(f"'{res['word']}' — all states: {res['k']} senses")
        self._plot_shares(axes[1], res, ts, share, emerged)
        axes[1].set_title("Sense shares (dotted = emergence)")
        axes[1].set_xlabel("Decade")
        axes[1].set_ylabel("Share of uses")

        ax = axes[2]
        top = sorted(res["totals"], key=res["totals"].get, reverse=True)[:n_labeled]
        for st in res["eligible"]:
            _, P = self._path(res, st)
            if st in top:
                c = COLORS[top.index(st) % 10]
                ax.plot(P[:, 0], P[:, 1], "-o", ms=3, color=c, label=st, zorder=3)
                ax.annotate(f"{ts[-1]}s", P[-1], fontsize=7, color=c)
            else:
                ax.plot(P[:, 0], P[:, 1], "-", color="0.8", lw=0.8, zorder=1)
        _, P = self._path(res)
        ax.plot(P[:, 0], P[:, 1], "--o", ms=3, color="k", lw=1.5, label="national", zorder=4)
        ax.annotate(f"{ts[0]}s", P[0], fontsize=7)
        ax.annotate(f"{ts[-1]}s", P[-1], fontsize=7)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"Mean vector per decade, {len(res['eligible'])} states (grey = unlabeled)")
        ax.legend(fontsize=7)
        return axes

    def plot_state(self, res, state):
        """One state's diagram in the national sense space: its uses and path, and its sense shares."""
        axes = plt.figure(figsize=(13, 5.5)).subplots(1, 2)
        emb, _, times = res["samples"][state]
        ts, share, emerged = self._shares(self._assign(res, emb)[1], times, res["k"], res["min_share"])
        self._plot_space(axes[0], res, emb, times, emerged, state=state)
        axes[0].set_title(f"'{res['word']}' — {state} ({res['totals'][state]:,} uses)")
        self._plot_shares(axes[1], res, ts, share, emerged)
        axes[1].set_title(f"Sense shares in {state} (dotted = emergence here)")
        axes[1].set_xlabel("Decade")
        axes[1].set_ylabel("Share of uses")
        return axes

    def plot_state_grid(self, res, ncols=6):
        """Small multiples: every state's sense shares per decade, to compare where and when senses spread."""
        states = sorted(res["samples"])
        ncols = min(ncols, len(states))
        nrows = -(-len(states) // ncols)
        fig = plt.figure(figsize=(3.2 * ncols, 2.6 * nrows + 1))
        axes = fig.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True).ravel()
        for ax, st in zip(axes, states):
            emb, _, times = res["samples"][st]
            ts, share, emerged = self._shares(self._assign(res, emb)[1], times, res["k"], res["min_share"])
            self._plot_shares(ax, res, ts, share, emerged)
            ax.set_title(st, fontsize=9)
            ax.tick_params(labelsize=6)
        for ax in axes[len(states):]:
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s % 10]) for s in range(res["k"])]
        fig.legend(handles, [f"{s + 1}. {n}" for s, n in enumerate(res["names"])], loc="lower center",
                   bbox_to_anchor=(0.5, 1.0), ncol=min(res["k"], 4), fontsize=9)
        fig.suptitle(f"'{res['word']}' — sense shares by state (dotted = emergence)", y=1.08)
        fig.tight_layout()
        return fig

    def state_emergence(self, res) -> pd.DataFrame:
        """Emergence decade of each sense in each state (rows), plus the national row."""
        rows = {}
        for st, (emb, _, times) in [("national", res["nat"])] + sorted(res["samples"].items()):
            rows[st] = self._shares(self._assign(res, emb)[1], times, res["k"], res["min_share"])[2]
        return pd.DataFrame(rows, index=[f"{s + 1}. {n}" for s, n in enumerate(res["names"])]).T


COLORS = plt.get_cmap("tab10").colors
EMBED_MODEL = "bert-base-uncased"  # as in Li et al.; any masked LM works, e.g. "sentence-transformers/all-mpnet-base-v2"
WORDS = [
    # Original set
    "trust",      # legal trust -> monopoly ("the Standard Oil trust"), 1880s
    "strike", "wire", "lobby",
    "deadline",   # Civil War prison line -> time limit, 1920s
    # Controls: new sense arrives with a technology at a documented date
    "tank",       # water tank -> military tank, 1916
    "broadcast",  # sowing seed -> radio, ~1920
    "plane",      # tool / geometry -> aeroplane, 1908+
    "film",       # thin layer -> motion picture, 1905-1915
    "record",     # written record -> phonograph record, 1890s+
    "car",        # railway car / carriage -> automobile, 1900-1910
    "station",    # railway station -> radio station, 1920s
    "screen",     # fire screen -> movie screen, 1910s
]  # data_pipeline.sh sizes the job array from this list
DECADES = [(y, y + 9) for y in range(1860, 1921, 10)]
# State diagrams: None = every state with >= 200 uses of the word in every decade; or name them, e.g. ["New York"]
STATES = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["clean", "word", "plot"],
                        help="clean: OCR-clean year files; word: extract+embed+all diagrams for one word; "
                             "plot: stack every word's overview into one figure")
    parser.add_argument("index", nargs="?", type=int, help="index into WORDS (word stage; SLURM_ARRAY_TASK_ID)")
    args = parser.parse_args()

    pipeline = DataPipeline(
        "/home/ewong/scratch/american_stories/merged_data",
        "/home/ewong/scratch/developing-word-senses/clean_data",
        EMBED_MODEL,
        1860, 1929,
    )
    concreteness = None

    if args.stage == "clean":
        pipeline.clean_pipeline()

    elif args.stage == "word":
        word = WORDS[args.index]
        out = pipeline.word_dir(word)
        if not pipeline.sentences_path(word).exists():
            pipeline.extract_sentences(word, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        if not pipeline.embeddings_path(word).exists():
            pipeline.embed_sentences(word)

        res = pipeline.analyze_word(word, DECADES, states=STATES, concreteness=concreteness)
        if res is None:
            raise SystemExit(f"'{word}': fewer than 2 decades with enough uses")

        pipeline.plot_overview(res)
        plt.tight_layout()
        plt.savefig(out / "sense_trajectory.png", dpi=150)
        plt.close("all")

        if res["samples"]:
            pipeline.plot_state_grid(res)
            plt.savefig(out / "state_shares.png", dpi=150, bbox_inches="tight")
            plt.close("all")
            pipeline.state_emergence(res).to_csv(out / "state_emergence.csv")

        (out / "states").mkdir(exist_ok=True)
        for state in res["samples"]:
            pipeline.plot_state(res, state)
            plt.tight_layout()
            plt.savefig(out / "states" / f"{state.replace(' ', '_')}.png", dpi=150)
            plt.close("all")
        print(f"'{word}': {res['k']} senses, {len(res['samples'])} state diagrams -> {out}")

    elif args.stage == "plot":
        # Stitch the per-word overviews (already rendered by the word tasks) instead of re-scanning everything
        pngs = [pipeline.word_dir(w) / "sense_trajectory.png" for w in WORDS]
        pngs = [p for p in pngs if p.exists()]  # a failed word task doesn't block the rest
        fig = plt.figure(figsize=(19, 5.5 * len(pngs)))
        for ax, png in zip(fig.subplots(len(pngs), 1, squeeze=False).ravel(), pngs):
            ax.imshow(plt.imread(png))
            ax.axis("off")
        plt.tight_layout()
        plt.savefig("sense_trajectories.png", dpi=150)
