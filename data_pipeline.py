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

    def _scan(self, word, periods, region=None, max_points=1000, min_period_count=500, batch_rows=65_536):
        """
        One streamed pass over the word's context vectors, returning
          - a sample of up to max_points uses per period (for PCA and clustering; balanced so no period dominates)
          - the mean vector and count per (period, state), and per period over all states
        Periods with fewer than min_period_count uses are dropped: a few hundred mostly-OCR-noise uses
        would otherwise date senses.
        """
        path = self.embeddings_path(word)
        meta = pq.read_table(path, columns=["year", "region"])
        years = meta.column("year").to_numpy()
        state_codes, states = pd.factorize(pd.Series(meta.column("region").to_pylist()))
        in_region = np.ones(len(years), bool) if region is None else states[state_codes] == region

        starts = np.array([s for s, _ in periods])
        p_idx = np.full(len(years), -1)
        for i, (s, e) in enumerate(periods):
            p_idx[(years >= s) & (years <= e) & in_region] = i
        counts = np.bincount(p_idx[p_idx >= 0], minlength=len(periods))
        p_idx[np.isin(p_idx, np.flatnonzero(counts < min_period_count))] = -1

        rng = np.random.default_rng(0)
        sampled = np.zeros(len(years), bool)
        for i in np.unique(p_idx[p_idx >= 0]):
            idx = np.flatnonzero(p_idx == i)
            sampled[rng.choice(idx, min(max_points, len(idx)), replace=False)] = True

        # Group sums via a sparse one-hot matrix: key = period * n_states + state; last n_periods keys = all states
        n_p, n_s = len(periods), len(states)
        key = np.where((p_idx >= 0) & (state_codes >= 0), p_idx * n_s + state_codes, -1)
        dim = pq.ParquetFile(path).schema_arrow.field("embedding").type.list_size
        sums, sizes = np.zeros((n_p * n_s + n_p, dim)), np.zeros(n_p * n_s + n_p)
        emb, sents, times, offset = [], [], [], 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=["embedding", "sentence"]):
            n = batch.num_rows
            X = self._to_matrix(batch)
            for keys in (key[offset:offset + n], np.where(p_idx[offset:offset + n] >= 0,
                                                          n_p * n_s + p_idx[offset:offset + n], -1)):
                ok = keys >= 0
                onehot = sp.csr_matrix((np.ones(ok.sum()), (keys[ok], np.flatnonzero(ok))), shape=(len(sums), n))
                sums += onehot @ X
                sizes += np.bincount(keys[ok], minlength=len(sums))
            take = np.flatnonzero(sampled[offset:offset + n])
            if len(take):
                emb.append(X[take])
                sents += [batch.column("sentence")[int(j)].as_py() for j in take]
                times += list(starts[p_idx[offset + take]])
            offset += n

        means = sums / np.maximum(sizes, 1)[:, None]
        by_state = {(starts[i], states[j]): (means[i * n_s + j], sizes[i * n_s + j])
                    for i in range(n_p) for j in range(n_s) if sizes[i * n_s + j]}
        overall = {starts[i]: means[n_p * n_s + i] for i in range(n_p) if sizes[n_p * n_s + i]}
        if not emb:
            return None
        return np.vstack(emb), sents, np.array(times), by_state, overall

    def plot_word(self, word, periods, region=None, n_senses=None, k_max=8, max_points=1000,
                  min_period_count=500, min_share=0.1, n_states=6, min_state_count=200,
                  concreteness=None, axes=None):
        """
        Following Li et al.: context vectors -> top 2 PCs -> k-means sense clusters in that 2-D space.
        (a) Sense space: every sampled use, coloured by sense; black line = the word's mean vector per period.
        (b) Sense shares per period: emergence and decline. A sense emerges at the first period where it holds
            >= min_share of uses and still does in the next (k-means hands every sense a few stray early uses).
        (c) The word's mean vector per period in each of the n_states states with the most uses, in the same
            PC space (a period is shown for a state only if it has >= min_state_count uses there).
        """
        if axes is None:
            axes = plt.figure(figsize=(19, 5.5)).subplots(1, 3)
        where = region or "all states"
        scan = self._scan(word, periods, region, max_points, min_period_count)
        if scan is None or len(np.unique(scan[2])) < 2:
            axes[0].set_title(f"'{word}' — {where} (not enough data)")
            return axes
        emb, sents, times, by_state, overall = scan

        pca = PCA(n_components=2).fit(emb)
        X = pca.transform(emb)
        k = n_senses or self._choose_k(X, k_max)
        labels = KMeans(k, n_init=10, random_state=0).fit_predict(X)

        # Number senses by emergence (then by mean year), so sense 1 is the oldest
        period_starts = np.unique(times)
        share = np.array([[np.mean(labels[times == t] == s) for t in period_starts] for s in range(k)])
        above = share >= min_share
        sustained = above & np.hstack([above[:, 1:], above[:, -1:]])  # last period only needs itself
        emerged = np.array([period_starts[np.argmax(row)] if row.any() else period_starts[share[s].argmax()]
                            for s, row in enumerate(sustained)])
        mean_time = np.array([times[labels == s].mean() for s in range(k)])
        order = np.lexsort((mean_time, emerged))
        names = self._sense_names(sents, labels, k)
        colors = plt.get_cmap("tab10").colors

        def label(s):
            extra = ""
            if concreteness:
                extra = f", conc {self._sense_concreteness([x for x, l in zip(sents, labels) if l == s], concreteness):.2f}"
            return f"{names[s]} (from {emerged[s]}s{extra})"

        # (a) Sense space
        ax = axes[0]
        for rank, s in enumerate(order):
            m = labels == s
            ax.scatter(X[m, 0], X[m, 1], s=3, alpha=0.25, color=colors[rank % 10], label=label(s))
        path = np.array([pca.transform(overall[t][None])[0] for t in period_starts])
        ax.plot(path[:, 0], path[:, 1], "-o", color="k", lw=2, ms=4)
        for t, (x, y) in zip(period_starts, path):
            ax.annotate(f"{t}s", (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"'{word}' — {where}: {k} senses")
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.0%})")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.0%})")
        ax.legend(fontsize=7, markerscale=4, loc="best")

        # (b) Sense shares over time
        ax = axes[1]
        ax.stackplot(period_starts, share[order], colors=[colors[r % 10] for r in range(k)], alpha=0.8)
        for rank, s in enumerate(order):
            ax.axvline(emerged[s], color=colors[rank % 10], ls=":", lw=1.5)
        ax.set_xticks(period_starts)
        ax.set_xlim(period_starts[0], period_starts[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Decade")
        ax.set_ylabel("Share of uses")
        ax.set_title("Sense shares (dotted = emergence)")

        # (c) Per-state trajectories in the same PC space
        ax = axes[2]
        totals = pd.Series({st: n for (_, st), (_, n) in by_state.items()}).groupby(level=0).sum()
        for i, st in enumerate(totals.nlargest(n_states).index if region is None else [region]):
            pts = [(t, v) for t in period_starts for (tt, s2), (v, n) in by_state.items()
                   if tt == t and s2 == st and n >= min_state_count]
            if len(pts) < 2:
                continue
            P = pca.transform(np.array([v for _, v in pts]))
            ax.plot(P[:, 0], P[:, 1], "-o", ms=3, color=colors[i % 10], label=st)
            ax.annotate(f"{pts[0][0]}s", P[0], fontsize=7, color=colors[i % 10])
            ax.annotate(f"{pts[-1][0]}s", P[-1], fontsize=7, color=colors[i % 10])
        ax.plot(path[:, 0], path[:, 1], "--", color="k", lw=1.5, label="all states")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Mean vector per decade, by state")
        ax.legend(fontsize=7)
        return axes


EMBED_MODEL = "bert-base-uncased"  # as in Li et al.; any masked LM works, e.g. "sentence-transformers/all-mpnet-base-v2"
WORDS = ["trust", "strike", "wire", "lobby", "deadline"]  # keep --array in data_pipeline.sh at 0-(len-1)
DECADES = [(y, y + 9) for y in range(1860, 1921, 10)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["clean", "word", "plot"],
                        help="clean: OCR-clean year files; word: extract+embed+plot one word; plot: combined figure")
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
        if not pipeline.sentences_path(word).exists():
            pipeline.extract_sentences(word, workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
        if not pipeline.embeddings_path(word).exists():
            pipeline.embed_sentences(word)
        pipeline.plot_word(word, DECADES, concreteness=concreteness)
        plt.tight_layout()
        plt.savefig(pipeline.word_dir(word) / "sense_trajectory.png", dpi=150)

    elif args.stage == "plot":
        fig = plt.figure(figsize=(19, 5.5 * len(WORDS)))
        axes = fig.subplots(len(WORDS), 3, squeeze=False)
        for word, row in zip(WORDS, axes):
            pipeline.plot_word(word, DECADES, concreteness=concreteness, axes=row)
        plt.tight_layout()
        plt.savefig("sense_trajectories.png", dpi=150)
