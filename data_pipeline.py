import argparse
import json
import multiprocessing as mp
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.metrics import silhouette_score


class Arrow3D(FancyArrowPatch):
    """Arrow drawn in screen space, so the head isn't distorted by unequal axis scales."""

    def __init__(self, start, end, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._start, self._end = start, end

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = zip(self._start, self._end)
        xs, ys, zs = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


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
        self.model = SentenceTransformer(model)
        self.mask_token = self.model.tokenizer.mask_token
        if self.mask_token is None:
            raise ValueError(f"Model '{model}' has no mask token.")
        self.start_year = start_year
        self.end_year = end_year

    def word_dir(self, word: str) -> Path:
        return self.result_dir / "words" / word

    def sentences_path(self, word: str) -> Path:
        return self.word_dir(word) / "sentences.jsonl"

    def embeddings_path(self, word: str) -> Path:
        return self.word_dir(word) / "embeddings.parquet"

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

    def embed_sentences(self, word: str, batch_size: int = 128, chunk_size: int = 16_384):
        """Stream sentences.jsonl in chunks, embed, append to embeddings.parquet (sentence kept for sense labels)."""
        dim = self.model.get_sentence_embedding_dimension()
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
                emb = self.model.encode([r["sentence"] for r in buffer], batch_size=batch_size,
                                        normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
                table = pa.table({
                    "word": [r["word"] for r in buffer],
                    "year": [r["year"] for r in buffer],
                    "region": [r["region"] for r in buffer],
                    "sentence": [r["sentence"] for r in buffer],
                    "embedding": pa.FixedSizeListArray.from_arrays(pa.array(emb.ravel()), dim),
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
    def _to_matrix(table: pa.Table) -> np.ndarray:
        col = table.column("embedding").combine_chunks()
        return col.flatten().to_numpy().reshape(table.num_rows, col.type.list_size)

    def _load_embeddings(self, word, start_year, end_year, region=None) -> Optional[np.ndarray]:
        table = self._read(word, start_year, end_year, region, ["embedding"])
        return None if table.num_rows == 0 else self._to_matrix(table)

    def get_embeddings(self, word, start_year, end_year, region=None) -> Optional[np.ndarray]:
        """Pooled: mean over sentences, re-normalized so cosine comparisons are valid."""
        emb = self._load_embeddings(word, start_year, end_year, region)
        if emb is None:
            return None
        mean = emb.mean(axis=0)
        return mean / np.linalg.norm(mean)

    @staticmethod
    def load_concreteness(path: str) -> dict:
        """Brysbaert, Warriner & Kuperman (2014) norms: tab-separated, columns 'Word' and 'Conc.M'."""
        df = pd.read_csv(path, sep="\t")
        return dict(zip(df["Word"].astype(str).str.lower(), df["Conc.M"]))

    @staticmethod
    def _choose_k(emb, k_max=7):
        """Number of senses = k with best silhouette score."""
        scores = {}
        for k in range(2, min(k_max, len(emb) - 1) + 1):
            labels = KMeans(k, n_init=10, random_state=0).fit_predict(emb)
            scores[k] = silhouette_score(emb, labels, sample_size=min(2000, len(emb)), random_state=0)
        return max(scores, key=scores.get)

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

    def plot_sense_trajectory(self, word, periods, region=None, n_senses=None, max_points=1000,
                              min_share=0.1, min_period_count=500, min_sense_points=20,
                              concreteness=None, ax=None):
        """
        Senses = clusters of contextual embeddings over all periods; every use is plotted at its own period.
        Lines = each sense's per-period centroid from its emergence on (drift = semantic shift within the
        sense); marker size = the sense's share of that period's uses (growth/decline).
        Emergence = first period where the sense holds >= min_share and still does in the next period.
        Arrows = branching: at a sense's emergence, from its most similar earlier sense to it.
        Colour = sense concreteness if norms given, else distance to the initial sense.
        """
        # Sample up to max_points uses per period, so no period dominates the clustering.
        # Sparse periods are skipped: a few hundred (mostly OCR-noise) uses would otherwise date senses.
        rng = np.random.default_rng(0)
        emb, sents, times = [], [], []
        for start, end in periods:
            table = self._read(word, start, end, region, ["embedding", "sentence"])
            if table.num_rows < min_period_count:
                continue
            table = table.take(rng.choice(table.num_rows, min(max_points, table.num_rows), replace=False))
            emb.append(self._to_matrix(table))
            sents += table.column("sentence").to_pylist()
            times += [start] * table.num_rows

        if ax is None:
            ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
        ax.set_title(f"'{word}' — {region or 'all'}")
        if len(emb) < 2:
            ax.set_title(f"'{word}' — {region or 'all'} (not enough data)")
            return ax
        emb, times = np.vstack(emb), np.array(times)

        # Senses: cluster all uses of the word across all periods
        k = n_senses or self._choose_k(emb)
        labels = KMeans(k, n_init=10, random_state=0).fit_predict(emb)
        centroids = np.array([emb[labels == s].mean(axis=0) for s in range(k)])
        unit = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

        # Emergence: first period where the sense holds >= min_share and keeps it the next period
        # (k-means gives every sense some stray early uses, so a one-period blip isn't emergence)
        period_starts = np.unique(times)
        share = np.array([[np.mean(labels[times == t] == s) for t in period_starts] for s in range(k)])
        above = share >= min_share
        sustained = above & np.hstack([above[:, 1:], above[:, -1:]])  # last period only needs itself
        emerged = np.array([period_starts[np.argmax(row)] if row.any() else period_starts[share[s].argmax()]
                            for s, row in enumerate(sustained)])
        mean_time = np.array([times[labels == s].mean() for s in range(k)])
        order = np.lexsort((mean_time, emerged))  # by emergence, ties broken by average year

        # Chaining: link each sense to its most similar earlier-emerged sense
        edges = [(order[:i][np.argmax(unit[order[:i]] @ unit[s])], s) for i, s in enumerate(order) if i > 0]

        # Trajectory: per-period centroid of each sense from its emergence on (skip periods with too few uses)
        traj = {s: [(t, emb[(labels == s) & (times == t)].mean(axis=0), share[s, j])
                    for j, t in enumerate(period_starts)
                    if t >= emerged[s] and ((labels == s) & (times == t)).sum() >= min_sense_points]
                for s in range(k)}

        # Colour
        if concreteness:
            values = np.array([self._sense_concreteness([x for x, l in zip(sents, labels) if l == s], concreteness)
                               for s in range(k)])
            cmap, clabel = plt.get_cmap("coolwarm"), "Sense concreteness"
        else:
            values = 1 - unit @ unit[order[0]]
            cmap, clabel = plt.get_cmap("coolwarm_r"), "Distance to initial sense"
        norm = plt.Normalize(np.nanmin(values), np.nanmax(values))

        # Plot: time x PC1 x PC2
        pca = PCA(n_components=2).fit(emb)
        pcs, cpcs = pca.transform(emb), pca.transform(centroids)
        names = self._sense_names(sents, labels, k)
        traj_pcs = {s: pca.transform(np.array([c for _, c, _ in traj[s]])) if traj[s] else None for s in range(k)}

        def position(s, t):
            """Sense s's centroid at its latest trajectory period <= t (overall centroid if none)."""
            idx = [i for i, (tt, _, _) in enumerate(traj[s]) if tt <= t]
            return traj_pcs[s][idx[-1]] if idx else cpcs[s]

        for s in range(k):
            m, color = labels == s, cmap(norm(values[s]))
            ax.scatter(times[m], pcs[m, 0], pcs[m, 1], s=2, alpha=0.08, color=color)
            if traj[s]:
                ts, shares = [t for t, _, _ in traj[s]], np.array([sh for _, _, sh in traj[s]])
                p = traj_pcs[s]
                ax.plot(ts, p[:, 0], p[:, 1], color=color, lw=2)
                ax.scatter(ts, p[:, 0], p[:, 1], s=20 + 400 * shares, color=color, edgecolor="k",
                           alpha=1, depthshade=False)
            ax.text(emerged[s], *position(s, emerged[s]), names[s], fontsize=8)
        for a, b in edges:
            t = emerged[b]
            ax.add_artist(Arrow3D((t, *position(a, t)), (t, *position(b, t)),
                                  arrowstyle="-|>", mutation_scale=15, color="k", lw=1.5))

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=clabel, shrink=0.6)
        pad = 0.3 * np.diff(period_starts).min() if len(period_starts) > 1 else 5
        ax.set_xlim(period_starts[0] - pad, period_starts[-1] + pad)
        ax.set_xticks(period_starts)
        ax.set_xlabel("Year")
        ax.set_ylabel("PC 1")
        ax.set_zlabel("PC 2")
        return ax


WORDS = ["trust", "strike", "wire", "lobby", "deadline"]
DECADES = [(y, y + 9) for y in range(1820, 1921, 10)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["clean", "word", "plot"],
                        help="clean: OCR-clean year files; word: extract+embed+plot one word; plot: combined figure")
    parser.add_argument("index", nargs="?", type=int, help="index into WORDS (word stage; SLURM_ARRAY_TASK_ID)")
    args = parser.parse_args()

    pipeline = DataPipeline(
        "/home/ewong/scratch/american_stories/merged_data",
        "/home/ewong/scratch/developing-word-senses/clean_data",
        "all-mpnet-base-v2",
        1820, 1929,
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
        pipeline.plot_sense_trajectory(word, DECADES, concreteness=concreteness)
        plt.tight_layout()
        plt.savefig(pipeline.word_dir(word) / "sense_trajectory.png", dpi=150)

    elif args.stage == "plot":
        fig = plt.figure(figsize=(18, 11))
        for i, word in enumerate(WORDS):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            pipeline.plot_sense_trajectory(word, DECADES, concreteness=concreteness, ax=ax)
        plt.tight_layout()
        plt.savefig("sense_trajectories.png", dpi=150)