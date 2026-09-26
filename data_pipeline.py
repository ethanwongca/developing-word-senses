import json
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
from sklearn.decomposition import PCA


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
        self.sentences_path = self.result_dir / "sentences.jsonl"
        self.embeddings_path = self.result_dir / "embeddings.parquet"

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

    def clean_pipeline(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        for file in self.data_dir.glob("*.parquet"):
            if self.start_year <= int(file.stem) <= self.end_year:
                df = pd.read_parquet(file)
                df["article"] = df["article"].map(self._clean_ocr_text)
                df.to_parquet(self.result_dir / file.name)

    def extract_sentences(self, words: list[str], region_col: str = "region"):
        """
        Single pass over the cleaned corpus. For every sentence containing any target
        word, write {word, year, region, sentence} as one JSONL line. Streaming write,
        so memory use stays flat regardless of corpus size.
        """
        patterns = {w: re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in words}

        with open(self.sentences_path, "w") as out:
            for year in range(self.start_year, self.end_year + 1):
                path = self.result_dir / f"{year}.parquet"
                if not path.exists():
                    continue
                df = pd.read_parquet(path, columns=["article", region_col])
                for article, region in zip(df["article"], df[region_col]):
                    if not isinstance(article, str):
                        continue
                    for sentence in sent_tokenize(article):
                        for word, pattern in patterns.items():
                            if pattern.search(sentence):
                                masked = pattern.sub(self.mask_token, sentence)
                                row = {"word": word, "year": year, "region": region, "sentence": masked}
                                out.write(json.dumps(row) + "\n")
                del df

    def embed_sentences(self, batch_size: int = 64):
        """
        Stream sentences.jsonl, encode in fixed-size batches, and append each batch's
        (word, year, region, embedding) rows to embeddings.parquet. Never holds more
        than one batch of text/embeddings in memory.
        """
        writer = None
        buffer = []

        def flush():
            nonlocal writer
            if not buffer:
                return
            embeddings = self.model.encode(
                [r["sentence"] for r in buffer], batch_size=batch_size, normalize_embeddings=True
            )
            table = pa.table({
                "word": [r["word"] for r in buffer],
                "year": [r["year"] for r in buffer],
                "region": [r["region"] for r in buffer],
                "embedding": [e.tolist() for e in embeddings],
            })
            if writer is None:
                writer = pq.ParquetWriter(self.embeddings_path, table.schema)
            writer.write_table(table)
            buffer.clear()

        with open(self.sentences_path) as f:
            for line in f:
                buffer.append(json.loads(line))
                if len(buffer) == batch_size:
                    flush()
        flush()
        if writer:
            writer.close()

    def _load_embeddings(
        self, word: str, start_year: int, end_year: int, region: Optional[str] = None
    ) -> Optional[np.ndarray]:
        filters = [("word", "==", word), ("year", ">=", start_year), ("year", "<=", end_year)]
        if region is not None:
            filters.append(("region", "==", region))
        df = pd.read_parquet(self.embeddings_path, filters=filters, columns=["embedding"])
        return None if df.empty else np.vstack(df["embedding"].to_numpy())

    def get_embeddings(
        self, word: str, start_year: int, end_year: int, region: Optional[str] = None
    ) -> Optional[np.ndarray]:
        emb = self._load_embeddings(word, start_year, end_year, region)
        return None if emb is None else emb.mean(axis=0)

    def plot_trajectory(self, word, periods, region=None, max_points=300, ax=None):
        data = []
        rng = np.random.default_rng(0)
        for start, end in periods:
            emb = self._load_embeddings(word, start, end, region)
            if emb is None:
                continue
            if len(emb) > max_points:
                emb = emb[rng.choice(len(emb), max_points, replace=False)]
            data.append((start, emb))

        if ax is None:
            ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
        ax.set_title(f"'{word}' — {region or 'all'}")
        if len(data) < 2:
            ax.set_title(f"'{word}' — {region or 'all'} (not enough data)")
            return ax

        pca = PCA(n_components=2).fit(np.vstack([e for _, e in data]))
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(data)))
        centroids = []
        for (start, emb), color in zip(data, colors):
            pcs = pca.transform(emb)
            ax.scatter(np.full(len(pcs), start), pcs[:, 0], pcs[:, 1], s=4, alpha=0.4, color=color)
            centroids.append([start, *pcs.mean(axis=0)])

        c = np.array(centroids)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], "k-o", markersize=4)
        ax.set_xlabel("Year")
        ax.set_ylabel("PC 1")
        ax.set_zlabel("PC 2")
        return ax


if __name__ == "__main__":
    pipeline = DataPipeline(
        "/home/ewong/scratch/american_stories/merged_data",
        "/home/ewong/scratch/developing-word-senses/clean_data",
        "all-mpnet-base-v2",
        1860, 1929,
    )
    pipeline.clean_pipeline()

    words = ["gay", "cool", "broadcast", "car", "square", "hot"]
    pipeline.extract_sentences(words)   # one corpus pass, all words at once
    pipeline.embed_sentences()          # streamed, appended to embeddings.parquet

    decades = [(y, y + 9) for y in range(1860, 1921, 10)]
    states = ["Virginia", "Maine", "California", "District of Columbia"]
    fig = plt.figure(figsize=(5 * len(states), 4 * len(words)))
    for i, word in enumerate(words):
        for j, state in enumerate(states):
            ax = fig.add_subplot(len(words), len(states), i * len(states) + j + 1, projection="3d")
            pipeline.plot_trajectory(word, decades, region=state, ax=ax)

    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=150)