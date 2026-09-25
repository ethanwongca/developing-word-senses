import re
import unicodedata
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


class DataPipeline:
    def __init__(self, data_dir: str, result_dir: str, model: str):
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.model = SentenceTransformer(model)
        # Use the model's own mask token ("[MASK]" for BERT, "<mask>" for MPNet/RoBERTa)
        self.mask_token = self.model.tokenizer.mask_token
        if self.mask_token is None:
            raise ValueError(f"Model '{model}' has no mask token.")

    @staticmethod
    def _clean_ocr_text(text: str, basic: bool = False) -> str:
        """
        Cleans OCR text. (From Deja Vu but simplified)

        - Removes line-wrap hyphenation ('-\\n', tolerant of spaces and '\\r\\n').
        - Collapses all whitespace (incl. newlines) into a single space.
        - When basic=False, also expands ligatures (NFKC) and strips accents.
        - Case is preserved: sent_tokenize relies on capitalization.
        """
        if not isinstance(text, str):
            return text

        text = re.sub(r"-[ \t]*\r?\n[ \t]*", "", text)

        if not basic:
            text = unicodedata.normalize("NFKC", text)
            text = "".join(
                c for c in unicodedata.normalize("NFKD", text)
                if not unicodedata.combining(c)
            )

        return re.sub(r"\s+", " ", text).strip()

    def clean_pipeline(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        for file in self.data_dir.glob("*.parquet"):
            df = pd.read_parquet(file)
            df["article"] = df["article"].map(self._clean_ocr_text)
            df.to_parquet(self.result_dir / file.name)

    def get_sentence_embeddings(
        self, word: str, start_year: int, end_year: int, region: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """One embedding per sentence containing `word` (word replaced by the mask token)."""
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        cols = ["article"] + (["region"] if region is not None else [])
        sentences = []

        for year in range(start_year, end_year + 1):
            path = self.result_dir / f"{year}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=cols)
            if region is not None:
                df = df[df["region"] == region]
            for article in df["article"].dropna():
                if not pattern.search(article):
                    continue
                for sentence in sent_tokenize(article):
                    if pattern.search(sentence):
                        sentences.append(pattern.sub(self.mask_token, sentence))

        if not sentences:
            return None
        return self.model.encode(sentences, batch_size=64, normalize_embeddings=True)

    def get_embeddings(
        self, word: str, start_year: int, end_year: int, region: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Mean embedding over all matching sentences."""
        emb = self.get_sentence_embeddings(word, start_year, end_year, region)
        return None if emb is None else emb.mean(axis=0)

    def plot_trajectory(self, word, periods, region=None, max_points=300, ax=None):
        """3D plot: time vs. PC1 vs. PC2, one point cloud per period, line through centroids."""
        data = []
        rng = np.random.default_rng(0)
        for start, end in periods:
            emb = self.get_sentence_embeddings(word, start, end, region)
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
    pipeline = DataPipeline("/home/ewong/scratch/american_stories/merged_data", "/home/ewong/scratch/developing-word-senses/clean_data", "all-mpnet-base-v2")
    pipeline.clean_pipeline()

    decades = [(y, y + 9) for y in range(1860, 1921, 10)] 
    words = ["gay", "cool", "broadcast", "car", "square", "hot"]
    states = ["Virginia", "Maine", "California", "District of Columbia"]
    fig = plt.figure(figsize=(5 * len(states), 4 * len(words)))
    for i, word in enumerate(words):
        for j, state in enumerate(states):
            ax = fig.add_subplot(len(words), len(states), i * len(states) + j + 1, projection="3d")
            pipeline.plot_trajectory(word, decades, region=state, ax=ax)

    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=150)