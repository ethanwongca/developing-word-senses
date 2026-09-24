import pathlib as Path
import re
import pandas as pd 
from typing import Optional
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

class DataPipeline:
    def __init__(self, data_dir: str, result_dir: str, model: str):
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.model = SentenceTransformer(model)
        self.store_embeddings = []

    @staticmethod
    def _clean_ocr_text(text: str, basic: bool = False) -> str:
        """
        Cleans OCR text.

        - Removes '-\\n' line-wrap hyphenation.
        - Replaces all remaining newlines with spaces.
        - Collapses multiple whitespace into a single space.
        - When basic=False, also replaces é→e, ï→i, ﬁ→fi, and ﬂ→fl.
        """
        if not isinstance(text, str):
            return text

        text = text.lower()
        text = text.replace("-\n", "")

        if not basic:
            text = (text.replace("é", "e")
                        .replace("ï", "i")
                        .replace("ﬁ", "fi")
                        .replace("ﬂ", "fl"))

        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()

    def clean_pipeline(self):
        self.result_dir.mkdir(parents=True, exist_ok=True)
        for file in self.data_dir.glob("*.parquet"):
            df = pd.read_parquet(file)
            df["article"] = df["article"].map(self._clean_ocr_text)
            df.to_parquet(self.result_dir / file.name)
 

    def get_embeddings(self, word: str, start_year: int, end_year: int, region: Optional[str] = None):
        pattern = re.compile(rf"\b{re.escape(word.lower())}\b")
        cols = ["article"] + (["region"] if region else [])
        sentences_with_word = []

        for year in range(start_year, end_year + 1):
            year_df = pd.read_parquet(self.result_dir / f"{year}.parquet", columns=cols)
            if region:
                year_df = year_df[year_df["region"] == region]
            for article in year_df["article"].dropna():
                for sentence in sent_tokenize(article):
                    if pattern.search(sentence):
                        sentences_with_word.append(pattern.sub("[MASK]", sentence))

        if not sentences_with_word:
            return np.zeros(self.model.get_sentence_embedding_dimension())

        embeddings = self.model.encode(sentences_with_word, batch_size=64)
        return embeddings.mean(axis=0)

    
        


            



            
