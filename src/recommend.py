import os
import io
import json
import pickle
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
from google.cloud import storage
from sentence_transformers import SentenceTransformer


BUCKET_NAME = os.getenv(
    "BUCKET_NAME",
    "sephora-customer-recommendation-model-2026"
)

ARTIFACT_PREFIX = os.getenv(
    "ARTIFACT_PREFIX",
    "artifacts"
)


class RecommendationEngine:
    def __init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(BUCKET_NAME)

        self.config = self._load_json("config.json")
        self.encoders = self._load_pickle("label_encoders.pkl")
        self.scoring_df = self._load_parquet("ml_scoring_table.parquet")

        self.product_embeddings = None
        if self._blob_exists("product_concern_embeddings.pkl"):
            self.product_embeddings = self._load_pickle("product_concern_embeddings.pkl")

        sbert_model = self.config.get("sbert_model", "all-MiniLM-L6-v2")
        self.sbert = SentenceTransformer(sbert_model)

        self.model = self._load_model()

    def _blob_path(self, filename: str) -> str:
        return f"{ARTIFACT_PREFIX}/{filename}"

    def _blob_exists(self, filename: str) -> bool:
        return self.bucket.blob(self._blob_path(filename)).exists()

    def _download_bytes(self, filename: str) -> bytes:
        blob = self.bucket.blob(self._blob_path(filename))
        return blob.download_as_bytes()

    def _load_json(self, filename: str) -> Dict:
        return json.loads(self._download_bytes(filename).decode("utf-8"))

    def _load_pickle(self, filename: str):
        return pickle.loads(self._download_bytes(filename))

    def _load_parquet(self, filename: str) -> pd.DataFrame:
        data = self._download_bytes(filename)
        return pd.read_parquet(io.BytesIO(data))

    def _load_model(self):
        model_type = self.config["model_type"]

        if model_type == "lgbm":
            model_bytes = self._download_bytes("final_ranker.txt")
            local_path = "/tmp/final_ranker.txt"
            with open(local_path, "wb") as f:
                f.write(model_bytes)
            return lgb.Booster(model_file=local_path)

        if model_type == "xgb":
            model_bytes = self._download_bytes("final_ranker.json")
            local_path = "/tmp/final_ranker.json"
            with open(local_path, "wb") as f:
                f.write(model_bytes)

            model = xgb.XGBRanker()
            model.load_model(local_path)
            return model

        model_bytes = self._download_bytes("final_ranker_catboost")
        local_path = "/tmp/final_ranker_catboost"
        with open(local_path, "wb") as f:
            f.write(model_bytes)

        model = CatBoostRanker()
        model.load_model(local_path)
        return model

    def get_concerns(self) -> List[str]:
        return sorted(self.scoring_df["concern"].dropna().unique().tolist())

    def get_skin_types(self) -> List[str]:
        return sorted(self.scoring_df["skin_type"].dropna().unique().tolist())

    def get_categories(self) -> List[str]:
        return sorted(self.scoring_df["secondary_category"].dropna().unique().tolist())

    def _apply_filters(
        self,
        df: pd.DataFrame,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_reviews: int = 3,
    ) -> pd.DataFrame:
        df = df[df["review_count"] >= min_reviews].copy()

        if category:
            df = df[df["secondary_category"] == category]

        if "price_usd_final" in df.columns:
            if min_price is not None:
                df = df[df["price_usd_final"] >= min_price]
            if max_price is not None:
                df = df[df["price_usd_final"] <= max_price]

        return df

    def _normalize(self, series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    def _format_results(self, df: pd.DataFrame, score_col: str) -> List[Dict]:
        results = []

        for _, row in df.iterrows():
            result = {
                "product_id": str(row.get("product_id", "")),
                "product_name": row.get("product_name", ""),
                "brand_name": row.get("brand_name", ""),
                "primary_category": row.get("primary_category", ""),
                "secondary_category": row.get("secondary_category", ""),
                "concern": row.get("concern", ""),
                "skin_type": row.get("skin_type", ""),
                "score": round(float(row.get(score_col, 0)), 4),
                "mean_rating": round(float(row.get("mean_rating", 0)), 2),
                "helped_ratio": round(float(row.get("helped_ratio", 0)), 3),
                "review_count": int(row.get("review_count", 0)),
            }

            if "price_usd_final" in row:
                result["price"] = round(float(row.get("price_usd_final", 0)), 2)
            else:
                result["price"] = 0.0

            results.append(result)

        return results

    def recommend(
        self,
        concern: str,
        skin_type: str,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        top_n: int = 5,
    ) -> List[Dict]:
        subset = self.scoring_df[
            (self.scoring_df["concern"] == concern)
            & (self.scoring_df["skin_type"] == skin_type)
        ].copy()

        if subset.empty:
            return []

        subset = self._apply_filters(
            subset,
            category=category,
            min_price=min_price,
            max_price=max_price,
        )

        if subset.empty:
            return []

        features = self.config["features"]
        subset["model_score"] = self.model.predict(subset[features].values)

        subset["review_count_log"] = np.log1p(subset["review_count"])
        subset["model_norm"] = self._normalize(subset["model_score"])
        subset["review_count_norm"] = self._normalize(subset["review_count_log"])

        weights = self.config.get(
            "ensemble_weights",
            {"w_model": 0.85, "w_reviews": 0.15},
        )

        w_model = weights.get("w_model", 0.85)
        w_reviews = weights.get("w_reviews", 0.15)

        subset["final_score"] = (
            w_model * subset["model_norm"]
            + w_reviews * subset["review_count_norm"]
        )

        subset = subset.sort_values("final_score", ascending=False).head(top_n)

        return self._format_results(subset, "final_score")

    def semantic_search(
        self,
        concern: str,
        skin_type: str,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        top_n: int = 5,
    ) -> List[Dict]:
        subset = self.scoring_df[
            (self.scoring_df["concern"] == concern)
            & (self.scoring_df["skin_type"] == skin_type)
        ].copy()

        if subset.empty:
            return []

        subset = self._apply_filters(
            subset,
            category=category,
            min_price=min_price,
            max_price=max_price,
        )

        if subset.empty:
            return []

        features = self.config["features"]
        subset["model_score"] = self.model.predict(subset[features].values)

        subset["review_count_log"] = np.log1p(subset["review_count"])
        subset["model_norm"] = self._normalize(subset["model_score"])
        subset["review_count_norm"] = self._normalize(subset["review_count_log"])

        if self.product_embeddings is None:
            weights = self.config.get(
                "ensemble_weights",
                {"w_model": 0.85, "w_reviews": 0.15},
            )

            w_model = weights.get("w_model", 0.85)
            w_reviews = weights.get("w_reviews", 0.15)

            subset["final_score"] = (
                w_model * subset["model_norm"]
                + w_reviews * subset["review_count_norm"]
            )

            subset = subset.sort_values("final_score", ascending=False).head(top_n)
            return self._format_results(subset, "final_score")

        query = (
            f"I have {skin_type} skin and I want a product "
            f"that helps with {concern}."
        )
        query_embedding = self.sbert.encode([query], normalize_embeddings=True)[0]

        semantic_scores = []
        for _, row in subset.iterrows():
            key = (row["product_id"], row["concern"])
            prod_emb = self.product_embeddings.get(key)

            if prod_emb is None:
                semantic_scores.append(0.0)
            else:
                semantic_scores.append(float(np.dot(query_embedding, prod_emb)))

        subset["semantic_score"] = semantic_scores
        subset["semantic_norm"] = self._normalize(subset["semantic_score"])

        weights = self.config.get(
            "ensemble_weights",
            {
                "w_model": 0.60,
                "w_semantic": 0.25,
                "w_reviews": 0.15,
            },
        )

        w_model = weights.get("w_model", 0.60)
        w_semantic = weights.get("w_semantic", 0.25)
        w_reviews = weights.get("w_reviews", 0.15)

        subset["final_score"] = (
            w_model * subset["model_norm"]
            + w_semantic * subset["semantic_norm"]
            + w_reviews * subset["review_count_norm"]
        )

        subset = subset.sort_values("final_score", ascending=False).head(top_n)

        return self._format_results(subset, "final_score")


_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


def get_concerns() -> List[str]:
    return get_engine().get_concerns()


def get_skin_types() -> List[str]:
    return get_engine().get_skin_types()


def get_categories() -> List[str]:
    return get_engine().get_categories()


def recommend_products(
    concern: str,
    skin_type: str,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    top_n: int = 5,
) -> List[Dict]:
    return get_engine().semantic_search(
        concern=concern,
        skin_type=skin_type,
        category=category,
        min_price=min_price,
        max_price=max_price,
        top_n=top_n,
    )
