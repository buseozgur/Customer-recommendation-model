import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRanker
from sentence_transformers import SentenceTransformer


class RecommendationEngine:
    def __init__(
        self,
        data_dir: str = "data/processed",
        models_dir: str = "outputs/models"
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)

        # Config
        with open(self.models_dir / "config.json", "r") as f:
            self.config = json.load(f)

        # Model load
        model_type = self.config["model_type"]
        if model_type == "lgbm":
            self.model = lgb.Booster(model_file=str(self.models_dir / "final_ranker.txt"))
        elif model_type == "xgb":
            self.model = xgb.XGBRanker()
            self.model.load_model(str(self.models_dir / "final_ranker.json"))
        else:
            self.model = CatBoostRanker()
            self.model.load_model(str(self.models_dir / "final_ranker_catboost"))

        # Encoders
        with open(self.models_dir / "label_encoders.pkl", "rb") as f:
            self.encoders = pickle.load(f)

        # Scoring table
        self.scoring_df = pd.read_parquet(self.data_dir / "ml_scoring_table.parquet")

        # SBERT
        sbert_model = self.config.get("sbert_model", "all-MiniLM-L6-v2")
        self.sbert = SentenceTransformer(sbert_model)

        # Product embeddings
        embeddings_path = self.models_dir / "product_concern_embeddings.pkl"
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                self.product_embeddings = pickle.load(f)
        else:
            self.product_embeddings = None

    def get_concerns(self) -> List[str]:
        return sorted(self.scoring_df["concern"].dropna().unique().tolist())

    def get_skin_types(self) -> List[str]:
        return sorted(self.scoring_df["skin_type"].dropna().unique().tolist())

    def _apply_filters(
        self,
        df: pd.DataFrame,
        min_reviews: int = 3
    ) -> pd.DataFrame:
        return df[df["review_count"] >= min_reviews].copy()

    def _format_results(self, df: pd.DataFrame, score_col: str) -> List[Dict]:
        results = []
        for _, row in df.iterrows():
            results.append({
                "product_id": str(row["product_id"]),
                "product_name": row.get("product_name", ""),
                "brand_name": row.get("brand_name", ""),
                "primary_category": row.get("primary_category", ""),
                "secondary_category": row.get("secondary_category", ""),
                "concern": row.get("concern", ""),
                "skin_type": row.get("skin_type", ""),
                "score": round(float(row[score_col]), 4),
                "mean_rating": round(float(row.get("mean_rating", 0)), 2),
                "helped_ratio": round(float(row.get("helped_ratio", 0)), 3),
                "review_count": int(row.get("review_count", 0)),
            })
        return results

    def recommend(
        self,
        concern: str,
        skin_type: str,
        top_n: int = 5
    ) -> List[Dict]:
        subset = self.scoring_df[
            (self.scoring_df["concern"] == concern) &
            (self.scoring_df["skin_type"] == skin_type)
        ].copy()

        if subset.empty:
            return []

        subset = self._apply_filters(subset)

        if subset.empty:
            return []

        features = self.config["features"]
        X = subset[features].values
        scores = self.model.predict(X)

        subset["model_score"] = scores
        subset = subset.sort_values("model_score", ascending=False).head(top_n)

        return self._format_results(subset, "model_score")

    def semantic_search(
        self,
        concern: str,
        skin_type: str,
        top_n: int = 5
    ) -> List[Dict]:
        if self.product_embeddings is None:
            return self.recommend(concern=concern, skin_type=skin_type, top_n=top_n)

        subset = self.scoring_df[
            (self.scoring_df["concern"] == concern) &
            (self.scoring_df["skin_type"] == skin_type)
        ].copy()

        if subset.empty:
            return []

        subset = self._apply_filters(subset)

        if subset.empty:
            return []

        query = f"I have {skin_type} skin and I want a product that helps with {concern}."
        query_embedding = self.sbert.encode([query], normalize_embeddings=True)[0]

        semantic_scores = []
        for _, row in subset.iterrows():
            key = (row["product_id"], row["concern"])
            if key in self.product_embeddings:
                prod_emb = self.product_embeddings[key]
                similarity = float(np.dot(query_embedding, prod_emb))
                semantic_scores.append(similarity)
            else:
                semantic_scores.append(0.0)

        subset["semantic_score"] = semantic_scores

        features = self.config["features"]
        X = subset[features].values
        subset["model_score"] = self.model.predict(X)

        # normalize
        subset["model_norm"] = (
            (subset["model_score"] - subset["model_score"].min()) /
            (subset["model_score"].max() - subset["model_score"].min() + 1e-9)
        )
        subset["semantic_norm"] = (
            (subset["semantic_score"] - subset["semantic_score"].min()) /
            (subset["semantic_score"].max() - subset["semantic_score"].min() + 1e-9)
        )

        weights = self.config.get(
            "ensemble_weights",
            {"w_model": 0.7, "w_semantic": 0.3}
        )
        w_model = weights.get("w_model", 0.7)
        w_semantic = weights.get("w_semantic", 0.3)

        subset["final_score"] = (w_model * subset["model_norm"]) + (w_semantic * subset["semantic_norm"])
        subset = subset.sort_values("final_score", ascending=False).head(top_n)

        return self._format_results(subset, "final_score")


_engine = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


def get_concerns() -> List[str]:
    return get_engine().get_concerns()


def get_skin_types() -> List[str]:
    return get_engine().get_skin_types()


def recommend_products(concern: str, skin_type: str, top_n: int = 5) -> List[Dict]:
    return get_engine().semantic_search(concern=concern, skin_type=skin_type, top_n=top_n)
