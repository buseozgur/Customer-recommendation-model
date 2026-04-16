import os
from io import StringIO

import pandas as pd
from google.cloud import storage


BUCKET_NAME = os.getenv("BUCKET_NAME", "sephora-customer-recommendation-model-2026")
BLOB_NAME = os.getenv("BLOB_NAME", "artifacts/ml_recommendation_table.csv")


def load_recommendation_table() -> pd.DataFrame:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_NAME)

    csv_data = blob.download_as_text()
    return pd.read_csv(StringIO(csv_data))


def get_primary_categories():
    df = load_recommendation_table()
    return sorted(df["primary_category"].dropna().unique().tolist())


def get_secondary_categories(primary_category: str):
    df = load_recommendation_table()
    filtered = df[df["primary_category"] == primary_category]
    return sorted(filtered["secondary_category"].dropna().unique().tolist())


def select_best_cluster(filtered_df: pd.DataFrame) -> int:
    cluster_summary = filtered_df.groupby("cluster").agg(
        avg_rating=("avg_rating", "mean"),
        avg_sentiment=("avg_sentiment", "mean"),
        positive_ratio=("positive_ratio", "mean"),
        negative_ratio=("negative_ratio", "mean")
    )

    cluster_score = (
        cluster_summary["avg_rating"]
        + cluster_summary["avg_sentiment"]
        + cluster_summary["positive_ratio"]
        - cluster_summary["negative_ratio"]
    )

    return cluster_score.idxmax()


def recommend_top_products(primary_category: str, secondary_category: str, top_n: int = 5):
    df = load_recommendation_table()

    filtered = df[
        (df["primary_category"] == primary_category) &
        (df["secondary_category"] == secondary_category)
    ].copy()

    if filtered.empty:
        return []

    best_cluster = select_best_cluster(filtered)

    final_df = (
        filtered[filtered["cluster"] == best_cluster]
        .sort_values(["pca_score", "avg_rating", "rating_count"], ascending=[False, False, False])
        .head(top_n)
    )

    return final_df[
        [
            "product_name",
            "brand_name",
            "avg_rating",
            "rating_count",
            "avg_sentiment",
            "positive_ratio",
            "negative_ratio",
            "price",
            "cluster",
            "pca_score",
        ]
    ].to_dict(orient="records")
