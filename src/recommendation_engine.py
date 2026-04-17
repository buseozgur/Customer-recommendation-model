import pandas as pd
from pathlib import Path

from src.cluster_model import (
    prepare_feature_matrix,
    scale_features,
    train_kmeans,
    train_pca,
    add_cluster_labels,
    add_pca_scores,
)


def build_scored_products_table(product_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a scored product table containing cluster labels and PCA scores
    for all products.

    Args:
        product_summary (pd.DataFrame): NLP-based product summary dataframe

    Returns:
        pd.DataFrame: Scored product table
    """
    df = product_summary.copy()

    X = prepare_feature_matrix(df)
    scaler, X_scaled = scale_features(X)

    kmeans_model = train_kmeans(X_scaled)
    df = add_cluster_labels(df, kmeans_model, X_scaled)

    pca_model = train_pca(X_scaled, n_components=1)
    df = add_pca_scores(df, pca_model, X_scaled)

    return df


def save_scored_products_table(df: pd.DataFrame, output_path: str) -> None:
    """
    Save scored product table to CSV.

    Args:
        df (pd.DataFrame): Scored dataframe
        output_path (str): Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
