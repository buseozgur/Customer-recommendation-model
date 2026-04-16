import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


FEATURES = [
    "avg_rating",
    "rating_count",
    "avg_sentiment",
    "positive_ratio",
    "negative_ratio",
    "price",
]


def prepare_feature_matrix(df: pd.DataFrame, features: list = FEATURES) -> pd.DataFrame:
    """
    Select and prepare numeric feature matrix for clustering and PCA.

    Args:
        df (pd.DataFrame): Product summary dataframe
        features (list): Feature column names

    Returns:
        pd.DataFrame: Clean feature matrix
    """
    return df[features].fillna(0)


def scale_features(X: pd.DataFrame):
    """
    Standardize feature matrix for unsupervised learning.

    Args:
        X (pd.DataFrame): Numeric feature matrix

    Returns:
        tuple:
            scaler (StandardScaler): fitted scaler
            X_scaled (np.ndarray): scaled feature matrix
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def train_kmeans(X_scaled, n_clusters: int = 3, random_state: int = 42):
    """
    Train KMeans clustering model.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix
        n_clusters (int): Number of clusters
        random_state (int): Random seed

    Returns:
        KMeans: fitted KMeans model
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    model.fit(X_scaled)
    return model


def build_cluster_summary(df: pd.DataFrame, features: list = FEATURES) -> pd.DataFrame:
    """
    Calculate cluster-level mean summary statistics.

    Args:
        df (pd.DataFrame): Dataframe containing 'cluster' column
        features (list): Feature column names

    Returns:
        pd.DataFrame: Cluster summary table
    """
    return df.groupby("cluster")[features].mean().round(3)


def select_best_cluster(cluster_summary: pd.DataFrame) -> int:
    """
    Select the best cluster using a simple quality formula:
    high avg_rating, high avg_sentiment, high positive_ratio,
    low negative_ratio.

    Args:
        cluster_summary (pd.DataFrame): Cluster summary table

    Returns:
        int: Best cluster label
    """
    cluster_score = (
        cluster_summary["avg_rating"]
        + cluster_summary["avg_sentiment"]
        + cluster_summary["positive_ratio"]
        - cluster_summary["negative_ratio"]
    )
    return cluster_score.idxmax()


def train_pca(X_scaled, n_components: int = 1, random_state: int = 42):
    """
    Train PCA model.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix
        n_components (int): Number of PCA components
        random_state (int): Random seed

    Returns:
        PCA: fitted PCA model
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_scaled)
    return pca


def add_cluster_labels(df: pd.DataFrame, kmeans_model, X_scaled) -> pd.DataFrame:
    """
    Add cluster predictions to dataframe.

    Args:
        df (pd.DataFrame): Product summary dataframe
        kmeans_model: fitted KMeans model
        X_scaled (np.ndarray): Scaled feature matrix

    Returns:
        pd.DataFrame: Dataframe with cluster column
    """
    df = df.copy()
    df["cluster"] = kmeans_model.predict(X_scaled)
    return df


def add_pca_scores(df: pd.DataFrame, pca_model, X_scaled) -> pd.DataFrame:
    """
    Add PCA-based recommendation scores to dataframe.

    Creates:
        - pca_score_raw
        - pca_score scaled to 0-100 range

    Args:
        df (pd.DataFrame): Product summary dataframe
        pca_model: fitted PCA model
        X_scaled (np.ndarray): Scaled feature matrix

    Returns:
        pd.DataFrame: Dataframe with PCA scores
    """
    df = df.copy()

    raw_scores = pca_model.transform(X_scaled)[:, 0]
    df["pca_score_raw"] = raw_scores

    min_score = df["pca_score_raw"].min()
    max_score = df["pca_score_raw"].max()

    if max_score == min_score:
        df["pca_score"] = 50.0
    else:
        df["pca_score"] = 100 * (df["pca_score_raw"] - min_score) / (max_score - min_score)

    return df
