import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def prepare_sentiment_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for sentiment model training.

    Keeps only the columns needed for NLP modeling and filters
    to positive/negative classes.

    Args:
        df (pd.DataFrame): Input dataframe containing clean_text and rating_category

    Returns:
        pd.DataFrame: Filtered dataframe for sentiment training
    """
    df_model = df[[
        "product_id",
        "brand_name",
        "product_name",
        "primary_category",
        "secondary_category",
        "clean_text",
        "rating",
        "price_usd",
        "rating_category"
    ]].copy()

    df_model = df_model[df_model["rating_category"].isin(["positive", "negative"])].copy()

    return df_model


def build_sentiment_pipeline(
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    max_iter: int = 1000
) -> Pipeline:
    """
    Create TF-IDF + Logistic Regression pipeline for sentiment classification.

    Args:
        max_features (int): Maximum number of TF-IDF features
        ngram_range (tuple): N-gram range for TF-IDF
        max_iter (int): Maximum iterations for Logistic Regression

    Returns:
        Pipeline: Scikit-learn pipeline
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=max_features,
            ngram_range=ngram_range
        )),
        ("model", LogisticRegression(max_iter=max_iter))
    ])

    return pipeline


def train_sentiment_model(
    df_model: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Train sentiment classification model.

    Args:
        df_model (pd.DataFrame): Prepared sentiment dataframe
        test_size (float): Test split ratio
        random_state (int): Random seed

    Returns:
        tuple:
            pipeline (Pipeline): Trained model
            X_test (pd.Series): Test texts
            y_test (pd.Series): Test labels
            y_pred (pd.Series): Predicted labels on test set
    """
    X = df_model["clean_text"]
    y = df_model["rating_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_sentiment_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    return pipeline, X_test, y_test, y_pred


def predict_sentiment(df_model: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    """
    Add sentiment predictions and sentiment scores to the dataframe.

    sentiment_pred:
        predicted class label (positive / negative)

    sentiment_score:
        probability of the positive class

    Args:
        df_model (pd.DataFrame): Sentiment modeling dataframe
        pipeline (Pipeline): Trained sentiment pipeline

    Returns:
        pd.DataFrame: Dataframe with sentiment_pred and sentiment_score
    """
    df_model = df_model.copy()

    df_model["sentiment_pred"] = pipeline.predict(df_model["clean_text"])
    df_model["sentiment_score"] = pipeline.predict_proba(df_model["clean_text"])[:, 1]

    return df_model
