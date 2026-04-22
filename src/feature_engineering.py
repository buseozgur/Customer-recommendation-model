"""
Feature Engineering Module for Sephora Product Recommendation System.

Generates text, numeric, and aggregated features from raw review data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import re


# ─────────────────────────────────────────────────────────────────────────────
# TEXT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive lightweight text-based features from clean_text.

    Adds:
        text_length        – character count
        word_count         – token count
        avg_word_length    – mean chars per word
        exclamation_count  – number of '!'
        question_count     – number of '?'
        uppercase_ratio    – ratio of uppercase chars to total alpha chars
        skin_type_mention  – 1 if a skin-type word is mentioned
        concern_mention    – 1 if a common skin-concern word is mentioned
    """
    df = df.copy()

    text = df["clean_text"].fillna("")

    df["text_length"]      = text.str.len()
    df["word_count"]       = text.str.split().str.len().fillna(0).astype(int)
    df["avg_word_length"]  = text.apply(
        lambda t: np.mean([len(w) for w in t.split()]) if t.split() else 0
    )
    df["exclamation_count"] = text.str.count(r"!")
    df["question_count"]    = text.str.count(r"\?")
    df["uppercase_ratio"]   = text.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(sum(1 for c in t if c.isalpha()), 1)
    )

    skin_type_pattern = r"\b(oily|dry|combination|sensitive|normal|acne[\- ]prone|mature)\b"
    concern_pattern   = r"\b(acne|wrinkle|pore|dark spot|hyperpigmentation|redness|dull|aging|moisture|hydrat|brightening)\b"

    df["skin_type_mention"] = text.str.contains(skin_type_pattern, case=False, regex=True).astype(int)
    df["concern_mention"]   = text.str.contains(concern_pattern,   case=False, regex=True).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC / PRODUCT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add product-level numeric and categorical encoded features.

    Adds:
        price_log          – log1p of price_usd
        price_scaled       – min-max scaled price within category
        rating_deviation   – review rating minus product mean rating
        review_count       – total reviews per product
        positive_ratio     – share of positive reviews per product
    """
    df = df.copy()

    df["price_log"] = np.log1p(df["price_usd"].fillna(0))

    # scale price within primary_category
    scaler = MinMaxScaler()
    df["price_scaled"] = df.groupby("primary_category")["price_usd"].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
        if x.notna().sum() > 0 else x
    )

    # product-level aggregates
    product_stats = (
        df.groupby("product_id")
        .agg(
            product_mean_rating  = ("rating", "mean"),
            review_count         = ("rating", "count"),
            positive_ratio       = ("rating_category",
                                    lambda x: (x == "positive").sum() / max(len(x), 1))
        )
        .reset_index()
    )

    df = df.merge(product_stats, on="product_id", how="left")
    df["rating_deviation"] = df["rating"] - df["product_mean_rating"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# USER / SKIN PROFILE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

SKIN_TYPE_KEYWORDS = {
    "oily":        ["oily", "shiny", "greasy", "pore"],
    "dry":         ["dry", "flaky", "tight", "rough", "dehydrated"],
    "combination": ["combination", "t-zone", "mixed"],
    "sensitive":   ["sensitive", "reactive", "redness", "irritat"],
    "normal":      ["normal", "balanced"],
    "acne_prone":  ["acne", "breakout", "pimple", "blemish", "zit"],
}

CONCERN_KEYWORDS = {
    "anti_aging":        ["wrinkle", "aging", "fine line", "firmness", "mature"],
    "brightening":       ["dull", "bright", "glow", "dark spot", "hyperpigment", "uneven"],
    "hydration":         ["moisture", "hydrat", "dry", "plump", "dewy"],
    "pore_minimizing":   ["pore", "minimize", "refin", "tight"],
    "acne_treatment":    ["acne", "breakout", "blemish", "clarify", "salicylic"],
    "soothing":          ["rednes", "calm", "soothe", "irritat", "sensitive"],
}


def extract_skin_profile(user_input: str) -> dict:
    """
    Parse free-text user complaint / skin-type input into a structured
    skin profile dictionary.

    Args:
        user_input (str): Raw user description of skin type and concerns.

    Returns:
        dict: {
            'detected_skin_types': list[str],
            'detected_concerns':   list[str],
            'clean_input':         str
        }
    """
    text = user_input.lower()

    detected_skin_types = [
        stype for stype, kws in SKIN_TYPE_KEYWORDS.items()
        if any(kw in text for kw in kws)
    ]

    detected_concerns = [
        concern for concern, kws in CONCERN_KEYWORDS.items()
        if any(kw in text for kw in kws)
    ]

    return {
        "detected_skin_types": detected_skin_types or ["unknown"],
        "detected_concerns":   detected_concerns   or ["general"],
        "clean_input":         re.sub(r"[^a-z0-9 ]", " ", text).strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "text_length",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "question_count",
    "uppercase_ratio",
    "skin_type_mention",
    "concern_mention",
    "price_log",
    "price_scaled",
    "rating_deviation",
    "review_count",
    "positive_ratio",
    "product_mean_rating",
]

TEXT_FEATURE = "clean_text"
TARGET       = "rating_category"


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only the numeric feature columns that exist in the dataframe."""
    return [c for c in NUMERIC_FEATURES if c in df.columns]


# ─────────────────────────────────────────────────────────────────────────────
# SKLEARN-COMPATIBLE TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class SephoraFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Drop-in sklearn transformer that applies all feature-engineering steps.

    Usage:
        fe = SephoraFeatureEngineer()
        df_enriched = fe.fit_transform(df_raw)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = add_text_features(X)
        X = add_product_features(X)
        return X
