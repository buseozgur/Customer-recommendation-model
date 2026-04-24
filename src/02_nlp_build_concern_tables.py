"""
03 - Concern Text and Preprocessing

This script transforms raw Sephora review text into a structured
concern-aware NLP dataset.

Outputs:
- data/processed/review_text_features.parquet
- data/processed/review_concern_level.parquet
"""

import re
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


CONCERN_DICT = {
    "acne": [
        "acne", "pimple", "pimples", "breakout", "breakouts", "blemish",
        "blemishes", "clogged pores", "zits", "purging", "skin purging",
        "my skin broke out", "made me break out", "caused breakouts",
        "gave me acne", "triggered acne", "caused pimples", "made me breakout"
    ],
    "dryness": [
        "dry", "dryness", "dehydrated", "dehydration", "flaky", "flakiness",
        "tight skin", "parched", "very dry", "super dry", "made my skin dry",
        "dried out my skin", "left my skin dry", "dry lips", "dry skin"
    ],
    "oiliness": [
        "oily", "oiliness", "greasy", "shine", "shiny", "excess oil",
        "too oily", "very oily", "super oily", "made my skin oily",
        "left my skin greasy"
    ],
    "sensitivity": [
        "sensitive", "sensitivity", "reactive skin", "reactive",
        "too strong", "harsh", "stung", "stinging", "burning", "burn",
        "irritated", "irritation", "made my skin sensitive",
        "caused irritation", "burned my skin", "stings"
    ],
    "redness": [
        "redness", "flushed", "rosacea", "made me red",
        "turned my skin red", "caused redness", "left my skin red"
    ],
    "pores": [
        "pores", "pore", "large pores", "clogged pores",
        "minimize pores", "visible pores", "enlarged pores",
        "reduce pore size"
    ],
    "dark_spots": [
        "dark spot", "dark spots", "hyperpigmentation",
        "post acne marks", "acne marks", "pigmentation",
        "uneven tone", "uneven skin tone", "discoloration"
    ],
    "aging": [
        "fine lines", "wrinkles", "wrinkle", "aging", "anti aging",
        "firming", "loss of elasticity", "mature skin",
        "reduce wrinkles", "smooth fine lines"
    ],
    "dullness": [
        "dull", "dullness", "glow", "radiance", "brightening",
        "brighter skin", "lack of glow", "improve radiance", "glowing skin"
    ],
    "texture": [
        "texture", "rough skin", "smooth skin", "uneven texture",
        "bumpy skin", "skin texture", "improve texture", "refine texture"
    ],
}


CONCERN_PROTOTYPES = {
    "acne": "This review is about acne, pimples, blemishes, clogged pores, breakouts, or purging.",
    "dryness": "This review is about dryness, dehydration, flakiness, tight skin, dry skin, or dry lips.",
    "oiliness": "This review is about oiliness, greasy skin, shine, or excess oil.",
    "sensitivity": "This review is about irritation, stinging, burning, harshness, or sensitive skin.",
    "redness": "This review is about redness, flushing, rosacea, or skin turning red.",
    "pores": "This review is about pores, visible pores, clogged pores, or pore size.",
    "dark_spots": "This review is about dark spots, pigmentation, acne marks, discoloration, or uneven tone.",
    "aging": "This review is about wrinkles, fine lines, elasticity loss, or aging concerns.",
    "dullness": "This review is about dull skin, lack of glow, brightness, radiance, or brightening.",
    "texture": "This review is about uneven texture, rough skin, bumps, smoothness, or skin texture.",
}


POSITIVE_EFFECT_PATTERNS = [
    "helped", "helps", "helping", "good for", "great for", "works for",
    "improved", "improves", "improving", "reduced", "reduces", "reduce",
    "cleared", "clears", "clear up", "soothed", "soothes", "calmed",
    "calms", "hydrated", "hydrates", "moisturized", "moisturizes",
    "brightened", "brightens", "smoothed", "smooths", "minimized",
    "minimizes", "refined", "refines", "made my skin feel better"
]

NEGATIVE_EFFECT_PATTERNS = [
    "caused", "causes", "made me", "gave me", "triggered",
    "worsened", "worsens", "dried out", "drying out",
    "irritated", "irritates", "burned", "burns", "burning",
    "stung", "stings", "stinging", "clogged", "clogs",
    "broke me out", "breaks me out", "left my skin",
    "made my skin", "made it worse"
]

TARGET_ONLY_PATTERNS = [
    "i have", "i've had", "i am prone to", "my skin is",
    "for my skin", "for dry skin", "for oily skin",
    "for sensitive skin", "for acne prone skin", "for acne-prone skin",
    "i struggle with", "i deal with", "i have issues with"
]

AREA_DICT = {
    "lips": ["lip", "lips"],
    "face": ["face", "skin", "my face", "facial skin"],
    "eyes": ["eye", "eyes", "under eye", "under-eye", "eyelid", "eyelids"],
    "cheeks": ["cheek", "cheeks"],
    "forehead": ["forehead"],
    "nose": ["nose"],
    "chin": ["chin"],
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_matching(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_concern_matches(text: str, concern_dict: Dict[str, List[str]]):
    matches = []

    for concern, phrases in concern_dict.items():
        for phrase in phrases:
            phrase_clean = phrase.lower().strip()
            if phrase_clean in text:
                matches.append((concern, phrase_clean))

    seen = set()
    unique_matches = []

    for concern, phrase in matches:
        if concern not in seen:
            unique_matches.append((concern, phrase))
            seen.add(concern)

    return unique_matches


def extract_concern_labels(match_tuples):
    return list(set([concern for concern, _ in match_tuples]))


def combine_concern_labels(a: List[str], b: List[str]) -> List[str]:
    return sorted(list(set(a + b)))


def get_local_context(text: str, phrase: str, window: int = 6) -> str:
    tokens = text.split()
    phrase_tokens = phrase.split()
    n = len(phrase_tokens)

    for i in range(len(tokens) - n + 1):
        candidate = " ".join(tokens[i:i + n])
        if candidate == phrase:
            start = max(0, i - window)
            end = min(len(tokens), i + n + window)
            return " ".join(tokens[start:end])

    return text


def detect_effect_label_from_context(context: str) -> str:
    context = context.lower()

    for pattern in POSITIVE_EFFECT_PATTERNS:
        if pattern in context:
            return "helped"

    for pattern in NEGATIVE_EFFECT_PATTERNS:
        if pattern in context:
            return "worsened"

    for pattern in TARGET_ONLY_PATTERNS:
        if pattern in context:
            return "target_only"

    return "unknown"


def detect_area_from_context(context: str, area_dict: Dict[str, List[str]]) -> str:
    context = context.lower()

    for area, keywords in area_dict.items():
        for keyword in keywords:
            if keyword in context:
                return area

    return "unknown"


def compute_concern_confidence(row) -> float:
    score = 0.0

    if row["matched_by_rule_based"] == 1:
        score += 0.6
    else:
        score += 0.3

    if row["effect_label"] != "unknown":
        score += 0.2

    if pd.notna(row["matched_phrase"]):
        score += 0.2

    return min(score, 1.0)


def main():
    print("Loading review_master.parquet...")
    review_master = pd.read_parquet(PROCESSED_DIR / "review_master.parquet")
    print("review_master shape:", review_master.shape)

    df = review_master.copy()

    if "raw_text" not in df.columns:
        df["review_title"] = df["review_title"].fillna("").astype(str)
        df["review_text"] = df["review_text"].fillna("").astype(str)

        df["raw_text"] = (
            df["review_title"].str.strip()
            + " "
            + df["review_text"].str.strip()
        ).str.strip()

    df["raw_text"] = df["raw_text"].fillna("").astype(str)
    df = df[df["raw_text"].str.len() > 0].copy()

    print("Shape after removing empty raw_text rows:", df.shape)

    print("Normalizing text...")
    df["normalized_text"] = df["raw_text"].apply(normalize_text)
    df["clean_text"] = df["normalized_text"].apply(clean_text_for_matching)

    print("Running exact concern matching...")
    df["exact_matches"] = df["clean_text"].apply(
        lambda x: exact_concern_matches(x, CONCERN_DICT)
    )

    df["fuzzy_matches"] = [[] for _ in range(len(df))]
    df["rule_based_matches"] = df["exact_matches"].copy()
    df["rule_based_concerns"] = df["rule_based_matches"].apply(extract_concern_labels)

    rule_based_coverage = (df["rule_based_concerns"].apply(len) > 0).mean()
    print("Rule-based concern coverage:", round(rule_based_coverage, 4))

    print("Running semantic matching with SentenceTransformer...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    concern_names = list(CONCERN_PROTOTYPES.keys())
    prototype_texts = list(CONCERN_PROTOTYPES.values())

    prototype_embeddings = embedding_model.encode(
        prototype_texts,
        show_progress_bar=False,
    )

    review_embeddings = embedding_model.encode(
        df["normalized_text"].tolist(),
        show_progress_bar=True,
    )

    similarity_matrix = cosine_similarity(review_embeddings, prototype_embeddings)

    semantic_threshold = 0.40
    semantic_matches = []

    for row_idx in range(similarity_matrix.shape[0]):
        row_scores = similarity_matrix[row_idx]
        row_matches = []

        for concern_idx, score in enumerate(row_scores):
            if score >= semantic_threshold:
                row_matches.append((concern_names[concern_idx], float(score)))

        semantic_matches.append(row_matches)

    df["semantic_matches"] = semantic_matches
    df["semantic_concerns"] = df["semantic_matches"].apply(
        lambda matches: [concern for concern, _ in matches]
    )

    print("Combining rule-based and semantic concerns...")
    df["final_concerns"] = df.apply(
        lambda row: combine_concern_labels(
            row["rule_based_concerns"],
            row["semantic_concerns"],
        ),
        axis=1,
    )

    df["concern_count"] = df["final_concerns"].apply(len)

    final_coverage = (df["final_concerns"].apply(len) > 0).mean()
    print("Final concern coverage:", round(final_coverage, 4))

    print("Building rule-based concern-level table...")
    concern_match_records = []

    for idx, row in df.iterrows():
        text = row["clean_text"]
        rule_based_matches = row["rule_based_matches"]

        for concern, matched_phrase in rule_based_matches:
            local_context = get_local_context(text, matched_phrase, window=6)
            effect_label = detect_effect_label_from_context(local_context)
            area = detect_area_from_context(local_context, AREA_DICT)

            concern_match_records.append({
                "row_index": idx,
                "author_id": row.get("author_id"),
                "product_id": row.get("product_id"),
                "product_name_final": row.get("product_name_final"),
                "brand_name_final": row.get("brand_name_final"),
                "skin_type": row.get("skin_type"),
                "rating": row.get("rating"),
                "rating_category": row.get("rating_category"),
                "is_recommended": row.get("is_recommended"),
                "primary_category": row.get("primary_category"),
                "secondary_category": row.get("secondary_category"),
                "raw_text": row.get("raw_text"),
                "normalized_text": row.get("normalized_text"),
                "clean_text": row.get("clean_text"),
                "concern": concern,
                "matched_phrase": matched_phrase,
                "local_context": local_context,
                "effect_label": effect_label,
                "area": area,
                "matched_by_rule_based": 1,
            })

    rule_based_concern_df = pd.DataFrame(concern_match_records)
    print("rule_based_concern_df shape:", rule_based_concern_df.shape)

    print("Building semantic-only concern rows...")
    semantic_records = []

    for idx, row in df.iterrows():
        rule_based_concerns = set(row["rule_based_concerns"])
        semantic_concerns = set(row["semantic_concerns"])
        semantic_only = semantic_concerns - rule_based_concerns

        for concern in semantic_only:
            semantic_records.append({
                "row_index": idx,
                "author_id": row.get("author_id"),
                "product_id": row.get("product_id"),
                "product_name_final": row.get("product_name_final"),
                "brand_name_final": row.get("brand_name_final"),
                "skin_type": row.get("skin_type"),
                "rating": row.get("rating"),
                "rating_category": row.get("rating_category"),
                "is_recommended": row.get("is_recommended"),
                "primary_category": row.get("primary_category"),
                "secondary_category": row.get("secondary_category"),
                "raw_text": row.get("raw_text"),
                "normalized_text": row.get("normalized_text"),
                "clean_text": row.get("clean_text"),
                "concern": concern,
                "matched_phrase": None,
                "local_context": row.get("clean_text"),
                "effect_label": "unknown",
                "area": "unknown",
                "matched_by_rule_based": 0,
            })

    semantic_only_df = pd.DataFrame(semantic_records)
    print("semantic_only_df shape:", semantic_only_df.shape)

    print("Combining concern-level rows...")
    review_concern_level = pd.concat(
        [rule_based_concern_df, semantic_only_df],
        ignore_index=True,
    )

    review_concern_level["matched_by_semantic"] = 1

    review_concern_level["concern_confidence"] = review_concern_level.apply(
        compute_concern_confidence,
        axis=1,
    )

    review_concern_level = review_concern_level.drop_duplicates(
        subset=["row_index", "concern", "effect_label", "matched_phrase"]
    ).copy()

    print("review_concern_level shape:", review_concern_level.shape)
    print(review_concern_level["effect_label"].value_counts(dropna=False))
    print(review_concern_level["area"].value_counts(dropna=False))

    print("Building compact review text feature table...")
    review_text_features = df[[
        "author_id",
        "product_id",
        "product_name_final",
        "brand_name_final",
        "rating",
        "rating_category",
        "is_recommended",
        "skin_type",
        "primary_category",
        "secondary_category",
        "review_title",
        "review_text",
        "raw_text",
        "normalized_text",
        "clean_text",
        "review_text_length",
        "raw_text_length",
        "has_title",
        "rule_based_concerns",
        "semantic_concerns",
        "final_concerns",
        "concern_count",
    ]].copy()

    print("review_text_features shape:", review_text_features.shape)

    print("Saving outputs...")
    review_text_features.to_parquet(
        PROCESSED_DIR / "review_text_features.parquet",
        index=False,
    )

    review_concern_level.to_parquet(
        PROCESSED_DIR / "review_concern_level.parquet",
        index=False,
    )

    print("Saved:")
    print(PROCESSED_DIR / "review_text_features.parquet")
    print(PROCESSED_DIR / "review_concern_level.parquet")


if __name__ == "__main__":
    main()
