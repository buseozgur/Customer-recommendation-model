from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# Configuration
# =========================================================

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_THRESHOLD = 0.40


# =========================================================
# Dictionaries and patterns
# =========================================================

CONCERN_DICT: Dict[str, List[str]] = {
    "acne": [
        "acne", "pimple", "pimples", "breakout", "breakouts", "blemish", "blemishes",
        "clogged pores", "zits", "purging", "skin purging",
        "my skin broke out", "made me break out", "caused breakouts",
        "gave me acne", "triggered acne", "caused pimples", "made me breakout"
    ],
    "dryness": [
        "dry", "dryness", "dehydrated", "dehydration", "flaky", "flakiness",
        "tight skin", "parched", "very dry", "super dry",
        "made my skin dry", "dried out my skin", "left my skin dry",
        "dry lips", "dry skin"
    ],
    "oiliness": [
        "oily", "oiliness", "greasy", "shine", "shiny", "excess oil",
        "too oily", "very oily", "super oily",
        "made my skin oily", "left my skin greasy"
    ],
    "sensitivity": [
        "sensitive", "sensitivity", "reactive skin", "reactive",
        "too strong", "harsh", "stung", "stinging", "burning", "burn",
        "irritated", "irritation",
        "made my skin sensitive", "caused irritation", "burned my skin", "stings"
    ],
    "redness": [
        "redness", "flushed", "rosacea",
        "made me red", "turned my skin red", "caused redness",
        "left my skin red"
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
        "dull", "dullness", "glow", "radiance",
        "brightening", "brighter skin",
        "lack of glow", "improve radiance", "glowing skin"
    ],
    "texture": [
        "texture", "rough skin", "smooth skin",
        "uneven texture", "bumpy skin",
        "skin texture", "improve texture", "refine texture"
    ],
}

CONCERN_PROTOTYPES: Dict[str, str] = {
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
    "helped", "helps", "helping",
    "good for", "great for", "works for",
    "improved", "improves", "improving",
    "reduced", "reduces", "reduce",
    "cleared", "clears", "clear up",
    "soothed", "soothes", "calmed", "calms",
    "hydrated", "hydrates", "moisturized", "moisturizes",
    "brightened", "brightens",
    "smoothed", "smooths",
    "minimized", "minimizes",
    "refined", "refines",
    "made my skin feel better",
]

NEGATIVE_EFFECT_PATTERNS = [
    "caused", "causes",
    "made me", "gave me", "triggered",
    "worsened", "worsens",
    "dried out", "drying out",
    "irritated", "irritates",
    "burned", "burns", "burning",
    "stung", "stings", "stinging",
    "clogged", "clogs",
    "broke me out", "breaks me out",
    "left my skin", "made my skin",
    "made it worse",
]

TARGET_ONLY_PATTERNS = [
    "i have", "i've had", "i am prone to",
    "my skin is", "for my skin",
    "for dry skin", "for oily skin", "for sensitive skin",
    "for acne prone skin", "for acne-prone skin",
    "i struggle with", "i deal with", "i have issues with",
]

AREA_DICT: Dict[str, List[str]] = {
    "lips": ["lip", "lips"],
    "face": ["face", "skin", "my face", "facial skin"],
    "eyes": ["eye", "eyes", "under eye", "under-eye", "eyelid", "eyelids"],
    "cheeks": ["cheek", "cheeks"],
    "forehead": ["forehead"],
    "nose": ["nose"],
    "chin": ["chin"],
}


# =========================================================
# Text utilities
# =========================================================

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


def build_raw_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "raw_text" not in df.columns:
        df["review_title"] = df["review_title"].fillna("").astype(str)
        df["review_text"] = df["review_text"].fillna("").astype(str)
        df["raw_text"] = (
            df["review_title"].str.strip() + " " + df["review_text"].str.strip()
        ).str.strip()

    df["raw_text"] = df["raw_text"].fillna("").astype(str)
    df = df[df["raw_text"].str.len() > 0].copy()

    df["normalized_text"] = df["raw_text"].apply(normalize_text)
    df["clean_text"] = df["normalized_text"].apply(clean_text_for_matching)

    return df


# =========================================================
# Rule-based concern extraction
# =========================================================

def exact_concern_matches(text: str, concern_dict: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    matches: List[Tuple[str, str]] = []

    for concern, phrases in concern_dict.items():
        for phrase in phrases:
            phrase_clean = phrase.lower().strip()
            if phrase_clean in text:
                matches.append((concern, phrase_clean))

    # Keep only the first matched phrase per concern
    seen = set()
    unique_matches = []
    for concern, phrase in matches:
        if concern not in seen:
            unique_matches.append((concern, phrase))
            seen.add(concern)

    return unique_matches


def extract_concern_labels(match_tuples: List[Tuple[str, str]]) -> List[str]:
    return sorted(list({concern for concern, _ in match_tuples}))


def add_rule_based_concerns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["exact_matches"] = df["clean_text"].apply(
        lambda x: exact_concern_matches(x, CONCERN_DICT)
    )

    # Fuzzy is intentionally disabled in this script version
    df["fuzzy_matches"] = [[] for _ in range(len(df))]

    df["rule_based_matches"] = df["exact_matches"].copy()
    df["rule_based_concerns"] = df["rule_based_matches"].apply(extract_concern_labels)

    return df


# =========================================================
# Semantic concern enrichment
# =========================================================

def add_semantic_concerns(
    df: pd.DataFrame,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    batch_size: int = 128,
) -> pd.DataFrame:
    df = df.copy()

    model = SentenceTransformer(model_name)

    concern_names = list(CONCERN_PROTOTYPES.keys())
    prototype_texts = list(CONCERN_PROTOTYPES.values())

    prototype_embeddings = model.encode(
        prototype_texts,
        show_progress_bar=False,
        batch_size=batch_size,
    )

    # Only encode rows that have no rule-based concerns
    no_rule_mask = df["rule_based_concerns"].apply(len) == 0

    df["semantic_matches"] = [[] for _ in range(len(df))]
    df["semantic_concerns"] = [[] for _ in range(len(df))]

    if no_rule_mask.sum() == 0:
        df["final_concerns"] = df["rule_based_concerns"]
        df["concern_count"] = df["final_concerns"].apply(len)
        return df

    subset_texts = df.loc[no_rule_mask, "normalized_text"].tolist()

    review_embeddings = model.encode(
        subset_texts,
        show_progress_bar=True,
        batch_size=batch_size,
    )

    similarity_matrix = cosine_similarity(review_embeddings, prototype_embeddings)

    subset_semantic_matches: List[List[Tuple[str, float]]] = []

    for row_idx in range(similarity_matrix.shape[0]):
        row_scores = similarity_matrix[row_idx]
        row_matches: List[Tuple[str, float]] = []

        for concern_idx, score in enumerate(row_scores):
            if score >= semantic_threshold:
                row_matches.append((concern_names[concern_idx], float(score)))

        subset_semantic_matches.append(row_matches)

    subset_semantic_concerns = [
        [concern for concern, _ in matches]
        for matches in subset_semantic_matches
    ]

    df.loc[no_rule_mask, "semantic_matches"] = subset_semantic_matches
    df.loc[no_rule_mask, "semantic_concerns"] = subset_semantic_concerns

    df["final_concerns"] = df.apply(
        lambda row: sorted(list(set(row["rule_based_concerns"] + row["semantic_concerns"]))),
        axis=1,
    )
    df["concern_count"] = df["final_concerns"].apply(len)

    return df


# =========================================================
# Effect and area extraction
# =========================================================

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
        for kw in keywords:
            if kw in context:
                return area

    return "unknown"


# =========================================================
# Output tables
# =========================================================

def build_review_text_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
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
    ]

    existing_cols = [c for c in cols if c in df.columns]
    return df[existing_cols].copy()


def build_review_concern_level(df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for idx, row in df.iterrows():
        text = row["clean_text"]
        rule_based_matches = row["rule_based_matches"]
        rule_based_concerns = set(row["rule_based_concerns"])
        semantic_concerns = set(row["semantic_concerns"])

        # Rule-based rows
        for concern, matched_phrase in rule_based_matches:
            local_context = get_local_context(text, matched_phrase, window=6)
            effect_label = detect_effect_label_from_context(local_context)
            area = detect_area_from_context(local_context, AREA_DICT)

            records.append({
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
                "matched_by_semantic": 0,
            })

        # Semantic-only rows
        semantic_only = semantic_concerns - rule_based_concerns

        for concern in semantic_only:
            records.append({
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
                "matched_by_semantic": 1,
            })

    review_concern_level = pd.DataFrame(records)

    if review_concern_level.empty:
        return review_concern_level

    review_concern_level = review_concern_level.drop_duplicates(
        subset=["row_index", "concern", "effect_label", "matched_phrase"]
    ).copy()

    review_concern_level["concern_confidence"] = review_concern_level.apply(
        compute_concern_confidence,
        axis=1,
    )

    return review_concern_level


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


# =========================================================
# Save utilities
# =========================================================

def save_parquet(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


# =========================================================
# Main pipeline
# =========================================================

def build_concern_outputs(
    review_master_path: str | Path,
    review_text_features_output: str | Path,
    review_concern_level_output: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    batch_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build concern-aware NLP outputs from the review master table.

    Steps:
        1. Load review master
        2. Build raw / normalized / clean text
        3. Add rule-based concerns
        4. Add semantic concerns (only for no-rule-based rows)
        5. Build review_text_features
        6. Build review_concern_level
        7. Save outputs

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            review_text_features and review_concern_level
    """
    df = pd.read_parquet(review_master_path)

    df = build_raw_text(df)
    df = add_rule_based_concerns(df)
    df = add_semantic_concerns(
        df,
        model_name=model_name,
        semantic_threshold=semantic_threshold,
        batch_size=batch_size,
    )

    review_text_features = build_review_text_features(df)
    review_concern_level = build_review_concern_level(df)

    save_parquet(review_text_features, review_text_features_output)
    save_parquet(review_concern_level, review_concern_level_output)

    return review_text_features, review_concern_level


if __name__ == "__main__":
    review_master_path = "../data/processed/review_master.parquet"
    review_text_features_output = "../data/processed/review_text_features.parquet"
    review_concern_level_output = "../data/processed/review_concern_level.parquet"

    review_text_features, review_concern_level = build_concern_outputs(
        review_master_path=review_master_path,
        review_text_features_output=review_text_features_output,
        review_concern_level_output=review_concern_level_output,
        batch_size=128,
        semantic_threshold=0.40,
    )

    print("Saved outputs:")
    print(f"- {review_text_features_output}")
    print(f"- {review_concern_level_output}")
    print("\nShapes:")
    print("review_text_features:", review_text_features.shape)
    print("review_concern_level:", review_concern_level.shape)
