import pandas as pd
from pathlib import Path
from typing import List, Tuple


def load_products(products_path: str) -> pd.DataFrame:
    """
    Load the product dataset from CSV.

    Args:
        products_path (str): Path to the product_info.csv file.

    Returns:
        pd.DataFrame: Product dataframe.
    """
    return pd.read_csv(products_path)


def load_reviews(review_paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple review CSV files.

    Args:
        review_paths (List[str]): List of review CSV file paths.

    Returns:
        pd.DataFrame: Combined review dataframe.
    """
    dfs = [
        pd.read_csv(
            path,
            low_memory=False,
            dtype={"author_id": "string"}
        )
        for path in review_paths
    ]
    return pd.concat(dfs, ignore_index=True)


def select_relevant_columns(
    products: pd.DataFrame,
    reviews: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only relevant columns needed for downstream modeling.

    Args:
        products (pd.DataFrame): Raw products dataframe.
        reviews (pd.DataFrame): Raw reviews dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            Filtered products and reviews dataframes.
    """
    review_cols = [
        "author_id",
        "rating",
        "is_recommended",
        "helpfulness",
        "total_feedback_count",
        "total_neg_feedback_count",
        "total_pos_feedback_count",
        "submission_time",
        "review_text",
        "review_title",
        "skin_tone",
        "eye_color",
        "skin_type",
        "hair_color",
        "product_id",
        "product_name",
        "brand_name",
        "price_usd",
    ]

    product_cols = [
        "product_id",
        "product_name",
        "brand_id",
        "brand_name",
        "loves_count",
        "rating",
        "reviews",
        "size",
        "variation_type",
        "variation_value",
        "variation_desc",
        "ingredients",
        "price_usd",
        "value_price_usd",
        "sale_price_usd",
        "limited_edition",
        "new",
        "online_only",
        "out_of_stock",
        "sephora_exclusive",
        "highlights",
        "primary_category",
        "secondary_category",
        "tertiary_category",
        "child_count",
        "child_max_price",
        "child_min_price",
    ]

    reviews = reviews[review_cols].copy()
    products = products[product_cols].copy()

    if "Unnamed: 0" in reviews.columns:
        reviews = reviews.drop(columns=["Unnamed: 0"])

    return products, reviews


def rename_product_columns(products: pd.DataFrame) -> pd.DataFrame:
    """
    Rename overlapping product columns before merge
    to avoid _x / _y suffixes.

    Args:
        products (pd.DataFrame): Product dataframe.

    Returns:
        pd.DataFrame: Renamed product dataframe.
    """
    return products.rename(columns={
        "product_name": "product_name_meta",
        "brand_name": "brand_name_meta",
        "rating": "product_rating",
        "reviews": "product_review_count",
        "price_usd": "price_usd_meta",
    })


def merge_datasets(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Merge reviews with product metadata using a review-centric left join.

    Args:
        products (pd.DataFrame): Product dataframe.
        reviews (pd.DataFrame): Reviews dataframe.

    Returns:
        pd.DataFrame: Merged dataframe.
    """
    return reviews.merge(products, on="product_id", how="left")


def create_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create unified final columns for product name, brand name, and price.

    Product metadata is prioritized. If metadata is missing,
    review-side values are used as fallback.

    Args:
        df (pd.DataFrame): Merged dataframe.

    Returns:
        pd.DataFrame: Dataframe with canonical columns.
    """
    df = df.copy()

    df["product_name_final"] = df["product_name_meta"].fillna(df["product_name"])
    df["brand_name_final"] = df["brand_name_meta"].fillna(df["brand_name"])
    df["price_usd_final"] = df["price_usd_meta"].fillna(df["price_usd"])

    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert numeric, datetime, and binary columns to appropriate types.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with corrected types.
    """
    df = df.copy()

    numeric_cols = [
        "rating",
        "is_recommended",
        "helpfulness",
        "total_feedback_count",
        "total_neg_feedback_count",
        "total_pos_feedback_count",
        "price_usd",
        "price_usd_meta",
        "price_usd_final",
        "brand_id",
        "loves_count",
        "product_rating",
        "product_review_count",
        "value_price_usd",
        "sale_price_usd",
        "child_count",
        "child_max_price",
        "child_min_price",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "submission_time" in df.columns:
        df["submission_time"] = pd.to_datetime(df["submission_time"], errors="coerce")

    binary_cols = [
        "limited_edition",
        "new",
        "online_only",
        "out_of_stock",
        "sephora_exclusive",
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")

    return df


def drop_low_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop selected columns with extremely high missingness
    and low value for the first modeling iteration.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Reduced dataframe.
    """
    df = df.copy()

    drop_cols = [
        "variation_desc",
        "sale_price_usd",
        "value_price_usd",
    ]

    return df.drop(columns=drop_cols, errors="ignore")


def filter_required_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows missing critical fields required for review-level analysis.

    Required fields:
    - product_id
    - review_text
    - rating

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    return df.dropna(subset=["product_id", "review_text", "rating"]).copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicated rows.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Deduplicated dataframe.
    """
    return df.drop_duplicates().copy()


def preserve_raw_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve raw text without cleaning it.

    This step combines review_title and review_text into raw_text.
    Text cleaning should be handled later in the NLP pipeline.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with raw_text column.
    """
    df = df.copy()

    df["review_title"] = df["review_title"].fillna("").astype(str)
    df["review_text"] = df["review_text"].fillna("").astype(str)

    df["raw_text"] = (
        df["review_title"].str.strip() + " " + df["review_text"].str.strip()
    ).str.strip()

    return df


def add_basic_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight text quality features.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with text-based helper features.
    """
    df = df.copy()

    df["review_text_length"] = df["review_text"].str.len()
    df["raw_text_length"] = df["raw_text"].str.len()
    df["has_title"] = (df["review_title"].str.len() > 0).astype(int)

    return df


def add_rating_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create helper rating_category labels from rating values.

    Categories:
        rating >= 4 -> positive
        rating == 3 -> neutral
        rating < 3 -> negative

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with rating_category column.
    """
    df = df.copy()

    def rating_cat(x: float) -> str:
        if x >= 4:
            return "positive"
        if x == 3:
            return "neutral"
        return "negative"

    df["rating_category"] = df["rating"].apply(rating_cat)
    return df


def build_review_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the review-level master dataframe.

    Args:
        df (pd.DataFrame): Cleaned merged dataframe.

    Returns:
        pd.DataFrame: Review-level master dataframe.
    """
    cols = [
        "author_id",
        "product_id",
        "product_name_final",
        "brand_name_final",
        "rating",
        "rating_category",
        "is_recommended",
        "helpfulness",
        "total_feedback_count",
        "total_neg_feedback_count",
        "total_pos_feedback_count",
        "submission_time",
        "review_title",
        "review_text",
        "raw_text",
        "review_text_length",
        "raw_text_length",
        "has_title",
        "skin_tone",
        "eye_color",
        "skin_type",
        "hair_color",
        "price_usd_final",
        "brand_id",
        "loves_count",
        "product_rating",
        "product_review_count",
        "size",
        "variation_type",
        "variation_value",
        "ingredients",
        "limited_edition",
        "new",
        "online_only",
        "out_of_stock",
        "sephora_exclusive",
        "highlights",
        "primary_category",
        "secondary_category",
        "tertiary_category",
        "child_count",
        "child_max_price",
        "child_min_price",
    ]

    existing_cols = [col for col in cols if col in df.columns]
    return df[existing_cols].copy()


def build_product_master(products: pd.DataFrame) -> pd.DataFrame:
    """
    Build the product-level master dataframe.

    Args:
        products (pd.DataFrame): Renamed product dataframe.

    Returns:
        pd.DataFrame: Product-level master dataframe.
    """
    product_master = products.copy()
    product_master["product_name_final"] = product_master["product_name_meta"]
    product_master["brand_name_final"] = product_master["brand_name_meta"]
    product_master["price_usd_final"] = product_master["price_usd_meta"]

    return product_master


def save_parquet(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataframe as parquet.

    Args:
        df (pd.DataFrame): Dataframe to save.
        output_path (str): Output parquet path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def build_master_datasets(
    products_path: str,
    review_paths: List[str],
    review_output_path: str,
    product_output_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline for building clean master datasets.

    Steps:
        1. Load products
        2. Load reviews
        3. Select relevant columns
        4. Rename product-side columns
        5. Merge datasets with left join
        6. Create canonical columns
        7. Fix data types
        8. Drop selected low-value columns
        9. Filter required rows
        10. Remove duplicates
        11. Preserve raw text
        12. Add basic text features
        13. Add rating category
        14. Build review_master
        15. Build product_master
        16. Save outputs

    Args:
        products_path (str): Path to product_info.csv.
        review_paths (List[str]): List of review CSV paths.
        review_output_path (str): Output path for review_master parquet.
        product_output_path (str): Output path for product_master parquet.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            review_master and product_master dataframes.
    """
    products = load_products(products_path)
    reviews = load_reviews(review_paths)

    products, reviews = select_relevant_columns(products, reviews)
    products = rename_product_columns(products)

    df = merge_datasets(products, reviews)
    df = create_canonical_columns(df)
    df = fix_data_types(df)
    df = drop_low_value_columns(df)
    df = filter_required_rows(df)
    df = remove_duplicates(df)
    df = preserve_raw_text(df)
    df = add_basic_text_features(df)
    df = add_rating_category(df)

    review_master = build_review_master(df)
    product_master = build_product_master(products)

    save_parquet(review_master, review_output_path)
    save_parquet(product_master, product_output_path)

    return review_master, product_master
