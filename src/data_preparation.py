import pandas as pd
from pathlib import Path
from typing import List


def load_products(products_path: str) -> pd.DataFrame:
    """
    Load product dataset from CSV file.

    Args:
        products_path (str): Path to product_info.csv

    Returns:
        pd.DataFrame: Products dataframe
    """
    return pd.read_csv(products_path)


def load_reviews(review_paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple review datasets.

    Args:
        review_paths (List[str]): List of review CSV file paths

    Returns:
        pd.DataFrame: Combined reviews dataframe
    """
    dfs = [pd.read_csv(path, low_memory=False) for path in review_paths]
    return pd.concat(dfs, ignore_index=True)


def merge_datasets(products: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Merge products and reviews datasets on product_id.

    Args:
        products (pd.DataFrame): Product dataframe
        reviews (pd.DataFrame): Reviews dataframe

    Returns:
        pd.DataFrame: Merged dataframe
    """
    df = pd.merge(reviews, products, on="product_id", how="outer")

    # Rename duplicated columns
    df = df.rename(columns={
        "rating_x": "rating",
        "product_name_x": "product_name",
        "brand_name_x": "brand_name",
        "price_usd_x": "price_usd"
    })

    # Drop duplicated columns
    drop_cols = [
        "rating_y",
        "product_name_y",
        "brand_name_y",
        "price_usd_y"
    ]
    df = df.drop(columns=drop_cols)

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean merged dataframe:
    - Drop high-missing columns
    - Remove null review_text
    - Remove duplicates
    - Normalize text

    Args:
        df (pd.DataFrame): Raw merged dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    # Drop columns with too many missing values
    drop_cols = [
        "variation_desc",
        "sale_price_usd",
        "value_price_usd",
        "child_max_price",
        "child_min_price"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Drop rows with missing review_text
    df = df.dropna(subset=["review_text"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Basic text cleaning
    df["review_text"] = df["review_text"].str.lower()
    df["review_text"] = df["review_text"].str.replace(r"[^\w\s]", "", regex=True)

    return df


def add_rating_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rating_category feature from rating.

    Categories:
        rating >= 4 -> positive
        rating == 3 -> neutral
        rating < 3 -> negative

    Args:
        df (pd.DataFrame): Cleaned dataframe

    Returns:
        pd.DataFrame: Dataframe with rating_category column
    """

    def rating_cat(x):
        if x >= 4:
            return "positive"
        elif x == 3:
            return "neutral"
        else:
            return "negative"

    df["rating_category"] = df["rating"].apply(rating_cat)

    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed dataframe to CSV.

    Args:
        df (pd.DataFrame): Final cleaned dataframe
        output_path (str): Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_clean_dataset(
    products_path: str,
    review_paths: List[str],
    output_path: str
) -> pd.DataFrame:
    """
    End-to-end pipeline to create clean dataset.

    Steps:
        1. Load products
        2. Load reviews
        3. Merge datasets
        4. Clean dataframe
        5. Add rating_category
        6. Save output

    Args:
        products_path (str): Path to product_info.csv
        review_paths (List[str]): List of review CSV paths
        output_path (str): Output CSV path

    Returns:
        pd.DataFrame: Final cleaned dataframe
    """

    products = load_products(products_path)
    reviews = load_reviews(review_paths)

    df = merge_datasets(products, reviews)
    df = clean_dataframe(df)
    df = add_rating_category(df)

    save_processed_data(df, output_path)

    return df
