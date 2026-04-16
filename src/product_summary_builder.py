import pandas as pd


def build_product_summary(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Build product-level summary table from review-level sentiment predictions.

    Aggregates each product by:
        - average rating
        - number of ratings
        - average sentiment score
        - positive ratio
        - negative ratio
        - average price

    Grouping columns:
        - primary_category
        - secondary_category
        - product_id
        - brand_name
        - product_name

    Args:
        df_model (pd.DataFrame): Review-level dataframe with sentiment outputs

    Returns:
        pd.DataFrame: Product-level summary dataframe
    """
    product_summary = df_model.groupby(
        ["primary_category", "secondary_category", "product_id", "brand_name", "product_name"],
        as_index=False
    ).agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        avg_sentiment=("sentiment_score", "mean"),
        positive_ratio=("sentiment_pred", lambda x: (x == "positive").mean()),
        negative_ratio=("sentiment_pred", lambda x: (x == "negative").mean()),
        price=("price_usd", "mean")
    )

    return product_summary


def save_product_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Save product summary dataframe to CSV.

    Args:
        df (pd.DataFrame): Product summary dataframe
        output_path (str): Output file path
    """
    df.to_csv(output_path, index=False)
