import pandas as pd
import re


def combine_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine review_title and review_text into a single text column.

    Steps:
        - Fill missing values in review_title and review_text with empty string
        - Concatenate them into a new column called 'text'
        - Strip extra whitespace
        - Remove rows where combined text is empty

    Args:
        df (pd.DataFrame): Input dataframe containing review_title and review_text

    Returns:
        pd.DataFrame: Dataframe with a cleaned 'text' column
    """
    df = df.copy()

    df["review_title"] = df["review_title"].fillna("")
    df["review_text"] = df["review_text"].fillna("")

    df["text"] = df["review_title"] + " " + df["review_text"]
    df["text"] = df["text"].str.strip()

    df = df[df["text"].str.len() > 0].copy()

    return df


def clean_text(text: str) -> str:
    """
    Clean a single text string for NLP modeling.

    Steps:
        - Convert to lowercase
        - Remove URLs
        - Keep only letters and spaces
        - Normalize multiple spaces

    Args:
        text (str): Raw input text

    Returns:
        str: Cleaned text
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_clean_text_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'clean_text' column to the dataframe using the combined text column.

    Args:
        df (pd.DataFrame): Input dataframe containing 'text'

    Returns:
        pd.DataFrame: Dataframe with 'clean_text'
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df


def prepare_text_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full text preprocessing pipeline.

    Steps:
        1. Combine review_title and review_text
        2. Create clean_text column

    Args:
        df (pd.DataFrame): Input dataframe

    Returns:
        pd.DataFrame: Text-prepared dataframe
    """
    df = combine_text_columns(df)
    df = add_clean_text_column(df)
    return df
