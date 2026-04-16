import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sephora Recommender", layout="wide")
st.title("SkinCare Product Recommendation System")


@st.cache_data
def load_data():
    df = pd.read_csv(
    "data/processed/sephora_clean.csv",
    usecols=[
        "brand_name",
        "product_name",
        "rating",
        "primary_category",
        "secondary_category",
    ],
    engine="python"
    )

    # rating sayısal olsun
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # text kolonlarını temizle
    for col in ["brand_name", "product_name", "primary_category", "secondary_category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # anlamsız / boş kayıtları temizle
    df = df[
        (df["brand_name"] != "") &
        (df["brand_name"] != "-") &
        (df["product_name"] != "") &
        (df["product_name"] != "-") &
        (df["primary_category"] != "") &
        (df["primary_category"] != "-")
    ].copy()

    return df


df = load_data()


# Ana kategori seçimi
primary_categories = sorted(df["primary_category"].dropna().unique().tolist())
selected_primary = st.selectbox("Ana kategori seçin", primary_categories)

# Ana kategoriye göre filtre
filtered_df = df[df["primary_category"] == selected_primary].copy()

if filtered_df.empty:
    st.warning("Bu ana kategoride veri bulunamadı.")
    st.stop()

# Alt kategori listesi
secondary_categories = sorted(
    [
        x for x in filtered_df["secondary_category"].dropna().unique().tolist()
        if x != "" and x != "-"
    ]
)

# Alt kategori varsa göster
if len(secondary_categories) > 0:
    selected_secondary = st.selectbox(
        "Alt kategori seçin",
        options=["Tümü"] + secondary_categories
    )

    if selected_secondary != "Tümü":
        filtered_df = filtered_df[
            filtered_df["secondary_category"] == selected_secondary
        ].copy()

if filtered_df.empty:
    st.warning("Seçtiğiniz filtrelerde veri bulunamadı.")
    st.stop()

st.subheader("Kategori bazlı en yüksek ortalama rating alan ürünler")

# Ürün bazlı ortalama rating + yorum sayısı
top_products = (
    filtered_df
    .groupby(["brand_name", "product_name"], as_index=False)
    .agg(
        avg_rating=("rating", "mean"),
        review_count=("rating", "count")
    )
)

# Çok az yorum alanları ele
top_products = top_products[top_products["review_count"] >= 5]

if top_products.empty:
    st.warning("Bu kategori için yeterli ürün verisi bulunamadı.")
    st.stop()

# Sıralama
top_products = (
    top_products
    .sort_values(["avg_rating", "review_count"], ascending=[False, False])
    .head(5)
    .reset_index(drop=True)
)

top_products = top_products.rename(columns={
    "brand_name": "Brand",
    "product_name": "Product",
    "avg_rating": "Average Rating",
    "review_count": "Review Count"
})

st.dataframe(
    top_products,
    use_container_width=True,
    hide_index=True
)
