import math
import streamlit as st
import requests

API_URL = "https://sephora-recommendation-api-289672719102.europe-west1.run.app"

st.set_page_config(
    page_title="Sephora Recommender",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f6f6f6;
}

.block-container {
    padding-top: 4.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

.sephora-title {
    text-align: center;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: 2px;
    color: #000000;
    margin-bottom: 0.35rem;
}

.sephora-subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 2.2rem;
}

.page-heading {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    color: #111111;
    margin-bottom: 2rem;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #000000;
    margin-top: 2rem;
    margin-bottom: 1rem;
}

.product-rank {
    display: inline-block;
    background-color: #000000;
    color: #ffffff;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 14px;
}

.product-name {
    font-size: 24px;
    font-weight: 800;
    color: #111111;
    margin-bottom: 4px;
}

.product-brand {
    font-size: 16px;
    color: #666666;
    margin-bottom: 12px;
}

.rating-stars {
    font-size: 28px;
    letter-spacing: 3px;
    margin-bottom: 18px;
    line-height: 1;
}

.star-wrapper {
    position: relative;
    display: inline-block;
    color: #d8d8d8;
}

.star-fill {
    position: absolute;
    top: 0;
    left: 0;
    white-space: nowrap;
    overflow: hidden;
    color: #000000;
}

.stButton > button {
    background-color: #000000;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 0.85rem 1.5rem;
    font-weight: 700;
    font-size: 16px;
    min-width: 260px;
}

.stButton > button:hover {
    background-color: #222222;
    color: #ffffff;
}

label {
    font-weight: 600 !important;
    color: #111111 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 14px 16px;
    border: 1.5px solid #dcdcdc;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    margin-bottom: 14px;
    max-width: 980px;
}

div[data-testid="stMetric"] {
    background: #f8f8f8;
    border: 1px solid #ececec;
    padding: 14px 16px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #555555 !important;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #111111 !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricDelta"] {
    color: #111111 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_primary_categories():
    response = requests.get(f"{API_URL}/categories/primary", timeout=20)
    response.raise_for_status()
    return response.json()["primary_categories"]


@st.cache_data(show_spinner=False)
def get_secondary_categories(primary_category):
    response = requests.get(
        f"{API_URL}/categories/secondary",
        params={"primary_category": primary_category},
        timeout=20
    )
    response.raise_for_status()
    return response.json()["secondary_categories"]


def get_recommendations(primary_category, secondary_category, top_n=5):
    response = requests.get(
        f"{API_URL}/recommend",
        params={
            "primary_category": primary_category,
            "secondary_category": secondary_category,
            "top_n": top_n
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["results"]


def render_star_rating(rating: float) -> str:
    safe_rating = max(0.0, min(5.0, float(rating)))
    fill_percent = (safe_rating / 5.0) * 100

    return f"""
    <div class="rating-stars">
        <div class="star-wrapper">
            ☆☆☆☆☆
            <div class="star-fill" style="width: {fill_percent}%;">★★★★★</div>
        </div>
        <span style="font-size:16px; color:#444; margin-left:10px; vertical-align:middle;">
            {safe_rating:.2f}/5
        </span>
    </div>
    """


st.markdown('<div class="sephora-title">SEPHORA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sephora-subtitle">AI-Powered Product Recommendation Experience</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="page-heading">Discover top-rated beauty products by category</div>',
    unsafe_allow_html=True
)

try:
    primary_categories = get_primary_categories()
except requests.RequestException:
    st.error("Primary category data could not be loaded from the API.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    selected_primary = st.selectbox("Beauty Category", primary_categories)

try:
    secondary_categories = get_secondary_categories(selected_primary)
except requests.RequestException:
    st.error("Secondary category data could not be loaded from the API.")
    st.stop()

with col2:
    selected_secondary = st.selectbox("Subcategory", secondary_categories)

button_left, button_center, button_right = st.columns([1.2, 1, 1.2])

with button_center:
    find_clicked = st.button("Find Recommendations")

if find_clicked:
    with st.spinner("Finding the best products for you..."):
        try:
            results = get_recommendations(
                primary_category=selected_primary,
                secondary_category=selected_secondary,
                top_n=5
            )
        except requests.RequestException:
            st.error("Recommendations could not be retrieved from the API.")
            st.stop()

    st.markdown('<div class="section-title">Customer Favorites</div>', unsafe_allow_html=True)

    if not results:
        st.warning("No recommendations were found for the selected category.")
    else:
        for i, item in enumerate(results, start=1):
            product_name = item.get("product_name", "Unknown Product")
            brand_name = item.get("brand_name", "Unknown Brand")
            avg_rating = float(item.get("avg_rating", 0))
            rating_count = int(item.get("rating_count", 0))
            avg_sentiment = float(item.get("avg_sentiment", 0))
            price = float(item.get("price", 0))

            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="product-rank">#{i} Recommended Product</div>
                    <div class="product-name">{product_name}</div>
                    <div class="product-brand">{brand_name}</div>
                    {render_star_rating(avg_rating)}
                    """,
                    unsafe_allow_html=True
                )

                c1, c2, c3 = st.columns(3)
                c1.metric("Review Count", f"{rating_count}")
                c2.metric("Sentiment Score", f"{avg_sentiment:.3f}")
                c3.metric("Price", f"${price:.2f}")