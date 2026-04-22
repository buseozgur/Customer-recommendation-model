import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

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
    padding-top: 4rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}
.sephora-title {
    text-align: center;
    font-size: 48px;
    font-weight: 900;
    color: #000;
    margin-bottom: 0.3rem;
}
.sephora-subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 2rem;
}
.product-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    border: 1px solid #e6e6e6;
    margin-bottom: 14px;
}
.product-name {
    font-size: 22px;
    font-weight: 800;
    color: #111;
}
.product-brand {
    color: #666;
    margin-bottom: 10px;
}
.score-badge {
    display: inline-block;
    background: #000;
    color: white;
    border-radius: 999px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}
.stButton > button {
    width: 100%;
    background-color: #000;
    color: white;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_concerns():
    response = requests.get(f"{API_URL}/concerns", timeout=20)
    response.raise_for_status()
    return response.json()["concerns"]


@st.cache_data(show_spinner=False)
def get_skin_types():
    response = requests.get(f"{API_URL}/skin-types", timeout=20)
    response.raise_for_status()
    return response.json()["skin_types"]


def get_recommendations(concern, skin_type, top_n=5):
    response = requests.get(
        f"{API_URL}/recommend",
        params={
            "concern": concern,
            "skin_type": skin_type,
            "top_n": top_n
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["results"]


st.markdown('<div class="sephora-title">SEPHORA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sephora-subtitle">AI-Powered Skincare Recommendation</div>',
    unsafe_allow_html=True
)

st.markdown("### Select your skin concern and skin type")

try:
    concerns = get_concerns()
    skin_types = get_skin_types()
except requests.RequestException:
    st.error("API’den dropdown verileri yüklenemedi.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    selected_concern = st.selectbox("Concern", concerns)

with col2:
    selected_skin_type = st.selectbox("Skin Type", skin_types)

top_n = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if st.button("Find Recommendations"):
    with st.spinner("Finding the best products for you..."):
        try:
            results = get_recommendations(
                concern=selected_concern,
                skin_type=selected_skin_type,
                top_n=top_n
            )
        except requests.RequestException:
            st.error("API’den öneriler alınamadı.")
            st.stop()

    st.markdown("### Recommended Products")

    if not results:
        st.warning("Bu concern + skin type kombinasyonu için sonuç bulunamadı.")
    else:
        for i, item in enumerate(results, start=1):
            st.markdown(
                f"""
                <div class="product-card">
                    <div class="score-badge">#{i} Recommendation</div>
                    <div class="product-name">{item.get("product_name", "")}</div>
                    <div class="product-brand">{item.get("brand_name", "")}</div>
                    <p><b>Score:</b> {item.get("score", 0)}</p>
                    <p><b>Rating:</b> {item.get("mean_rating", 0)}/5</p>
                    <p><b>Helped Ratio:</b> {item.get("helped_ratio", 0)}</p>
                    <p><b>Review Count:</b> {item.get("review_count", 0)}</p>
                    <p><b>Category:</b> {item.get("primary_category", "")} › {item.get("secondary_category", "")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
