import os
import streamlit as st
import requests
from typing import Optional

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Sephora Recommendation Agent",
    page_icon="https://i.ibb.co/bRW3RW20/sephora-icon.jpg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state FIRST
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Custom CSS - different for each page
if st.session_state.page == 'landing':
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding: 0 !important; max-width: 100% !important;}
        .main {padding: 0 !important;}

        /* Hide the button */
        div[data-testid="stButton"] {
            position: fixed;
            top: 40px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 1000;
            width: 200px;
            height: 150px;
        }

        div[data-testid="stButton"] > button {
            width: 100% !important;
            height: 100% !important;
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            cursor: pointer !important;
            opacity: 0 !important;
        }

        div[data-testid="stButton"] > button:hover {
            opacity: 0.1 !important;
            background: rgba(255, 255, 255, 0.1) !important;
        }

        .stImage {
            line-height: 0;
            margin: 0;
            padding: 0;
        }
    </style>
    """, unsafe_allow_html=True)

else:  # recommendations page
    st.markdown("""
    <style>
        #MainMenu, footer, header {
            visibility: hidden;
        }

        .stApp, .main {
            background-color: #000000 !important;
            padding: 0 !important;
        }

        section.main > div {
            padding-top: 0rem !important;
        }

        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 900px !important;
            background-color: #000000 !important;
        }

        .recommendation-header {
            background: #ffffff !important;
            padding: 0.8rem 1rem !important;
            text-align: center !important;
            margin-top: -1rem !important;
            margin-left: calc(-50vw + 50%) !important;
            margin-right: calc(-50vw + 50%) !important;
            margin-bottom: 0rem !important;
            width: 100vw !important;
        }

        .recommendation-header img {
            max-height: 80px !important;
            object-fit: contain !important;
        }

        .back-button-wrapper {
            margin-top: 2rem !important;
            margin-bottom: 2rem !important;
        }

        h1, h2, h3, h4, h5, h6, p, .stMarkdown {
            color: #ffffff !important;
        }

        label,
        .stSelectbox label,
        .stSlider label,
        .stCheckbox label {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        .stSelectbox > div > div,
        .stSelectbox div[data-baseweb="select"] {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 8px !important;
        }

        .stSelectbox div[data-baseweb="select"] * {
            color: #000000 !important;
        }

        .stSelectbox svg {
            fill: #000000 !important;
            color: #000000 !important;
        }

        ul[role="listbox"],
        ul[role="listbox"] li {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        ul[role="listbox"] li div,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
            color: #000000 !important;
        }

        input {
            color: #000000 !important;
            background-color: #ffffff !important;
        }

        .stButton > button {
            background-color: #ff4b4b !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px !important;
        }

        .stButton > button p {
            color: #ffffff !important;
        }

        .stButton > button:hover {
            background-color: #ff3333 !important;
            color: #ffffff !important;
        }

        button[kind="secondary"] {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: none !important;
        }

        button[kind="secondary"] p {
            color: #000000 !important;
        }

        .product-card {
            background: #1a1a1a !important;
            border: 2px solid #333333 !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin-bottom: 1rem !important;
        }

        .product-rank {
            display: inline-block !important;
            background: #ffffff !important;
            color: #000000 !important;
            padding: 0.5rem 1rem !important;
            border-radius: 20px !important;
            font-size: 14px !important;
            font-weight: 800 !important;
            margin-bottom: 1rem !important;
        }

        .product-name {
            font-size: 22px !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            margin-bottom: 0.5rem !important;
        }

        .product-brand {
            font-size: 16px !important;
            color: #cccccc !important;
            margin-bottom: 1rem !important;
        }

        .product-stats {
            color: #aaaaaa !important;
            font-size: 14px !important;
            line-height: 1.8 !important;
        }

        .product-stats b {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

# API Functions
@st.cache_data(show_spinner=False)
def get_concerns():
    try:
        response = requests.get(f"{API_URL}/concerns", timeout=20)
        response.raise_for_status()
        return response.json()["concerns"]
    except:
        return ["acne", "aging", "dark_spots", "dryness", "dullness", "pores", "sensitivity", "uneven_texture"]

@st.cache_data(show_spinner=False)
def get_skin_types():
    try:
        response = requests.get(f"{API_URL}/skin-types", timeout=20)
        response.raise_for_status()
        return response.json()["skin_types"]
    except:
        return ["combination", "dry", "normal", "oily", "sensitive"]

@st.cache_data(show_spinner=False)
def get_categories():
    try:
        response = requests.get(f"{API_URL}/categories", timeout=20)
        response.raise_for_status()
        return response.json()["categories"]
    except:
        return []

def get_recommendations(concern, skin_type, category=None, min_price=None, max_price=None, top_n=5):
    try:
        params = {"concern": concern, "skin_type": skin_type, "top_n": top_n}
        if category:
            params["category"] = category
        if min_price is not None:
            params["min_price"] = min_price
        if max_price is not None:
            params["max_price"] = max_price

        response = requests.get(f"{API_URL}/recommend", params=params, timeout=30)
        response.raise_for_status()
        return response.json()["results"]
    except Exception as e:
        st.error(f"API Error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════

if st.session_state.page == 'landing':
    # Invisible clickable button
    if st.button(" ", key="invisible_button"):
        st.session_state.page = 'recommendations'
        st.rerun()

    # Full screen Sephora homepage image
    st.image(
        "https://i.ibb.co/ccQGWStG/sephora-homepage.jpg",
        use_container_width=True
    )


# ═══════════════════════════════════════════════════════════════════
# RECOMMENDATION PAGE
# ═══════════════════════════════════════════════════════════════════

elif st.session_state.page == 'recommendations':
    # Header - Full width, at top
    st.markdown("""
    <div class="recommendation-header">
        <img src="https://i.ibb.co/nq0s0Fc1/Sephora-Logo.png" alt="Sephora">
    </div>
    """, unsafe_allow_html=True)

    # Back button wrapper with spacing
    st.markdown('<div class="back-button-wrapper">', unsafe_allow_html=True)
    if st.button("← Back to Home"):
        st.session_state.page = 'landing'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Find Your Perfect Products")

    # Load data
    concerns = get_concerns()
    skin_types = get_skin_types()
    categories = get_categories()

    # Form
    col1, col2 = st.columns(2)

    with col1:
        if categories:
            selected_category = st.selectbox("Product Category", ["All Categories"] + categories)
        else:
            selected_category = "All Categories"

        selected_concern = st.selectbox("Skin Concern", concerns)

    with col2:
        selected_skin_type = st.selectbox("Skin Type", skin_types)

    # Price filter
    st.markdown("### Price Range")
    use_price = st.checkbox("Enable price filter")

    if use_price:
        price_range = st.slider("Price ($)", 0.0, 200.0, (0.0, 200.0), 5.0)
        min_price, max_price = price_range
    else:
        min_price, max_price = None, None

    # Number of results
    top_n = st.slider("Number of recommendations", 1, 10, 5)

    # Search button
    if st.button("🔍 FIND PRODUCTS", use_container_width=True, type="primary"):
        with st.spinner("Finding the best products for you..."):
            results = get_recommendations(
                concern=selected_concern,
                skin_type=selected_skin_type,
                category=selected_category if selected_category != "All Categories" else None,
                min_price=min_price,
                max_price=max_price,
                top_n=top_n
            )

        st.markdown("---")
        st.markdown("## Recommended Products")

        if not results:
            st.warning("⚠️ No products found. Try different filters.")
        else:
            for idx, item in enumerate(results, 1):
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-rank">#{idx} Recommendation</div>
                    <div class="product-name">{item.get('product_name', 'Unknown')}</div>
                    <div class="product-brand">{item.get('brand_name', 'N/A')}</div>
                    <div class="product-stats">
                        <b>Price:</b> ${item.get('price', 0):.2f}<br>
                        <b>Score:</b> {item.get('score', 0):.4f}<br>
                        <b>Rating:</b> {item.get('mean_rating', 0):.1f}/5 |
                        <b>Helped:</b> {item.get('helped_ratio', 0)*100:.1f}%<br>
                        <b>Reviews:</b> {item.get('review_count', 0)}<br>
                        <b>Category:</b> {item.get('secondary_category', '')}<br>
                        <b>For:</b> {item.get('concern', '')} | {item.get('skin_type', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
