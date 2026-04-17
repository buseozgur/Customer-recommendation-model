import streamlit as st
import requests

API_URL = "https://sephora-recommendation-api-289672719102.europe-west1.run.app"

st.set_page_config(page_title="Sephora Recommender", layout="wide")
st.title("SkinCare Product Recommendation System")

# primary categories
primary_resp = requests.get(f"{API_URL}/categories/primary")
primary_categories = primary_resp.json()["primary_categories"]

selected_primary = st.selectbox("Ana kategori seçin", primary_categories)

# secondary categories
secondary_resp = requests.get(
    f"{API_URL}/categories/secondary",
    params={"primary_category": selected_primary}
)
secondary_categories = secondary_resp.json()["secondary_categories"]

selected_secondary = st.selectbox("Alt kategori seçin", secondary_categories)

if st.button("Önerileri Getir"):
    rec_resp = requests.get(
        f"{API_URL}/recommend",
        params={
            "primary_category": selected_primary,
            "secondary_category": selected_secondary,
            "top_n": 5
        }
    )
    results = rec_resp.json()["results"]

    st.markdown("## Customers Favorite")

    if not results:
        st.warning("Seçtiğiniz kategori için öneri bulunamadı.")
    else:
        for i, item in enumerate(results, start=1):
            st.markdown(f"### {i}. {item['product_name']}")
            st.write(f"**Brand:** {item['brand_name']}")
            st.write(f"**Average Rating:** {item['avg_rating']:.2f}")
            st.write(f"**Review Count:** {int(item['rating_count'])}")
            st.write(f"**Average Sentiment:** {item['avg_sentiment']:.3f}")
            st.write(f"**Price:** ${item['price']:.2f}")
            st.write("---")
