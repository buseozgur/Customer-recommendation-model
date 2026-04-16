from fastapi import FastAPI
from src.recommend import (
    get_primary_categories,
    get_secondary_categories,
    recommend_top_products
)

# 🔥 BU SATIR ÇOK KRİTİK
app = FastAPI(title="Sephora Recommendation API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories/primary")
def primary_categories():
    return {"primary_categories": get_primary_categories()}


@app.get("/categories/secondary")
def secondary_categories(primary_category: str):
    return {
        "primary_category": primary_category,
        "secondary_categories": get_secondary_categories(primary_category)
    }


@app.get("/recommend")
def recommend(primary_category: str, secondary_category: str, top_n: int = 5):
    results = recommend_top_products(
        primary_category=primary_category,
        secondary_category=secondary_category,
        top_n=top_n
    )
    return {
        "primary_category": primary_category,
        "secondary_category": secondary_category,
        "results": results
    }
