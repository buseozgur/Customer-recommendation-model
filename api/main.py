from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from src.recommend import (
    get_concerns,
    get_skin_types,
    get_categories,
    recommend_products,
)

app = FastAPI(
    title="Sephora Recommendation API",
    version="5.0.0",
    description="Concern + skin type + category based Sephora product recommendation API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Sephora Recommendation API",
        "version": "5.0.0",
        "endpoints": {
            "health": "/health",
            "concerns": "/concerns",
            "skin_types": "/skin-types",
            "categories": "/categories",
            "recommend": "/recommend?concern=acne&skin_type=oily&category=Moisturizers&top_n=5",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "5.0.0"}


@app.get("/concerns")
def concerns():
    """Get all available skin concerns"""
    try:
        return {"concerns": get_concerns()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skin-types")
def skin_types():
    """Get all available skin types"""
    try:
        return {"skin_types": get_skin_types()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
def categories():
    """Get all product categories (secondary categories)"""
    try:
        return {"categories": get_categories()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
def recommend(
    concern: str = Query(..., description="Skin concern (e.g., acne, aging)"),
    skin_type: str = Query(..., description="Skin type (e.g., oily, dry)"),
    category: Optional[str] = Query(None, description="Product category filter"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations")
):
    """
    Get product recommendations based on concern, skin type, and optional filters.

    Filters:
    - concern: Required skin concern
    - skin_type: Required skin type
    - category: Optional product category filter
    - min_price: Optional minimum price (USD)
    - max_price: Optional maximum price (USD)
    - top_n: Number of results (1-20)
    """
    try:
        results = recommend_products(
            concern=concern,
            skin_type=skin_type,
            category=category,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n
        )

        return {
            "concern": concern,
            "skin_type": skin_type,
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
            "top_n": top_n,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
