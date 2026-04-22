from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.recommend import (
    get_concerns,
    get_skin_types,
    recommend_products,
)

app = FastAPI(
    title="Sephora Recommendation API",
    version="3.0.0",
    description="Concern + skin type based Sephora product recommendation API"
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
        "endpoints": {
            "health": "/health",
            "concerns": "/concerns",
            "skin_types": "/skin-types",
            "recommend": "/recommend?concern=acne&skin_type=oily&top_n=5",
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/concerns")
def concerns():
    try:
        return {"concerns": get_concerns()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skin-types")
def skin_types():
    try:
        return {"skin_types": get_skin_types()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend")
def recommend(
    concern: str = Query(...),
    skin_type: str = Query(...),
    top_n: int = Query(5, ge=1, le=20)
):
    try:
        results = recommend_products(
            concern=concern,
            skin_type=skin_type,
            top_n=top_n
        )
        return {
            "concern": concern,
            "skin_type": skin_type,
            "top_n": top_n,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
