from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from schema.alternative_response import AlternativeResponse
from schema.price_prediction_response import PricePredictionResponse
from model.predict import (
    get_medicine_alternatives, 
    predict_fair_price, 
    model, 
    MODEL_VERSION
)

app = FastAPI(
    title="Medicine Price & Alternative Finder API",
    description="API for finding generic/cheaper alternative medicines and predicting fair prices using ML.",
    version=MODEL_VERSION
)

# 1. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Endpoints --------------------

@app.get("/")
def home():
    return {"message": "Medicine Price API is live and running!"}


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_version": MODEL_VERSION,
        "model_loaded": model is not None
    }


@app.get("/alternatives", response_model=AlternativeResponse)
def alternatives_endpoint(medicine_name: str = Query(..., description="Brand name of the medicine")):
    result = get_medicine_alternatives(medicine_name)
    if result is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Medicine '{medicine_name}' not found in database."
        )
    return result


@app.get("/predict-price", response_model=PricePredictionResponse)
def predict_price_endpoint(medicine_name: str = Query(..., description="Brand name of the medicine")):
    try:
        result = predict_fair_price(medicine_name)
        if result is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Medicine '{medicine_name}' not found in database."
            )
        return result
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")