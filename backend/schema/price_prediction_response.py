from pydantic import BaseModel, Field

class PricePredictionResponse(BaseModel):
    medicine_name: str = Field(..., description="Brand name of the medicine")
    composition: str = Field(..., description="Active chemical composition")
    actual_price: float = Field(..., description="Actual market price in INR")
    predicted_fair_price: float = Field(..., description="Model predicted fair price in INR")