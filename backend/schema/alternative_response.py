from typing import List, Optional
from pydantic import BaseModel, Field

class MedicineItem(BaseModel):
    brand_name: str = Field(..., description="Brand name of alternative medicine")
    manufacturer: str = Field(..., description="Manufacturer name")
    price_inr: float = Field(..., description="Retail price in INR")
    pack_size: Optional[float] = Field(None, description="Pack size")
    pack_unit: Optional[str] = Field(None, description="Unit of the pack (e.g. tablet, syrup)")

class AlternativeResponse(BaseModel):
    searched_medicine: str = Field(..., description="Input medicine name")
    composition: str = Field(..., description="Active chemical composition")
    alternatives: List[MedicineItem] = Field(default_factory=list, description="Top cheaper alternatives")
    message: Optional[str] = Field(None, description="Optional status message")