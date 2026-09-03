from pathlib import Path
import pandas as pd
import joblib
from typing import Dict, Any

# Explicit imports to register classes into Python namespace before unpickling
import sklearn
import sklearn.ensemble
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# Path resolution relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_medicines.csv"
MODEL_PATH = BASE_DIR / "model" / "rf_model.joblib"
TRF_PATH = BASE_DIR / "model" / "preprocessor.joblib"
PT_PATH = BASE_DIR / "model" / "power_transformer.joblib"

MODEL_VERSION = "1.0.0"

# Verify files exist before loading to avoid silent crashes
for path in [DATA_PATH, MODEL_PATH, TRF_PATH, PT_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

# Load artifacts
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
trf = joblib.load(TRF_PATH)
pt = joblib.load(PT_PATH)

# Precompute unit price
df['pack_size_valid'] = df['pack_size'].apply(lambda x: x if pd.notna(x) and x > 0 else 1.0)
df['unit_price'] = df['price_inr'] / df['pack_size_valid']


def get_medicine_alternatives(medicine_name: str) -> Dict[str, Any]:
    med_row = df[df['brand_name'].str.lower() == medicine_name.strip().lower()]
    
    if med_row.empty:
        return None
    
    selected_med = med_row.iloc[0]
    target_comp = selected_med['cleaned_composition']
    current_unit_price = selected_med['unit_price']
    
    cheaper_meds = df[
        (df['cleaned_composition'] == target_comp) & 
        (df['unit_price'] < current_unit_price) &
        (df['brand_name'].str.lower() != medicine_name.strip().lower())
    ]
    
    if cheaper_meds.empty:
        return {
            "searched_medicine": selected_med['brand_name'],
            "composition": target_comp,
            "alternatives": [],
            "message": "No cheaper alternatives found."
        }
    
    top_5 = cheaper_meds.sort_values(by='unit_price').head(5)
    records = top_5[[
        'brand_name', 
        'manufacturer', 
        'price_inr', 
        'pack_size', 
        'pack_unit'
    ]].to_dict(orient='records')
    
    return {
        "searched_medicine": selected_med['brand_name'],
        "composition": target_comp,
        "alternatives": records,
        "message": None
    }


def predict_fair_price(medicine_name: str) -> Dict[str, Any]:
    med_row = df[df['brand_name'].str.lower() == medicine_name.strip().lower()]
    
    if med_row.empty:
        return None
    
    # Exact 12 features required by preprocessor.joblib
    feature_columns = [
        'manufacturer', 
        'is_discontinued', 
        'dosage_form', 
        'pack_size', 
        'pack_unit', 
        'num_active_ingredients', 
        'primary_strength', 
        'active_ingredients', 
        'therapeutic_class', 
        'packaging_raw', 
        'manufacturer_raw', 
        'cleaned_composition'
    ]
    
    # 1. Extract the row with all required columns
    input_data = med_row[feature_columns].iloc[[0]].copy()
    
    # 2. Safe preprocessing of missing values/types
    input_data['pack_size'] = input_data['pack_size'].fillna(-1)
    input_data['pack_unit'] = input_data['pack_unit'].fillna('Unknown')
    input_data['is_discontinued'] = input_data['is_discontinued'].fillna(0).astype(int)
    
    # Fill remaining text columns if null
    for col in ['primary_strength', 'active_ingredients', 'therapeutic_class', 'packaging_raw', 'manufacturer_raw', 'cleaned_composition', 'manufacturer', 'dosage_form']:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('Unknown')
            
    # 3. Transform with preprocessor.joblib
    transformed_data = trf.transform(input_data)
    
    # 4. Predict with rf_model.joblib
    pred_transformed = model.predict(transformed_data)
    
    # 5. Inverse transform target price back to Rupees
    fair_price = pt.inverse_transform(pred_transformed.reshape(-1, 1))[0][0]
    actual_price = float(med_row.iloc[0]['price_inr'])
    
    return {
        "medicine_name": med_row.iloc[0]['brand_name'],
        "composition": med_row.iloc[0]['cleaned_composition'],
        "actual_price": round(actual_price, 2),
        "predicted_fair_price": round(float(fair_price), 2)
    }