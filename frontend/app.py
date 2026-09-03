from pathlib import Path
import os
import difflib
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Medicine Price & Alternative Finder", 
    page_icon="💊", 
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2. Global Backend URL & Directory Setup
# -----------------------------------------------------------------------------
API_URL = "https://fair-medicine-api-key.onrender.com"


BASE_DIR = Path(__file__).resolve().parent
LOCAL_OCR_DIR = BASE_DIR / "easyocr_data"
LOCAL_OCR_DIR.mkdir(exist_ok=True)
os.environ["EASYOCR_MODULE_PATH"] = str(LOCAL_OCR_DIR)

DATA_PATH = BASE_DIR / "data" / "cleaned_medicines.csv"

# -----------------------------------------------------------------------------
# 3. Load Brand Names for OCR Fuzzy Matching
# -----------------------------------------------------------------------------
@st.cache_data
def load_brand_names():
    try:
        if DATA_PATH.exists():
            df_local = pd.read_csv(DATA_PATH, usecols=['brand_name'])
            return df_local['brand_name'].dropna().drop_duplicates().tolist()
        return []
    except Exception as e:
        st.warning(f"Could not load local dataset for OCR matching: {e}")
        return []

all_brand_names = load_brand_names()

# -----------------------------------------------------------------------------
# 4. Cache EasyOCR Reader
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ocr_reader():
    try:
        import easyocr
        return easyocr.Reader(['en'], gpu=False, model_storage_directory=str(LOCAL_OCR_DIR))
    except Exception as exc:
        st.error(f"EasyOCR Error: {exc}")
        return None

# -----------------------------------------------------------------------------
# 5. Matching OCR Output to Known Brand Names
# -----------------------------------------------------------------------------
def match_ocr_to_brand(extracted_lines, brand_list):
    if not extracted_lines or not brand_list:
        return None
        
    full_text_lower = " ".join(extracted_lines).lower()
    sorted_brands = sorted(brand_list, key=len, reverse=True)
    
    # 1. Direct substring match
    for brand in sorted_brands:
        brand_clean = brand.lower()
        core_brand = brand_clean.replace("tablet", "").replace("capsule", "").replace("injection", "").replace("syrup", "").strip()
        
        if len(core_brand) >= 3 and core_brand in full_text_lower:
            return brand

    # 2. Line-by-Line Fuzzy Matching
    for line in extracted_lines:
        line_clean = line.strip()
        if len(line_clean) >= 3:
            matches = difflib.get_close_matches(line_clean, brand_list, n=1, cutoff=0.5)
            if matches:
                return matches[0]

    # 3. Word-by-Word Fuzzy Matching
    for line in extracted_lines:
        for word in line.split():
            if len(word) >= 4:
                matches = difflib.get_close_matches(word, brand_list, n=1, cutoff=0.6)
                if matches:
                    return matches[0]

    return None

# -----------------------------------------------------------------------------
# 6. UI: Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("💊 Medicine Finder")
st.sidebar.info("Upload a medicine strip photo or type a brand name to find fair prices and generic alternatives.")

# Backend wake-up / health check status in sidebar
with st.sidebar.expander("API Status", expanded=False):
    if st.button("Check Backend Status"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=10)
            if r.status_code == 200:
                st.success("Backend is LIVE 🟢")
            else:
                st.warning(f"Backend status code: {r.status_code}")
        except Exception:
            st.error("Backend is sleeping / unreachable 🔴")

st.sidebar.subheader("💡 Sample Medicines to Try")
sample_med = st.sidebar.selectbox(
    "Click a sample medicine to test:",
    [
        "Select a sample...",
        "Lulifer 1% Cream",
        "Moxind 500mg Capsule",
        "Ero 150mg Tablet",
        "Dial 0.5mg Tablet"
    ]
)

# -----------------------------------------------------------------------------
# 7. UI: Main Header
# -----------------------------------------------------------------------------
st.title("Medicine Price & Generic Alternative Finder")
st.markdown("Find out if you are overpaying for your medicine and discover cheaper alternatives with the exact same active ingredients!")

# -----------------------------------------------------------------------------
# 8. UI: OCR Section
# -----------------------------------------------------------------------------
st.subheader("Scan Medicine Strip (OCR)")
uploaded_image = st.file_uploader("Upload a photo of a medicine strip or box:", type=["jpg", "jpeg", "png"])

ocr_detected_name = ""

if uploaded_image is not None:
    col_img, col_info = st.columns([1, 2])
    
    with col_img:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Medicine Photo", width=240)
        
    with col_info:
        with st.spinner("Scanning image using EasyOCR..."):
            reader = load_ocr_reader()
            if reader is not None:
                try:
                    img_np = np.array(image)
                    extracted_lines = reader.readtext(img_np, detail=0)
                    
                    if extracted_lines:
                        st.write("📝 **Text Read from Image:**")
                        st.code("\n".join(extracted_lines))
                        
                        ocr_detected_name = match_ocr_to_brand(extracted_lines, all_brand_names)
                        
                        if ocr_detected_name:
                            st.success(f"**Matched Medicine in Database:** `{ocr_detected_name}`")
                        else:
                            st.warning("Text extracted, but no exact brand matched in dataset. Please type name manually below.")
                    else:
                        st.warning("No text detected in image. Please type the medicine name manually below.")
                except Exception as e:
                    st.error(f"OCR Processing Error: {str(e)}")

# -----------------------------------------------------------------------------
# 9. UI: Search Bar
# -----------------------------------------------------------------------------
default_search = ""
if ocr_detected_name:
    default_search = ocr_detected_name
elif sample_med != "Select a sample...":
    default_search = sample_med

medicine_name = st.text_input(
    "Enter Medicine Brand Name:", 
    value=default_search,
    placeholder="e.g., Moxind 500mg Capsule"
)

# -----------------------------------------------------------------------------
# 10. UI: Search Action & API Call
# -----------------------------------------------------------------------------
if st.button("Search Medicine", type="primary"):
    query = medicine_name.strip()
    if query:
        with st.spinner("Connecting to API (Render may take up to 60s to wake up on free tier)..."):
            try:
                # 90s timeout handles Render free-tier cold boot
                price_res = requests.get(f"{API_URL}/predict-price", params={"medicine_name": query}, timeout=90)
                alt_res = requests.get(f"{API_URL}/alternatives", params={"medicine_name": query}, timeout=90)
                
                if price_res.status_code == 200 and alt_res.status_code == 200:
                    price_data = price_res.json()
                    alt_data = alt_res.json()
                    
                    actual_price = price_data.get("actual_price", 0.0)
                    fair_price = price_data.get("predicted_fair_price", 0.0)
                    composition = price_data.get("composition", "N/A")
                    alternatives = alt_data.get("alternatives", [])
                    
                    st.divider()
                    st.success(f"**Medicine Found:** {price_data.get('medicine_name')}")
                    st.caption(f"**Composition:** `{composition.upper()}`")
                    
                    # --- Metrics Display ---
                    st.subheader("Price Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Actual Brand Price", f"₹{actual_price:.2f}")
                    with col2:
                        st.metric("AI Predicted Fair Price", f"₹{fair_price:.2f}")
                    with col3:
                        diff = actual_price - fair_price
                        if diff > 0:
                            st.metric("Price Status", "Overpriced", delta=f"-₹{abs(diff):.2f}", delta_color="inverse")
                        else:
                            st.metric("Price Status", "Fair / Below Average", delta=f"+₹{abs(diff):.2f}", delta_color="normal")
                    
                    st.divider()
                    
                    # --- Alternatives Table & Chart ---
                    st.subheader("Cheaper Alternatives")
                    
                    if isinstance(alternatives, list) and len(alternatives) > 0:
                        df_alts = pd.DataFrame(alternatives)
                        
                        st.dataframe(
                            df_alts.rename(columns={
                                "brand_name": "Brand Name", 
                                "manufacturer": "Manufacturer",
                                "price_inr": "Price (₹)",
                                "pack_size": "Pack Size",
                                "pack_unit": "Pack Unit"
                            }), 
                            use_container_width=True
                        )
                        
                        st.subheader("Manufacturer Price Comparison")
                        chart_data = df_alts[['manufacturer', 'price_inr']].copy()
                        st.bar_chart(data=chart_data, x="manufacturer", y="price_inr", color="#2e7bcf")
                        
                    else:
                        st.info("ℹ️ No cheaper alternatives found with the exact same composition.")
                        
                elif price_res.status_code == 404 or alt_res.status_code == 404:
                    st.error("❌ Medicine not found in database. Please check the spelling or select a sample from the sidebar.")
                else:
                    err_msg = price_res.json().get('detail', 'Unknown error occurred')
                    st.error(f"❌ Server Error: {err_msg}")
                    
            except requests.exceptions.Timeout:
                st.warning("⏱️ The request timed out. Render's free tier is likely still waking up. Please wait 10-15 seconds and click Search again!")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI backend. Please check if your Render API service is active.")
    else:
        st.warning("Please enter a medicine name or select a sample from the sidebar.")