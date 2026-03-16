import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# 1. Load model, model columns, and scaler
model = joblib.load("house_modelv5.pkl")
model_columns = joblib.load("model_columnsv5.pkl")
scaler = joblib.load("scalerv5.pkl")

st.set_page_config(page_title="HDB Resale Price Predictor", layout="centered")
st.title("🏠 Singapore HDB Resale Price Predictor")

st.sidebar.header("Enter Flat Details")

# --- Inputs ---
selected_date = st.sidebar.date_input("Transaction Month", datetime(2024, 1, 1))
month_numeric = selected_date.year + (selected_date.month - 1) / 12

town = st.sidebar.selectbox(
    "Town",
    ["ANG MO KIO","BEDOK","BISHAN","BUKIT BATOK","BUKIT MERAH", 
     "BUKIT PANJANG","CHOA CHU KANG","CLEMENTI","GEYLANG",
     "HOUGANG","JURONG EAST","JURONG WEST","KALLANG/WHAMPOA",
     "PASIR RIS","PUNGGOL","QUEENSTOWN","SEMBAWANG",
     "SENGKANG","SERANGOON","TAMPINES","TOA PAYOH",
     "WOODLANDS","YISHUN"]
)

flat_type = st.sidebar.selectbox(
    "Flat Type", ["1 ROOM","2 ROOM","3 ROOM","4 ROOM","5 ROOM","EXECUTIVE","MULTI-GENERATION"]
)

storey_range = st.sidebar.selectbox(
    "Storey Range",
    ["01 TO 03","04 TO 06","07 TO 09","10 TO 12","13 TO 15","16 TO 18",
     "19 TO 21","22 TO 24","25 TO 27","28 TO 30","31 TO 33","34 TO 36",
     "37 TO 39","40 TO 42","43 TO 45","46 TO 48"]
)

floor_area = st.sidebar.number_input("Floor Area (sqm)", 30.0, 200.0, value=90.0)

flat_model = st.sidebar.selectbox(
    "Flat Model",
    ["Improved","New Generation","Model A","Premium Apartment",
     "Simplified","Apartment","Maisonette","DBSS","Type S1","Type S2"]
)

remaining_lease_years = st.sidebar.number_input("Remaining Lease (years)", 0.0, 99.0, value=90.0)

# --- Data Transformation ---
storey_order = ["01 TO 03","04 TO 06","07 TO 09","10 TO 12",
                "13 TO 15","16 TO 18","19 TO 21","22 TO 24",
                "25 TO 27","28 TO 30","31 TO 33","34 TO 36",
                "37 TO 39","40 TO 42","43 TO 45","46 TO 48"]
storey_level = storey_order.index(storey_range) + 1

flat_type_mapping = {"1 ROOM":1, "2 ROOM":2, "3 ROOM":3, "4 ROOM":4,
                     "5 ROOM":5, "EXECUTIVE":6, "MULTI-GENERATION":7}
flat_type_numeric = flat_type_mapping[flat_type]

remaining_lease_months = remaining_lease_years * 12

# --- Build Input DataFrame ---
input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
numeric_cols = ["month_numeric", "storey_level", "flat_type_numeric", "floor_area_sqm", "remaining_lease_months"]

# Fill numeric values
input_df.at[0, "month_numeric"] = month_numeric
input_df.at[0, "storey_level"] = storey_level
input_df.at[0, "flat_type_numeric"] = flat_type_numeric
input_df.at[0, "floor_area_sqm"] = floor_area
input_df.at[0, "remaining_lease_months"] = remaining_lease_months

# Scale numeric features
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# One-hot encoding helper
def safe_activate(prefix, value):
    target = value.lower().replace(" ", "").replace("-", "")
    for col in input_df.columns:
        if col.startswith(prefix):
            col_clean = col.replace(prefix, "").lower().replace("_", "").replace(" ", "").replace("-", "")
            if col_clean == target:
                input_df.at[0, col] = 1
                return col
    return None

town_found = safe_activate("town_", town)
flat_model_found = safe_activate("flat_model_", flat_model)

if town_found is None:
    st.sidebar.error(f"Town '{town}' not recognised by model")

if flat_model_found is None:
    st.sidebar.error(f"Flat model '{flat_model}' not recognised by model")  

# --- Predict ---
st.divider()

if st.button("🚀 Predict Resale Price", use_container_width=True, key="predict_button"):

    input_ready = input_df[model_columns]

    if input_ready.isnull().values.any():
        st.error("Error: Input contains NaN values.")

    else:
        # --- Model prediction ---
        pred = model.predict(input_ready)
        final_price = pred[0]

        # ⭐ --- Confidence score logic (percentage based) ---
        confidence_score = 90
        reasons = []

        # lease reasoning
        if remaining_lease_years < 30:
            confidence_score -= 40
            reasons.append(
                "The flat has a very short remaining lease. "
                "Such flats often experience accelerated price depreciation "
                "and fewer comparable resale transactions exist in the dataset."
            )
        elif remaining_lease_years < 50:
            confidence_score -= 20
            reasons.append(
                "The remaining lease is moderately low. "
                "Buyer demand may be affected by CPF usage limits and financing constraints, "
                "which introduces greater uncertainty in pricing behaviour."
            )

        # floor area reasoning
        if floor_area < 40:
            confidence_score -= 20
            reasons.append(
                "The flat size is unusually small. "
                "Small units tend to have more volatile price patterns due to niche buyer demand "
                "and limited comparable transactions."
            )
        elif floor_area > 150:
            confidence_score -= 20
            reasons.append(
                "The flat size is significantly larger than typical HDB resale units. "
                "Such properties may include premium layouts or rare attributes not fully captured by the model."
            )

        # clamp confidence
        confidence_score = max(5, min(95, confidence_score))

        if len(reasons) == 0:
            confidence_reason = (
                "All selected flat characteristics fall within common market ranges. "
                "The model has seen many similar transactions during training, "
                "which improves prediction reliability."
            )
        else:
            confidence_reason = "Prediction uncertainty arises because:\n\n• " + "\n• ".join(reasons)

        # ⭐ --- Dynamic price range based on confidence ---
        spread = (100 - confidence_score) / 200   # lower confidence → wider range
        lower_price = final_price * (1 - spread)
        upper_price = final_price * (1 + spread)

        if final_price < 0:
            st.warning("⚠️ Model predicted unrealistic price. Input may be outside training range.")
            st.error(f"Model Output: -${abs(final_price):,.0f}")

        else:
            st.success(f"## 💰 Estimated Resale Price: ${final_price:,.0f}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Low Estimate", f"${lower_price:,.0f}")
            col2.metric("Expected", f"${final_price:,.0f}")
            col3.metric("High Estimate", f"${upper_price:,.0f}")

            st.info(f"""
📊 **Model Confidence: {confidence_score}%**

**Interpretation:**  
{confidence_reason}
""")

            # ⭐ --- Investment Insight ---
            if confidence_score >= 75 and remaining_lease_years > 60:
                insight = "💡 **Investment Outlook: Strong** — Long lease duration and typical flat characteristics suggest relatively stable long-term resale value."
            elif confidence_score >= 50:
                insight = "⚠️ **Investment Outlook: Moderate Risk** — Some characteristics may introduce price variability. Buyers should consider future lease decay effects."
            else:
                insight = "❌ **Investment Outlook: Higher Risk** — Short lease or uncommon flat attributes may lead to faster value depreciation and lower buyer demand."

            st.markdown(f"### 📈 Investment Insights\n{insight}") 