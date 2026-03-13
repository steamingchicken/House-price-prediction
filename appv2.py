import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("house_modelv2.pkl", "rb"))
model_columns = pickle.load(open("model_columnsv2.pkl", "rb"))

st.title("🏠 Singapore HDB Resale Price Predictor")

st.sidebar.header("Enter Flat Details")

# ---------- USER INPUT UI ----------
month = st.sidebar.text_input("Month (YYYY-MM)", "2024-01")

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
    "Flat Type",
    ["1 ROOM","2 ROOM","3 ROOM","4 ROOM","5 ROOM","EXECUTIVE"]
)

block = st.sidebar.text_input("Block", "123")

street_name = st.sidebar.text_input("Street Name", "ANG MO KIO AVE 3")

storey_range = st.sidebar.selectbox(
    "Storey Range",
    ["01 TO 03","04 TO 06","07 TO 09","10 TO 12",
     "13 TO 15","16 TO 18","19 TO 21","22 TO 24",
     "25 TO 27","28 TO 30","31 TO 33","34 TO 36",
     "37 TO 39","40 TO 42","43 TO 45","46 TO 48"]
)

floor_area = st.sidebar.number_input("Floor Area (sqm)", 30.0, 200.0)

flat_model = st.sidebar.selectbox(
    "Flat Model",
    ["Improved","New Generation","Model A","Premium Apartment",
     "Simplified","Apartment","Maisonette","DBSS","Type S1","Type S2"]
)

lease_commence = st.sidebar.number_input("Lease Commence Year", 1960, 2024)

remaining_lease = st.sidebar.number_input("Remaining Lease (years)", 0.0, 99.0)

# ---------- BUILD INPUT DATAFRAME ----------
input_dict = {
    "month": month,
    "town": town,
    "flat_type": flat_type,
    "block": block,
    "street_name": street_name,
    "storey_range": storey_range,
    "floor_area_sqm": floor_area,
    "flat_model": flat_model,
    "lease_commence_date": lease_commence,
    "remaining_lease": remaining_lease
}

input_df = pd.DataFrame([input_dict])

# ---------- APPLY OHE SAME AS TRAINING ----------
input_df = pd.get_dummies(input_df)

input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ---------- PREDICTION ----------
if st.button("Predict Resale Price"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Resale Price: ${prediction[0]:,.0f}") 