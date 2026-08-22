import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (resolved via Docker internal DNS)
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart — Sales Revenue Prediction")
st.markdown(
    "Predict the **Product Store Sales Revenue** for a given product-store combination "
    "using the deployed XGBoost model."
)

# ── Online Prediction ─────────────────────────────────────────────────────────
st.subheader("Online Prediction (Single Record)")

col1, col2 = st.columns(2)

with col1:
    product_weight = st.number_input("Product Weight (g)", min_value=0.0, value=12.66, step=0.01)
    product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    product_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.027, step=0.001)
    product_mrp = st.number_input("Product MRP (₹)", min_value=0.0, value=117.08, step=0.01)
    product_id_char = st.selectbox("Product ID Prefix", ["FD", "DR", "NC"])

with col2:
    store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = st.selectbox("Store Type", [
        "Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"
    ])
    store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16, step=1)
    product_type_cat = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

if st.button("Predict Sales Revenue", type="primary"):
    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": product_sugar,
        "Product_Allocated_Area": product_area,
        "Product_MRP": product_mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": store_city_type,
        "Store_Type": store_type,
        "Product_Id_char": product_id_char,
        "Store_Age_Years": store_age,
        "Product_Type_Category": product_type_cat,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=10)
        if response.status_code == 200:
            pred = response.json()["Predicted_Sales_Revenue"]
            st.success(f"Predicted Sales Revenue: $ {pred:,.2f}")
        else:
            st.error(f"API error: {response.status_code} — {response.text}")
    except Exception as e:
        st.error(f"Unable to connect to the prediction API. Error: {e}")

# ── Batch Prediction ──────────────────────────────────────────────────────────
st.subheader("Batch Prediction (CSV Upload)")
st.markdown(
    "Upload a CSV file containing the following columns: "
    "`Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`, "
    "`Store_Size`, `Store_Location_City_Type`, `Store_Type`, `Product_Id_char`, "
    "`Store_Age_Years`, `Product_Type_Category`."
)

uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

if uploaded_file is not None:
    if st.button("Run Batch Prediction", type="primary"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": uploaded_file},
                timeout=60,
            )
            if response.status_code == 200:
                predictions = response.json()
                st.success("Batch predictions completed!")
                pred_df = pd.DataFrame(
                    list(predictions.items()), columns=["Row Index", "Predicted Sales Revenue ($)"]
                )
                st.dataframe(pred_df)
            else:
                st.error(f"API error: {response.status_code} — {response.text}")
        except Exception as e:
            st.error(f"Unable to connect to the prediction API. Error: {e}")
