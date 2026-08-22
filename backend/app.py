# Import necessary libraries
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Revenue Predictor")

# Load the trained machine learning model pipeline
model = joblib.load("superkart_model.joblib")


@superkart_api.get("/")
def home():
    """Health-check endpoint."""
    return "Welcome to the SuperKart Sales Revenue Prediction API!"


@superkart_api.post("/v1/predict")
def predict_sales():
    """
    Single-record (online) prediction endpoint.
    Expects a JSON payload with all required feature fields.
    Returns predicted Product_Store_Sales_Total.
    """
    product_data = request.get_json()

    sample = {
        "Product_Weight": product_data["Product_Weight"],
        "Product_Sugar_Content": product_data["Product_Sugar_Content"],
        "Product_Allocated_Area": product_data["Product_Allocated_Area"],
        "Product_MRP": product_data["Product_MRP"],
        "Store_Size": product_data["Store_Size"],
        "Store_Location_City_Type": product_data["Store_Location_City_Type"],
        "Store_Type": product_data["Store_Type"],
        "Product_Id_char": product_data["Product_Id_char"],
        "Store_Age_Years": product_data["Store_Age_Years"],
        "Product_Type_Category": product_data["Product_Type_Category"],
    }

    input_df = pd.DataFrame([sample])
    predicted_sales = float(round(model.predict(input_df)[0], 2))

    return jsonify({"Predicted_Sales_Revenue": predicted_sales})


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """
    Batch prediction endpoint.
    Accepts a CSV file upload and returns predictions for every row.
    """
    file = request.files["file"]
    input_df = pd.read_csv(file)

    predictions = model.predict(input_df).tolist()
    predictions_rounded = [round(float(p), 2) for p in predictions]

    # Return index → predicted sales mapping
    output = {str(i): pred for i, pred in enumerate(predictions_rounded)}
    return jsonify(output)


if __name__ == "__main__":
    superkart_api.run(debug=True)
