from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

app = Flask(__name__)

# ----------------------------
# Load required files
# ----------------------------
# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load ARMA forecast features
with open("store_type_forecasts.pkl", "rb") as f:
    store_type_forecasts = pickle.load(f)

with open("location_type_forecasts.pkl", "rb") as f:
    location_type_forecasts = pickle.load(f)

with open("region_code_forecasts.pkl", "rb") as f:
    region_code_forecasts = pickle.load(f)

# Load historical training data (used for lag/rolling and encodings)
historical_df = pd.read_csv("train.csv")

# ----------------------------
# Preprocessing Setup
# ----------------------------

# 1. Convert date column
historical_df["Date"] = pd.to_datetime(historical_df["Date"])

# 2. Cast types
historical_df["Holiday"] = historical_df["Holiday"].astype(bool)
# NOTE: Commenting out this cast since Discount is likely numeric
# historical_df["Discount"] = historical_df["Discount"].astype(bool)

# 3. Ensure Store_id is string for target encoding
historical_df["Store_id"] = historical_df["Store_id"].astype(str)

# 4. Compute log of Sales
historical_df["Sales_log"] = np.log1p(historical_df["Sales"])

# 5. Target Encoding of Store_id
store_mean_mapping = historical_df.groupby("Store_id")["Sales_log"].mean().to_dict()
global_mean = historical_df["Sales_log"].mean()

# 6. One-hot encoding
historical_df = pd.get_dummies(historical_df, columns=["Store_Type", "Location_Type", "Region_Code"], drop_first=False)

# Save list of encoded columns to align test set later
encoded_columns = [col for col in historical_df.columns if any(x in col for x in ["Store_Type_", "Location_Type_", "Region_Code_"])]

def preprocess_test_data(test_df, historical_df):
    # 1. Date & type conversions
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df["Holiday"] = test_df["Holiday"].astype(bool)
    # Do NOT cast Discount to bool if it is numeric
    # test_df["Discount"] = test_df["Discount"].astype(bool) 
    

    test_df["Store_id"] = test_df["Store_id"].astype(str)

    # 2. Add time-based features
    test_df["Month"] = test_df["Date"].dt.month
    test_df["Week"] = test_df["Date"].dt.isocalendar().week
    test_df["Weekday"] = test_df["Date"].dt.weekday

    # 3. Merge for lag/rolling sales features
    hist = historical_df[["Store_id", "Date", "Sales_log"]].copy()
    merged_df = pd.concat([hist, test_df], axis=0, ignore_index=True).sort_values("Date")

    # 4. Create lag and rolling features
    lags = [1, 2, 3, 4, 7]
    for lag in lags:
        merged_df[f"Sales_lag_{lag}"] = merged_df.groupby("Store_id")["Sales_log"].shift(lag)
    merged_df["Sales_roll_7"] = merged_df.groupby("Store_id")["Sales_log"].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    merged_df["Sales_roll_14"] = merged_df.groupby("Store_id")["Sales_log"].transform(lambda x: x.rolling(window=14, min_periods=1).mean())

    # 5. Filter back only test rows
    test_df = merged_df[merged_df["Sales_log"].isna()].copy()

    # 6. One-hot encode & align columns
    test_df = pd.get_dummies(test_df, columns=["Store_Type", "Location_Type", "Region_Code"], drop_first=False)
    for col in encoded_columns:
        if col not in test_df:
            test_df[col] = 0
    test_df = test_df.reindex(columns=[*test_df.columns[:-len(encoded_columns)], *encoded_columns])

    # 7. Add ARMA forecast features
    def map_arma(df, arma_dict, prefix):
        for col in arma_dict:
            mask = df[col] == 1
            df.loc[mask, f"{prefix}_ARMA_Pred"] = df.loc[mask, "Date"].map(arma_dict[col])
        return df

    test_df = map_arma(test_df, store_type_forecasts, "Store_Type")
    test_df = map_arma(test_df, location_type_forecasts, "Location_Type")
    test_df = map_arma(test_df, region_code_forecasts, "Region_Code")

    # 8. Drop columns not needed by model
    test_df.drop(columns=["Sales_log", "Store_id", "Date"], inplace=True, errors="ignore")

    # 9. Align columns to model input if available
    if hasattr(model, "feature_names_in_"):
        test_df = test_df.reindex(columns=model.feature_names_in_, fill_value=0)

    return test_df

@app.route("/")

def home():
    return "Welcome to Product Sales Forecasting"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        instructions = {
            "message": "Send POST requests to this endpoint to get sales predictions.",
            "usage": {
                "json": {
                    "description": "Send a single row or list of rows as JSON.",
                    "example_single": {
                        "Store_id": "123",
                        "Date": "2025-05-20",
                        "Store_Type": "TypeA",
                        "Location_Type": "Urban",
                        "Region_Code": "R1",
                        "Holiday": False,
                        "Discount": 0.1
                    },
                    "example_batch": [
                        {
                            "Store_id": "123",
                            "Date": "2025-05-20",
                            "Store_Type": "TypeA",
                            "Location_Type": "Urban",
                            "Region_Code": "R1",
                            "Holiday": False,
                            "Discount": 0.1
                        },
                        {
                            "Store_id": "456",
                            "Date": "2025-05-21",
                            "Store_Type": "TypeB",
                            "Location_Type": "Rural",
                            "Region_Code": "R2",
                            "Holiday": True,
                            "Discount": 0.2
                        }
                    ]
                },
                "csv": {
                    "description": "Send CSV file with columns matching model features.",
                    "content_type": "multipart/form-data",
                    "form_field_name": "file"
                }
            }
        }
        return jsonify(instructions)

    # POST request
    # Handle JSON input: single or batch
    if request.is_json:
        try:
            input_data = request.get_json()
            # Support both single dict or list of dicts
            if isinstance(input_data, dict):
                test_df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                test_df = pd.DataFrame(input_data)
            else:
                return jsonify({"error": "JSON payload must be a dict or list of dicts"}), 400

            processed = preprocess_test_data(test_df, historical_df)
            preds_log = model.predict(processed)
            preds = np.expm1(preds_log)
            
            if len(preds) == 1:
                return jsonify({"prediction": float(preds[0])})
            else:
                return jsonify({"predictions": preds.tolist()})

        except Exception as e:
            return jsonify({"error": f"Invalid JSON input: {str(e)}"}), 400

    # Handle CSV file upload
    file = request.files.get("file")
    if file:
        try:
            test_df = pd.read_csv(file)
            processed = preprocess_test_data(test_df, historical_df)
            preds_log = model.predict(processed)
            preds = np.expm1(preds_log)
            return jsonify({"predictions": preds.tolist()})
        except Exception as e:
            return jsonify({"error": f"Invalid CSV input: {str(e)}"}), 500

    # If no valid input found
    return jsonify({"error": "No valid input provided. Please send JSON payload or upload CSV file with 'file' field."}), 400

if __name__ == "__main__":
    app.run(debug=True)
