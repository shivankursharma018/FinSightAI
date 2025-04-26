import fastapi
import uvicorn
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import os

# --- Configuration ---
MODEL_FILENAME = "financial_advisor_model.joblib"
SCALER_FILENAME = "scaler.joblib"
FEATURES_FILENAME = "model_features.joblib"
TARGET_NAMES = ['Gold_Allocation', 'MutualFunds_Allocation', 'FD_Allocation', 'Equity_Allocation', 'RealEstate_Allocation'] # These correspond to the *percentage* prediction targets

# --- Global Variables for Loaded Objects ---
model = None
scaler = None
features = None # This will hold the feature names for the *input* profile

# --- Lifespan Management (Keep as before) ---
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    global model, scaler, features
    print("Lifespan startup: Loading model and associated objects...")
    try:
        # File existence checks... (same as before)
        if not os.path.exists(MODEL_FILENAME): raise FileNotFoundError(f"Model file not found: {MODEL_FILENAME}")
        if not os.path.exists(SCALER_FILENAME): raise FileNotFoundError(f"Scaler file not found: {SCALER_FILENAME}")
        if not os.path.exists(FEATURES_FILENAME): raise FileNotFoundError(f"Features file not found: {FEATURES_FILENAME}")

        model = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
        features = joblib.load(FEATURES_FILENAME) # Load feature names expected by model
        print("Lifespan startup: Model, scaler, and features loaded successfully.")
        print(f"Lifespan startup: Model expects profile features: {features}")

    except Exception as e:
        print(f"Lifespan startup Error: {e}")
        model = None
    yield
    print("Lifespan shutdown: Cleaning up resources...")
    model = None
    scaler = None
    features = None


# --- Pydantic Models ---

# UPDATED: Input model now includes Total_Investment_Amount
class UserProfile(BaseModel):
    Age: int = Field(..., gt=17, lt=120, description="User's Age (18-119)")
    Income: float = Field(..., gt=0, description="User's Annual Income")
    Monthly_Expenses: float = Field(..., gt=0, description="User's Average Monthly Expenses")
    Risk_Tolerance: int = Field(..., ge=1, le=5, description="User's Risk Tolerance (1=Low, 5=High)")
    Investment_Horizon: int = Field(..., gt=0, lt=100, description="Investment Horizon in Years (1-99)")
    Expected_Return_perc: float = Field(..., ge=0, le=50, description="Desired Annual Return Percentage (0-50)")
    Total_Investment_Amount: float = Field(..., gt=0, description="Total amount of money to invest") # NEW FIELD

# UPDATED: Output model now shows amounts
class AllocationAdvice(BaseModel):
    Gold_Amount: float = Field(..., description="Amount to invest in Gold")
    MutualFunds_Amount: float = Field(..., description="Amount to invest in Mutual Funds")
    FD_Amount: float = Field(..., description="Amount to invest in Fixed Deposits")
    Equity_Amount: float = Field(..., description="Amount to invest in Equity")
    RealEstate_Amount: float = Field(..., description="Amount to invest in Real Estate")
    Total_Allocated_Amount: float = Field(..., description="Sum of amounts allocated (for verification)") # RENAMED/PURPOSE CHANGED
    Warnings: Optional[List[str]] = None # Optional field for rounding warnings etc.
    Disclaimer: str

# --- FastAPI Application (Keep as before) ---
app = fastapi.FastAPI(
    title="Synthetic Financial Advisor API",
    description="Provides investment allocation advice (in amounts) based on user profile and total investment amount.",
    version="0.2.0", # Increment version
    lifespan=lifespan
)

# --- Helper Function (Still returns percentages) ---
# This function remains unchanged as the ML model predicts percentages
def get_percentage_prediction_and_normalize(input_data: pd.DataFrame) -> Dict[str, float]:
    """
    Scales profile input, predicts allocation percentages, and normalizes.
    Returns a dictionary mapping asset class name (from TARGET_NAMES) to percentage.
    """
    if model is None or scaler is None or features is None:
        print("Error: Model/Scaler/Features not available for prediction.")
        raise fastapi.HTTPException(status_code=503, detail="Model not loaded or failed to load.")

    try:
        # Ensure only the features needed by the model are used for scaling/prediction
        profile_features_df = input_data[features]
    except KeyError as e:
         print(f"Prediction Error: Input data missing expected profile feature {e}. Expected: {features}")
         raise fastapi.HTTPException(status_code=400, detail=f"Input data missing expected profile feature: {e}. Expected features are: {features}")

    input_scaled = scaler.transform(profile_features_df)
    predicted_allocations = model.predict(input_scaled)[0] # Raw percentage predictions

    # --- Normalization Logic (Keep exactly as before) ---
    pred_norm = np.maximum(predicted_allocations, 0)
    total = np.sum(pred_norm)
    if total > 0: pred_norm = (pred_norm / total) * 100
    else: pred_norm = np.full_like(predicted_allocations, 100.0 / len(TARGET_NAMES))

    advice_perc_raw = {target: round(alloc, 2) for target, alloc in zip(TARGET_NAMES, pred_norm)}

    final_sum = sum(advice_perc_raw.values())
    diff = 100.0 - final_sum
    if abs(diff) > 0.001:
        try:
            largest_key = max(advice_perc_raw, key=advice_perc_raw.get)
            advice_perc_raw[largest_key] = round(advice_perc_raw[largest_key] + diff, 2)
            advice_perc_raw[largest_key] = max(0, advice_perc_raw[largest_key]) # Ensure non-negative

            final_sum_check = sum(advice_perc_raw.values())
            if abs(final_sum_check - 100.0) > 0.1:
                 current_total = sum(v for v in advice_perc_raw.values() if v > 0)
                 if current_total > 0:
                    factor = 100.0 / current_total
                    temp_sum = 0
                    keys = list(advice_perc_raw.keys())
                    for i, k in enumerate(keys):
                         advice_perc_raw[k] = max(0, advice_perc_raw[k])
                         if i < len(keys) - 1: advice_perc_raw[k] = round(advice_perc_raw[k] * factor, 2); temp_sum += advice_perc_raw[k]
                         else: advice_perc_raw[k] = round(100.0 - temp_sum, 2)
        except ValueError: pass # Ignore errors if dict empty

    advice_perc_final = {k: max(0, v) for k, v in advice_perc_raw.items()}
    # Final check/renormalize if needed (same as before)
    final_total_abs = sum(advice_perc_final.values())
    if abs(final_total_abs - 100.0) > 0.1:
        if final_total_abs > 0:
            factor = 100.0 / final_total_abs; temp_sum = 0; keys = list(advice_perc_final.keys())
            for i, k in enumerate(keys):
                if i < len(keys) - 1: advice_perc_final[k] = round(advice_perc_final[k] * factor, 2); temp_sum += advice_perc_final[k]
                else: advice_perc_final[k] = round(100.0 - temp_sum, 2)

    return advice_perc_final # Returns percentages summing to 100

# --- API Endpoints ---

@app.get("/", tags=["Status"])
async def read_root():
    # Keep as before - checks model readiness
    model_loaded_status = model is not None and scaler is not None and features is not None
    return {
        "status": "OK" if model_loaded_status else "Error",
        "message": "Synthetic Financial Advisor API is running." if model_loaded_status else "Model/Scaler/Features failed to load.",
        "model_ready": model_loaded_status,
        "docs_url": "/docs"
        }

# UPDATED: /advise endpoint logic
@app.post("/advise", response_model=AllocationAdvice, tags=["Advisor"])
async def get_advice(profile: UserProfile):
    """
    Receives user profile data and total investment amount,
    returns suggested investment allocation in monetary amounts.
    """
    try:
        # 1. Extract profile data for percentage prediction
        profile_dict = profile.dict()
        total_investment = profile.Total_Investment_Amount # Get the new input
        input_df = pd.DataFrame([profile_dict])

        # 2. Get Allocation Percentages using the helper function
        advice_percentages = get_percentage_prediction_and_normalize(input_df)

        # 3. Calculate Amounts
        advice_amounts = {}
        calculated_total = 0.0
        asset_keys_map = { # Map percentage keys to amount keys in output model
             'Gold_Allocation': 'Gold_Amount',
             'MutualFunds_Allocation': 'MutualFunds_Amount',
             'FD_Allocation': 'FD_Amount',
             'Equity_Allocation': 'Equity_Amount',
             'RealEstate_Allocation': 'RealEstate_Amount'
        }

        for perc_key, amount_key in asset_keys_map.items():
            percentage = advice_percentages.get(perc_key, 0.0) # Get percentage (default to 0 if missing)
            amount = round((percentage / 100.0) * total_investment, 2) # Calculate amount, round to 2 decimals
            advice_amounts[amount_key] = amount
            calculated_total += amount

        # 4. Add Verification Total and Disclaimer
        advice_amounts["Total_Allocated_Amount"] = round(calculated_total, 2)
        advice_amounts["Disclaimer"] = ("IMPORTANT: This is a simulation based on synthetic data and simplified rules. "
                                        "It is NOT real financial advice. Consult a qualified human advisor.")

        # Add warning if allocated total doesn't match input due to rounding
        warnings = []
        if abs(advice_amounts["Total_Allocated_Amount"] - total_investment) > 0.01 * len(asset_keys_map): # Allow tolerance based on number of items
            warnings.append(f"Total allocated amount ({advice_amounts['Total_Allocated_Amount']:.2f}) may differ slightly from requested investment amount ({total_investment:.2f}) due to rounding.")
            advice_amounts["Warnings"] = warnings

        # 5. Return using the updated AllocationAdvice model
        return AllocationAdvice(**advice_amounts)

    # Keep existing error handling
    except fastapi.HTTPException as e:
         raise e # Re-raise HTTPExceptions (like 503 or 400)
    except KeyError as e: # Should be less likely now with explicit feature selection
         print(f"Error: Key error during processing. Details: {e}")
         raise fastapi.HTTPException(status_code=400, detail=f"Data processing error, potentially missing key: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in /advise endpoint: {e}")
        # Log traceback in production
        raise fastapi.HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Run the Server (Keep as before) ---
if __name__ == "__main__":
    print("Starting FastAPI server via script execution...")
    uvicorn.run("fin_api_m2:app", host="0.0.0.0", port=8007, reload=True)