import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# --- Step 1: Define Allocation Rules (Heuristics) ---
# *** MODIFIED TO STRONGLY EMPHASIZE RISK TOLERANCE AND EXPECTED RETURN ***

def generate_allocation(profile):
    """
    Generates asset allocation considering multiple factors in a balanced way.
    Args:
        profile (dict): Dictionary containing user profile features.
    Returns:
        dict: Dictionary with allocation percentages for each asset class.
    """
    # --- Extract and Normalize Inputs ---
    age = profile['Age']
    income = profile['Income']
    monthly_expenses = profile['Monthly_Expenses']
    risk_tolerance = profile['Risk_Tolerance'] # 1-5
    horizon = profile['Investment_Horizon']     # years
    expected_return_perc = profile['Expected_Return_perc'] # e.g., 4-16

    # Normalize Age (e.g., 20-70 -> 1 (young) to 0 (old))
    age_norm = max(0, min(1, (70 - age) / (70 - 20)))

    # Normalize Horizon (e.g., 1-40 -> 0 (short) to 1 (long))
    horizon_norm = max(0, min(1, (horizon - 1) / (40 - 1)))

    # Combine Age and Horizon for a 'Long-Term Capacity' factor
    long_term_factor = (age_norm + horizon_norm) / 2.0 # Average them

    # Normalize Risk Tolerance (1-5 -> 0 (low) to 1 (high))
    risk_norm = (risk_tolerance - 1) / 4.0

    # Normalize Expected Return (e.g., 4-16% -> 0 (low) to 1 (high))
    # Adjust the 4.0 and 16.0 if your expected range is different
    return_norm = max(0, min(1, (expected_return_perc - 4.0) / (16.0 - 4.0)))

    # Calculate Expense Ratio (Monthly Expenses * 12 / Income)
    # Handle potential zero income
    if income > 0:
        expense_ratio = (monthly_expenses * 12) / income
    else:
        expense_ratio = 1.0 # Assume fully constrained if no income

    # Normalize Expense Ratio (e.g., 0-1 -> 1 (low ratio/flexible) to 0 (high ratio/constrained))
    # Cap expense ratio at 1.0 for this calculation (expenses >= income means 0 flexibility)
    financial_flexibility_norm = max(0, 1.0 - min(expense_ratio, 1.0))


    # --- Base Allocation (Start relatively balanced) ---
    # Using fractions for easier calculation initially
    alloc = {
        'Gold': 0.10,
        'MutualFunds': 0.25,
        'FD': 0.25,
        'Equity': 0.30,
        'RealEstate': 0.10
    }

    # --- Apply Adjustments Based on Factors ---

    # 1. Long-Term Capacity (Age/Horizon) -> Shifts between Growth (Eq/MF) and Safety (FD)
    # Max shift potential (e.g., 20 percentage points total)
    lt_shift_potential = 0.20
    # Shift amount: higher factor -> more growth, less FD
    lt_shift = (long_term_factor - 0.5) * 2 * lt_shift_potential # Scale factor from -lt_shift_potential to +lt_shift_potential
    alloc['Equity'] += lt_shift * 0.6 # Equity gets larger share of shift
    alloc['MutualFunds'] += lt_shift * 0.4 # MF gets smaller share
    alloc['FD'] -= lt_shift # FD decreases proportionally

    # 2. Risk Appetite (Risk Tolerance + Expected Return) -> Adjusts overall risk level
    # Combine risk and return preference (weighted average)
    risk_appetite_factor = (0.6 * risk_norm + 0.4 * return_norm) # Weight risk tolerance slightly higher
    # Max shift potential (e.g., 25 percentage points total)
    risk_shift_potential = 0.25
    # Shift amount: higher appetite -> more Eq/MF/RE, less FD/Gold
    risk_shift = (risk_appetite_factor - 0.5) * 2 * risk_shift_potential
    alloc['Equity'] += risk_shift * 0.40
    alloc['MutualFunds']+= risk_shift * 0.25
    alloc['RealEstate'] += risk_shift * 0.10 # Risky assets increase
    alloc['FD'] -= risk_shift * 0.45
    alloc['Gold'] -= risk_shift * 0.20 # Safer assets decrease

    # 3. Financial Flexibility (Expense Ratio / Income) -> Influences RE and FD/Gold
    # Max shift potential for RE (e.g., 10 percentage points)
    flex_shift_potential_re = 0.10
    # Shift amount: higher flexibility -> potentially more RE, less need for excessive safety buffer
    flex_shift_re = (financial_flexibility_norm - 0.5) * 2 * flex_shift_potential_re
    alloc['RealEstate'] += flex_shift_re

    # Counter-balance the RE shift:
    if flex_shift_re > 0: # If RE increased, take from slightly safer assets
        alloc['FD'] -= flex_shift_re * 0.6
        alloc['Gold'] -= flex_shift_re * 0.4
    else: # If RE decreased (low flexibility), add to safer assets
        alloc['FD'] -= flex_shift_re * 0.7 # Remember flex_shift_re is negative here, so this increases FD
        alloc['Gold'] -= flex_shift_re * 0.3 # Increases Gold

    # --- Post-Adjustment Clamping and Normalization ---

    # Ensure no allocation is negative (set a small floor, e.g., 0.5%)
    min_alloc = 0.005
    alloc = {k: max(v, min_alloc) for k, v in alloc.items()}

    # Normalize to sum to 1.0
    total = sum(alloc.values())
    if total <= 0: # Very unlikely edge case
        num_assets = len(alloc)
        alloc_normalized = {k: 1.0 / num_assets for k in alloc.keys()}
    else:
        alloc_normalized = {k: v / total for k, v in alloc.items()}

    # Convert to percentages
    alloc_perc = {k: round(v * 100, 2) for k, v in alloc_normalized.items()}

    # --- Final Rounding Adjustment ---
    # Ensure sum is exactly 100 due to rounding
    diff = 100.0 - sum(alloc_perc.values())
    if abs(diff) > 0.001:
        try:
            # Add difference to the largest allocation category
            largest_cat = max(alloc_perc, key=alloc_perc.get)
            alloc_perc[largest_cat] = round(alloc_perc[largest_cat] + diff, 2)
            # Ensure the adjustment didn't push it below zero (or minimum)
            alloc_perc[largest_cat] = max(min_alloc * 100, alloc_perc[largest_cat])

            # If adjustments caused issues, re-normalize as a fallback
            final_sum_check = sum(alloc_perc.values())
            if abs(final_sum_check - 100.0) > 0.1: # Use a tolerance
                 current_total = sum(v for v in alloc_perc.values() if v > 0)
                 if current_total > 0:
                    factor = 100.0 / current_total
                    temp_sum = 0
                    keys = list(alloc_perc.keys())
                    for i, k in enumerate(keys):
                         # Ensure non-negative during recalc
                         alloc_perc[k] = max(0, alloc_perc[k])
                         if i < len(keys) - 1:
                              alloc_perc[k] = round(alloc_perc[k] * factor, 2)
                              temp_sum += alloc_perc[k]
                         else: # Last element gets remainder
                              alloc_perc[k] = round(100.0 - temp_sum, 2)

        except ValueError: # Handle cases like all allocations being zero before rounding
             num_assets = len(alloc_perc)
             if num_assets > 0:
                 alloc_perc = {k: round(100.0 / num_assets, 2) for k in alloc_perc.keys()}


    # Final check for any negative values after all adjustments
    alloc_perc_final = {k: max(0, v) for k, v in alloc_perc.items()}
    # One last re-distribution if clamping to zero caused sum != 100
    final_total_abs = sum(alloc_perc_final.values())
    if abs(final_total_abs - 100.0) > 0.1:
        if final_total_abs > 0:
            factor = 100.0 / final_total_abs
            temp_sum = 0
            keys = list(alloc_perc_final.keys())
            for i, k in enumerate(keys):
                if i < len(keys) - 1:
                    alloc_perc_final[k] = round(alloc_perc_final[k] * factor, 2)
                    temp_sum += alloc_perc_final[k]
                else: # Last element gets remainder
                    alloc_perc_final[k] = round(100.0 - temp_sum, 2)

    return alloc_perc_final

# --- Step 2: Generate Synthetic Data (Using the NEW rules) ---

def generate_synthetic_data(num_samples=10000): # Increased samples might help learning complex rules
    data = []
    count = 0
    while count < num_samples:
        profile = {
            'Age': np.random.randint(20, 71),
            'Income': np.random.randint(30000, 300001),
            'Monthly_Expenses': np.random.randint(500, 10001),
            'Risk_Tolerance': np.random.randint(1, 6), # 1 to 5
            'Investment_Horizon': np.random.randint(1, 41), # 1 to 40 years
            'Expected_Return_perc': np.random.uniform(4.0, 16.0) # Desired annual return %
        }

        allocation = generate_allocation(profile)

        # Basic validation: Check if sum is close to 100
        if abs(sum(allocation.values()) - 100.0) < 0.1:
             profile.update({
                 'Gold_Allocation': allocation['Gold'],
                 'MutualFunds_Allocation': allocation['MutualFunds'],
                 'FD_Allocation': allocation['FD'],
                 'Equity_Allocation': allocation['Equity'],
                 'RealEstate_Allocation': allocation['RealEstate']
             })
             data.append(profile)
             count += 1
        # else:
            # print(f"Skipping sample due to normalization issue. Sum: {sum(allocation.values())}, Profile: {profile}")


    return pd.DataFrame(data)

print("Generating synthetic data with new allocation rules...")
synthetic_df = generate_synthetic_data(num_samples=15000) # Maybe generate more data
print(f"Generated {len(synthetic_df)} valid samples.")
# Optional: print(synthetic_df.head())
# Optional: print(synthetic_df.describe())

# Check sums again
# allocation_sums = synthetic_df[['Gold_Allocation', 'MutualFunds_Allocation', 'FD_Allocation', 'Equity_Allocation', 'RealEstate_Allocation']].sum(axis=1)
# print("Checking allocation sums stats:")
# print(allocation_sums.describe())


# --- Step 3: Choose and Train Model (Same Model, New Data) ---

features = ['Age', 'Income', 'Monthly_Expenses', 'Risk_Tolerance', 'Investment_Horizon', 'Expected_Return_perc']
targets = ['Gold_Allocation', 'MutualFunds_Allocation', 'FD_Allocation', 'Equity_Allocation', 'RealEstate_Allocation']

X = synthetic_df[features]
y = synthetic_df[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Using the same model configuration as before
model = RandomForestRegressor(n_estimators=150,
                              random_state=42,
                              n_jobs=-1,
                              max_depth=20,
                              min_samples_split=10,
                              min_samples_leaf=5)

print("Training the model on data reflecting new rules...")
model.fit(X_train_scaled, y_train)
print("Model training complete.")

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel Evaluation (Mean Squared Error on Test Set): {mse:.4f}")
# Lower MSE indicates the model learned the *new* synthetic rules well.


# --- Step 4: Create Prediction Function (Same function structure) ---

def get_financial_advice(user_input):
    """
    Takes user input, preprocesses it, gets prediction from model, and returns advice.
    (Includes normalization post-prediction)
    Args:
        user_input (dict): Dictionary with user's financial details matching features.
    Returns:
        dict: Dictionary containing the suggested asset allocation percentages.
    """
    input_df = pd.DataFrame([user_input])
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)
    predicted_allocations = model.predict(input_scaled)[0]

    # Post-processing: Normalize model output to ensure sum is 100%
    pred_norm = np.maximum(predicted_allocations, 0) # Ensure non-negative
    total = np.sum(pred_norm)
    if total > 0:
        pred_norm = (pred_norm / total) * 100
    else:
        pred_norm = np.full_like(predicted_allocations, 100.0 / len(targets)) # Equal weight if sum is zero

    advice = {target: round(alloc, 2) for target, alloc in zip(targets, pred_norm)}

    # Final rounding adjustment
    final_sum = sum(advice.values())
    diff = 100.0 - final_sum
    if abs(diff) > 0.001:
        largest_key = max(advice, key=advice.get)
        advice[largest_key] = round(advice[largest_key] + diff, 2)
         # Final check to ensure none are negative after adjustment
        if advice[largest_key] < 0:
             # Fallback: Re-normalize if the adjustment made the largest negative
             total_perc = sum(max(0, p) for p in advice.values())
             if total_perc > 0:
                  factor = 100.0 / total_perc
                  temp_sum = 0
                  keys = list(advice.keys())
                  for i, k in enumerate(keys):
                       advice[k] = max(0, advice[k])
                       if i < len(keys) - 1:
                           advice[k] = round(advice[k] * factor, 2)
                           temp_sum += advice[k]
                       else:
                           advice[k] = round(100.0 - temp_sum, 2)

    return advice


# --- Step 5: Example Usage and Disclaimer ---

print("\n--- Financial Advisor Simulation (Risk/Return Focused) ---")

# Example 1: Low Risk, Low Return Expectation
conservative_user = {
    'Age': 60,
    'Income': 60000,
    'Monthly_Expenses': 2000,
    'Risk_Tolerance': 1, # VERY Low
    'Investment_Horizon': 5, # Short
    'Expected_Return_perc': 5.0 # Modest
}
print(f"\nInput Profile (Conservative): {conservative_user}")
suggested_allocation_c = get_financial_advice(conservative_user)
print("Suggested Asset Allocation:")
for asset, percentage in suggested_allocation_c.items():
    print(f"- {asset.replace('_Allocation', '')}: {percentage:.2f}%")
print(f"Check: Total Allocation = {sum(suggested_allocation_c.values()):.2f}%")
# EXPECTATION: High FD/Gold, Low Equity/MF

# Example 2: High Risk, High Return Expectation
aggressive_user = {
    'Age': 30,
    'Income': 120000,
    'Monthly_Expenses': 3000,
    'Risk_Tolerance': 5, # VERY High
    'Investment_Horizon': 30, # Long
    'Expected_Return_perc': 14.0 # High
}
print(f"\nInput Profile (Aggressive): {aggressive_user}")
suggested_allocation_a = get_financial_advice(aggressive_user)
print("Suggested Asset Allocation:")
for asset, percentage in suggested_allocation_a.items():
    print(f"- {asset.replace('_Allocation', '')}: {percentage:.2f}%")
print(f"Check: Total Allocation = {sum(suggested_allocation_a.values()):.2f}%")
# EXPECTATION: High Equity/MF/RealEstate, Low FD/Gold

# Example 3: Balanced Profile
balanced_user = {
    'Age': 45,
    'Income': 90000,
    'Monthly_Expenses': 2800,
    'Risk_Tolerance': 3, # Medium
    'Investment_Horizon': 15, # Medium
    'Expected_Return_perc': 9.0 # Medium
}
print(f"\nInput Profile (Balanced): {balanced_user}")
suggested_allocation_b = get_financial_advice(balanced_user)
print("Suggested Asset Allocation:")
for asset, percentage in suggested_allocation_b.items():
    print(f"- {asset.replace('_Allocation', '')}: {percentage:.2f}%")
print(f"Check: Total Allocation = {sum(suggested_allocation_b.values()):.2f}%")
# EXPECTATION: More balanced allocation across categories


# --- IMPORTANT DISCLAIMER ---
print("\n" + "="*40)
print("          IMPORTANT DISCLAIMER")
print("="*40)
print("This is a simplified simulation based on synthetic data")
print("generated using heuristic rules focused on risk/return.")
print("The model's advice reflects these programmed rules,")
print("NOT professional financial expertise or real market conditions.")
print("\nDO NOT consider this as actual financial advice.")
print("Consult with a qualified human financial advisor before")
print("making any investment decisions.")
print("="*40)

import joblib

# Define filenames
MODEL_FILENAME = "financial_advisor_model.joblib"
SCALER_FILENAME = "scaler.joblib"
FEATURES_FILENAME = "model_features.joblib" # Also save feature names/order

print(f"\nSaving model to {MODEL_FILENAME}")
joblib.dump(model, MODEL_FILENAME)

print(f"Saving scaler to {SCALER_FILENAME}")
joblib.dump(scaler, SCALER_FILENAME)

print(f"Saving feature list to {FEATURES_FILENAME}")
joblib.dump(features, FEATURES_FILENAME) # Save the list of feature names in correct order

print("\nModel, scaler, and feature list saved successfully.")