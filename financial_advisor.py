# uvicorn fin_api:app --host 0.0.0.0 --port 8000
# streamlit run .\main.py

# financial_advisor.py
import streamlit as st
import requests
import json


def main():
    # st.set_page_config(page_title="AI Financial Advisor", layout="centered")

    st.title("💼 AI Financial Advisor")
    st.markdown("""
    Welcome to your **AI-powered investment assistant**.  
    Fill in your profile to receive **tailored financial advice** aligned with your goals.
    """)

    # --- User Input Form ---
    with st.form("advisor_form"):
        st.markdown("### 👤 Personal Information")
        cols = st.columns(2)
        with cols[0]:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with cols[1]:
            age = st.number_input("Age", min_value=18, value=30, step=1)

        st.markdown("### 💡 Investment Preferences")
        investment_avenues_interested = st.radio("Interested in Investment Avenues?", ["Yes", "No"])

        st.markdown("**Rate your preference for each investment type** _(1 = Low, 10 = High)_")
        pref_cols = st.columns(2)
        with pref_cols[0]:
            mutual_funds = st.slider("Mutual Funds", 1, 10, 5)
            equity_market = st.slider("Equity Market", 1, 10, 5)
            debentures = st.slider("Debentures", 1, 10, 5)
            government_bonds = st.slider("Government Bonds", 1, 10, 5)
        with pref_cols[1]:
            fixed_deposits = st.slider("Fixed Deposits", 1, 10, 5)
            ppf = st.slider("PPF", 1, 10, 5)
            gold = st.slider("Gold", 1, 10, 5)

        st.markdown("### 🎯 Investment Goals")
        goal_cols = st.columns(2)
        with goal_cols[0]:
            investment_objectives = st.text_input("Investment Objectives", "Growth")
            investment_purpose = st.text_input("Investment Purpose", "Wealth Creation")
            investment_duration = st.selectbox("Investment Duration", ["<1 year", "1-3 years", "3-5 years", "5+ years"])
        with goal_cols[1]:
            expected_returns = st.text_input("Expected Returns", "10%-20%")
            savings_objective = st.text_input("Savings Objective", "Education")
            source_of_information = st.selectbox("Source of Information",
                                                 ["Friends", "Online", "Advisor", "Television", "Other"])

        submitted = st.form_submit_button("🚀 Get Advice")

    # --- API Call ---
    if submitted:
        payload = {
            "Gender": gender,
            "Age": age,
            "Investment_Avenues_Interested": investment_avenues_interested,
            "Preferred_Investments": {
                "Mutual Funds": mutual_funds,
                "Equity Market": equity_market,
                "Debentures": debentures,
                "Government Bonds": government_bonds,
                "Fixed Deposits": fixed_deposits,
                "PPF": ppf,
                "Gold": gold
            },
            "Investment_Objectives": investment_objectives,
            "Investment_Purpose": investment_purpose,
            "Investment_Duration": investment_duration,
            "Expected_Returns": expected_returns,
            "Savings_Objective": savings_objective,
            "Source_of_Information": source_of_information
        }

        with st.spinner("Generating your personalized advice..."):
            try:
                response = requests.post("http://localhost:8000/generate_advice", json=payload)
                response.raise_for_status()
                result = response.json()

                st.success("✅ Advice Generated Successfully!")
                st.markdown("### 🧠 Your Personalized Advice")
                st.info(result["response"])

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Error connecting to the API:\n{str(e)}")
            except json.JSONDecodeError:
                st.error("❌ Error: Invalid response from the API.")
            except KeyError:
                st.error("❌ Error: Unexpected response format. Missing 'response' field.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred:\n{str(e)}")
    # --- SECTION 2: Portfolio Advisor (New API Integration) ---
    with st.expander("📊 Portfolio Advisor (Model-Based Allocation)", expanded=True):
        st.markdown("### 📋 Enter your financial profile for model-generated investment allocation:")

        with st.form("portfolio_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=18, value=30, step=1, key="age2")
                income = st.number_input("Annual Income (₹)", min_value=1000.0, value=600000.0, step=1000.0)
                expenses = st.number_input("Monthly Expenses (₹)", min_value=500.0, value=20000.0, step=500.0)
            with col2:
                risk = st.slider("Risk Tolerance (1 = Low, 5 = High)", 1, 5, 3)
                horizon = st.number_input("Investment Horizon (Years)", min_value=1, max_value=99, value=5)
                expected_return = st.number_input("Expected Annual Return (%)", min_value=0.0, max_value=50.0,
                                                  value=12.0)
                total_investment = st.number_input("Total Investment Amount (₹)", min_value=1000.0, value=100000.0,
                                                   step=1000.0)

            portfolio_submit = st.form_submit_button("🧠 Get Portfolio Advice")

        if portfolio_submit:
            profile = {
                "Age": age,
                "Income": income,
                "Monthly_Expenses": expenses,
                "Risk_Tolerance": risk,
                "Investment_Horizon": horizon,
                "Expected_Return_perc": expected_return,
                "Total_Investment_Amount": total_investment
            }

            with st.spinner("Fetching model-based portfolio advice..."):
                try:
                    response = requests.post("http://localhost:8007/advise", json=profile)
                    response.raise_for_status()
                    result = response.json()

                    st.success("✅ Portfolio Allocation Ready!")
                    st.markdown("### 💼 Recommended Investment Allocation")
                    st.markdown(f"""
                    - 🪙 **Gold**: ₹{result['Gold_Amount']}
                    - 📈 **Mutual Funds**: ₹{result['MutualFunds_Amount']}
                    - 🏦 **Fixed Deposits**: ₹{result['FD_Amount']}
                    - 📊 **Equity**: ₹{result['Equity_Amount']}
                    - 🏠 **Real Estate**: ₹{result['RealEstate_Amount']}
                    - 💰 **Total Allocated**: ₹{result['Total_Allocated_Amount']}
                    """)

                    if result.get("Warnings"):
                        for warn in result["Warnings"]:
                            st.warning(warn)

                    st.caption(f"📌 {result['Disclaimer']}")

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Error connecting to the Portfolio API:\n{str(e)}")
                except json.JSONDecodeError:
                    st.error("❌ Error: Invalid response from the Portfolio API.")
                except KeyError:
                    st.error("❌ Error: Unexpected format. Missing expected keys.")
                except Exception as e:
                    st.error(f"❌ Unexpected error:\n{str(e)}")

