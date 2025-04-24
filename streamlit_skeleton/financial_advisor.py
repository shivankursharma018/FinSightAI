# import streamlit as st

# def main():
#     # st.set_page_config(page_title="Investment Advisor", page_icon="💼")

#     st.title("💼 Financial Investment Advisor")
#     st.markdown("Get tailored suggestions based on your financial goals.")

#     with st.form("advisor_form"):
#         age = st.number_input("Your Age", min_value=18, max_value=100, value=30)
#         risk_profile = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
#         objective = st.selectbox("Investment Objective", ["Wealth Growth", "Retirement", "Tax Saving", "Education"])
#         duration = st.selectbox("Investment Duration", ["< 1 year", "1-3 years", "3-5 years", "5+ years"])
#         amount = st.number_input("Investment Amount ($)", min_value=100, value=5000, step=100)
        
#         submit = st.form_submit_button("Get Suggestions")

#     if submit:
#         st.subheader("📊 Suggested Strategy")
        
#         # todo: replace with model
#         if risk_profile == "High":
#             advice = "Explore Equity Mutual Funds, Stocks, and SIPs in growth sectors."
#         elif risk_profile == "Medium":
#             advice = "A mix of Mutual Funds and Bonds could balance risk and return."
#         else:
#             advice = "Consider PPF, Government Bonds, and Fixed Deposits for stable returns."
        
#         st.markdown(f"**Profile Summary:** {risk_profile} risk tolerance, {objective.lower()} goal, for {duration.lower()}.\n\n💡 {advice}")

#         # Optional: Show a download link for PDF summary (later enhancement)

import streamlit as st

def main():

    st.title("💼 AI Financial Advisor")
    st.write("Fill in your profile to get personalized investment advice.")

    # User Input Form
    with st.form("advisor_form"):
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 18, 100, 30)
        avenues = st.multiselect(
            "Investment Avenues Interested In",
            ["Mutual Funds", "Equity Market", "Debentures", "Government Bonds", "Fixed Deposits", "PPF", "Gold"]
        )
        objective = st.text_input("Investment Objectives")
        purpose = st.text_input("Investment Purpose")
        duration = st.selectbox("Investment Duration", ["<1 year", "1-3 years", "3-5 years", "5+ years"])
        expect = st.text_input("Expected Returns (%)")
        savings_goal = st.text_input("Savings Objectives")
        source = st.selectbox("How did you hear about investing?", ["Friends", "Online", "Advisor", "Other"])

        submitted = st.form_submit_button("Get Advice")

    if submitted:
        profile = f"""
        Gender: {gender}
        Age: {age}
        Interested In: {', '.join(avenues)}
        Objective: {objective}
        Purpose: {purpose}
        Duration: {duration}
        Expected Return: {expect}%
        Savings Goal: {savings_goal}
        Source: {source}
        """

        st.subheader("🧠 Suggested Strategy")
        # placeholder response (replace with model later)
        st.success(f"Based on your inputs, consider diversifying across: {', '.join(avenues)}.\n\n""A long-term goal might suit equity or mutual funds. For shorter horizons, fixed deposits or gold may help.")
