import streamlit as st

def main():    
        # st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

        st.title("🤖 AI Chatbot")
        st.markdown("Chat with your personal AI assistant!")

        # session state to store chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You:", key="user_input")
        if st.button("Send") and user_input.strip() != "":
            st.session_state.chat_history.append(("user", user_input))
            
            # todo: actual bot response
            bot_response = "I'm still learning! But I'm here to help soon. 🙂"
            st.session_state.chat_history.append(("bot", bot_response))
            st.session_state.user_input = ""

        # display chat history
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f"🧑‍💻 **You:** {message}")
            else:
                st.markdown(f"🤖 **Bot:** {message}")

