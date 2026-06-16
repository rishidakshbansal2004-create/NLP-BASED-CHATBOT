
import streamlit as st
from Chatbot import get_response
import pickle
import datetime
st.set_page_config(page_title="NLP Chatbot", page_icon="🤖")

time = datetime.datetime.now().strftime("%H:%M")
@st.cache_resource  ### I USED THIS DECORATOR TO AVOID LOADING THE MODEL EVERY TIME THE USER SENDS A MESSAGE, IT WILL LOAD ONCE AND THEN CACHE IT FOR FUTURE USE.
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Model loaded successfully!")
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("<h1 style='text-align: center;'>🤖 Personal Chatbot</h1>", unsafe_allow_html=True)
with st.sidebar:
    st.title("About")
    st.write("This is an NLP chatbot built by Rishi using Python, NLTK and Sklearn.")
    st.write("**Intents it understands:**")
    st.write("👋 Greeting")
    st.write("😂 Jokes")
    st.write("💪 Motivation")
    st.write("ℹ️ About")
    st.write("🆘 Help")
    st.write("🙏 Thanks")
    st.write("👋 Farewell")
    st.write(" 🙃 Small Talk")
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": "Hey! I am your Personal Chatbot 🤖 Ask me for a joke, motivation, or just say hi!",
        "time": datetime.datetime.now().strftime("%H:%M")
    })
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-end; margin: 8px 0;'>
                <div style='background-color: #dcf8c6; padding: 10px 15px; 
                border-radius: 18px 18px 0px 18px; max-width: 70%; color: black;'>
                    {message['content']}
               <div style='font-size: 11px; color: gray; text-align: right; margin-top: 4px;'>
            {message.get('time', '')}
            </div>
            </div>
                <span style='margin-left: 8px; font-size: 20px;'>👤</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; justify-content: flex-start; margin: 8px 0;'>
                <span style='margin-right: 8px; font-size: 20px;'>🤖</span>
                <div style='background-color: #f0f0f0; padding: 10px 15px;
                border-radius: 18px 18px 18px 0px; max-width: 70%; color: black;'>
                    {message['content']}
                <div style='font-size: 11px; color: gray; text-align: right; margin-top: 4px;'>
            {message.get('time', '')}
             </div>
        </div>
            </div>
            """,
            unsafe_allow_html=True
        )


st.divider()
col1, col2 = st.columns([8, 1], vertical_alignment="bottom")

with col1:
    user_input = st.text_input("Type a message...", key="input")
with col2:
    send = st.button("Send")


if send and user_input.strip():
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": datetime.datetime.now().strftime(" %H:%M")
    })
    response = get_response(user_input, model, vectorizer)
    st.session_state.messages.append({
        "role": "bot",
        "content": response,
        "time": datetime.datetime.now().strftime(" %H:%M")
    })
    st.rerun()