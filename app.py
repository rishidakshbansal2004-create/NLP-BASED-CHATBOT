import streamlit as st
import datetime
import json
from preprocessing import preprocess
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import random

st.set_page_config(page_title="NLP Chatbot", page_icon="🤖")

# Load intents for responses
with open("intent.json", "r") as f:
    data = json.load(f)

@st.cache_resource
def load_model():
    patterns = []
    tags = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern)
            tags.append(intent["tag"])
    
    cleaned = [" ".join(preprocess(p)) for p in patterns]
    
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(cleaned)
    
    model = SVC(kernel="linear", probability=True)
    model.fit(X, tags)
    
    print("Model trained successfully!")
    return model, vectorizer

model, vectorizer = load_model()

def get_response(user_input, model, vectorizer):
    cleaned = " ".join(preprocess(user_input))
    X = vectorizer.transform([cleaned])
    intent = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    if confidence < 0.3:
        intent = "fallback"
    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
    return "Sorry, I didn't understand that!"

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
    st.write("🙃 Small Talk")
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
        "time": datetime.datetime.now().strftime("%H:%M")
    })
    response = get_response(user_input, model, vectorizer)
    st.session_state.messages.append({
        "role": "bot",
        "content": response,
        "time": datetime.datetime.now().strftime("%H:%M")
    })
    st.rerun()
