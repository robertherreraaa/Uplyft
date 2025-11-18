import pandas as pd
import numpy as np
import time
import zipfile
import os
import streamlit as st
from openai import OpenAI
from transformers import pipeline
from dotenv import load_dotenv


# Load dataset

zip_file_path = r'sentiment_analyzer.zip'
file_folder_path = r''
with zipfile.ZipFile(zip_file_path) as zip_ref:
    zip_ref.extractall(file_folder_path)

file_path = r'sentiment_analysis.csv'
df = pd.read_csv(file_path)


# Sentiment model

def load_sentiment_model():
    model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    return model

sentiment_model = load_sentiment_model()

def predict_sentiment(text):
    result = sentiment_model(text)[0]
    label = result['label'].lower()
    return label


# Streamlit UI setup

st.set_page_config(layout='wide', page_title="Uplyft")

st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(to bottom right, #87CEEB, #B0E0E6);
    background-attachment: fixed;
}

/* Center container card */
.main > div {
    background-color: rgba(255,255,255,0.95);
    border-radius: 20px;
    padding: 3rem 4rem;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.08);
    max-width: 900px;
    margin: 2rem auto;
}

/* Headings */
h1, h2, h3, h4 { color: #003366 !important; }

/* Buttons */
.stButton>button {
    background-color: #4682B4 !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 500 !important;
    transition: transform 0.2s ease;
}
.stButton>button:hover {
    background-color: #5A9BD5 !important;
    transform: scale(1.05);
}

/* Input fields */
input, select, textarea {
    border-radius: 10px !important;
    background-color: #f9f9f9 !important;
}

/* Chat bubbles */
.stChatMessage { border-radius: 15px !important; padding: 12px 16px !important; margin-bottom: 12px !important; }
.stChatMessage.user { background-color: rgba(255,255,255,0.9) !important; border: 1px solid #D3D3D3 !important; color: #003366 !important; }
.stChatMessage.assistant { background-color: rgba(173,216,230,0.85) !important; border: 1px solid #ADD8E6 !important; color: #00264D !important; }

/* Sentiment caption */
.stCaption { font-style: italic; color: #002F5E !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background-color: #87CEEB; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "Home"

def go_to(page):
    st.session_state.page = page
    st.rerun()

def reset_user_session():
    for key in ["name", "age", "gender", "hobby", "happiness", "chat_history"]:
        if key in st.session_state:
            del st.session_state[key]


if st.session_state.page == "Home":
    st.title("Uplyft")
    st.write("Lift your life. Find your purpose. ✨")
    st.markdown("---")
    st.subheader("🌈 Please complete the details below:")

    # Save inputs to session_state
    name = st.text_input("👤 What is your name?", value=st.session_state.get("name",""))
    age = st.number_input("🎂 How old are you?", min_value=0, max_value=75, value=st.session_state.get("age",0))
    gender = st.selectbox("🚻 Your gender or preference?", ["Male","Female","Other"])
    hobby = st.text_input("🎨 Your hobby or passion in life")
    happiness = st.select_slider("😊 Rate your happiness currently (1–10)", [1,2,3,4,5,6,7,8,9,10])

    # Store values in session_state
    st.session_state["name"] = name
    st.session_state["age"] = age
    st.session_state["gender"] = gender
    st.session_state["hobby"] = hobby
    st.session_state["happiness"] = happiness

    if st.button("Are you ready to Uplyft yourself?"):
        progress_text = 'Running...'
        bar = st.progress(0, text=progress_text)
        for i in range(100):
            time.sleep(0.02)
            bar.progress(i + 1, text=f"Progress: {i + 1}")
        st.success('Done! 🎉')
        st.balloons()
        time.sleep(3)
        go_to("Next Page")

elif st.session_state.page == "Next Page":
    st.title("What's on your mind? Let's talk about it.💬")
    st.write("I'm your friendly life coach. Let's explore how you're feeling and find a way forward 💫")


    # Load OpenAI API Key

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    # Chat Interface

    st.markdown("---")
    st.subheader("🧘🏻‍♂️ Life Coach Chat")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("What are you feeling or thinking right now, at this very moment? Let's talk about it...")

    if user_input:
        sentiment = predict_sentiment(user_input)

        # Show typing feedback
        with st.spinner("Life Coach is thinking... 🧏🏻‍♂️"):
            time.sleep(2)

        user_name = st.session_state.get("name", "my friend")

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
            {
                'role': 'system',
                'content': f'''You are a warm, supportive life coach named Uplyft. 
                The user's name is {user_name}. 
                The user's current mood sentiment is "{sentiment}".

                - If sentiment is negative → be compassionate, provide emotional support, and suggest gentle steps. 
                - If sentiment is positive → encourage and celebrate their success. 
                - If sentiment is neutral → explore their feelings or help find motivation. 

                Address the user by name naturally in your reply (e.g., "Hey {user_name}, I’m really glad you shared that."). 
                Keep your tone brief but caring.
                '''
            },
            {'role': 'user', 'content': user_input}
            ]
        )

        bot_reply = response.choices[0].message.content

        # Save to chat history
        st.session_state.chat_history.append(("sentiment", sentiment))
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", bot_reply))


    # Display chat history

    for i, (role, msg) in enumerate(st.session_state.chat_history):
        if role == "user":
            st.chat_message("user").markdown(f"**You:** {msg}")
        elif role == "bot":
            st.chat_message("assistant").markdown(f"**Life Coach:** {msg}")
        elif role == "sentiment":
            st.caption(f"🧭 Detected mood: *{msg.capitalize()}*")


    # Add button to clear and end the chat and go back to main page

    if st.button("Let's end our session for today 👋🏻👦🏻"):
        st.success(f"It’s been lovely talking with you, {st.session_state.name or 'my dear friend'} 🌿 "
               "Remember to be kind to yourself — you’re doing your best. Take care! 💖")
        st.session_state.chat_history = []
        reset_user_session()
        time.sleep(3)
        go_to("Home")