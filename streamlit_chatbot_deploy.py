import streamlit as st
import pickle
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from dotenv import load_dotenv
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Initialize the Groq client
api_key = os.getenv('GROQ_API_KEY')
client = Groq(api_key=api_key)

# Initialize the ChatGroq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Initialize Streamlit app
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there, how can I help you?"}
        ]

def manage_history(max_length=50):
    if len(st.session_state.messages) > max_length:
        st.session_state.messages = st.session_state.messages[-max_length:]

def add_message(role, content=None, image=None):
    message = {"role": role, "content": content}
    if image:
        message["image"] = image
    st.session_state.messages.append(message)
    manage_history()

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant for urban gardening enthusiasts. Provide plant care tips, composting advice, and plant recommendations."),
    MessagesPlaceholder(variable_name="messages")
])

chain = prompt | llm

# Feature extraction function
def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Plant classification
models = {
            "Decision Tree": "decision_tree_model.pkl",
            # "Random Forest": "random_forest_model.pkl",
            #"SVM": "svm_model.pkl",
            "K-Nearest Neighbors": "knn_model.pkl",
            #"Logistic Regression": "logistic_regression_model.pkl"
        }

def classify_plant(image):
    try:
        IMG_SIZE = 128
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        features = extract_color_histogram(image)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        features = scaler.transform([features])

        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        predictions = {}
        for model_name, model_file in models.items():
            with open(model_file, "rb") as f:
                model = pickle.load(f)
                pred = model.predict(features)
                pred_label = label_encoder.inverse_transform(pred)[0]
                predictions[model_name] = pred_label

        return predictions
    except Exception as e:
        return str(e)

# ChatGPT-like responses

def get_text_response(query):
    add_message("user", content=query)
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    add_message("assistant", content=answer)
    return answer

def get_image_response(image, question):
    combined_input = f"Image uploaded with question: {question}" if question else "Image uploaded."
    add_message("user", content=question, image=image)
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    add_message("assistant", content=answer)
    return answer

# Streamlit interface
initialize_session_state()

st.title("I am Your GrowBuddy 🌱")
st.write("Let me help you with gardening and plant identification!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], caption="User uploaded image", use_container_width=True)
            if message.get("content"):
                st.markdown(message["content"])
        else:
            st.markdown(message["content"])

# User interaction
user_prompt = st.chat_input("Type your question or upload an image below:")
if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            answer = get_text_response(user_prompt)
            st.write(answer)

uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    predictions = classify_plant(image_cv)
    st.subheader("Plant Classification Results")
    for model_name, prediction in predictions.items():
        st.write(f"{model_name}: {prediction}")

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your image..."):
            answer = get_image_response(image, "What type of plant is this?")
            st.write(answer)
