import streamlit as st
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
from dotenv import load_dotenv
load_dotenv()

# Get the API key from the environment variable
api_key=os.getenv('GROQ_API_KEY')

# Create a Groq client
client = Groq(api_key=api_key)

# Initialize the ChatGroq LLM
llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature=0
)

# App title and description
st.title("I am Your GrowBuddy 🌱")
st.write("Let me help u start gardening. Let's grow together.")

# Initialize chat history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, how can I help you"}
    ]
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user" and "image" in message:
            st.image(message["image"], caption="User uploaded image", use_column_width=True)
            if message.get("content"):
                st.markdown(message["content"])
        else:
            st.markdown(message["content"])


# Define the prompt template
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly and patient assistant for urban gardening enthusiasts.Your role is to provide encouraging advice on plant recommendations, composting guides,and plant care tips. If you don't know the answer, kindly say that you don't know."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Define the chain
chain = prompt | llm

# Maximum number of messages to keep in the history
MAX_HISTORY_LENGTH = 50

# Function to manage conversation history
def manage_history():
    """Trims the conversation history to the most recent messages."""
    if len(st.session_state.messages) > MAX_HISTORY_LENGTH:
        st.session_state.messages = st.session_state.messages[-MAX_HISTORY_LENGTH:]

# Update the session state messages after each interaction
def add_message(role, content=None, image=None):
    """Add a message to the session state and manage history."""
    message = {"role": role, "content": content}
    if image:
        message["image"] = image
    st.session_state.messages.append(message)
    manage_history()

# Update the functions with the new `add_message` utility
def get_text_response(query):
    """Process text-based user input and return the LLM response."""
    add_message("user", content=query)
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    add_message("assistant", content=answer)
    return answer

def get_image_response(image, question):
    """Process image-based user input and return the LLM response."""
    combined_input = f"Image uploaded with question: {question}" if question else "Image uploaded. Please identify the plant."
    add_message("user", content=question, image=image)
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    add_message("assistant", content=answer)
    return answer

# Incorporate the helper functions where required
user_prompt = st.chat_input("Type your question or upload an image below:")

if user_prompt:
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            answer = get_text_response(user_prompt)
            st.write(answer)

uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.chat_message("assistant"):
        with st.spinner("Identifying plant and providing an answer..."):
            answer = get_image_response(image, "")
            st.write(answer)
