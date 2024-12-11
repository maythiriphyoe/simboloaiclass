import streamlit as st
import os
from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the API key from the environment variable
#Save groq api key in a separate .env file 
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is not set. Please check your .env file or environment variables.")
    st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize button clicked state
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = None

# App title and greeting
st.title("I am Your GrowBuddy \U0001F331")
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(
            """Hey Buddy! Welcome to our urban gardening community. I'm here to help you with any questions or concerns 
            you may have about gardening. What brings you to our little corner of the world? Are you a seasoned gardener 
            or just starting out?"""
        )
        st.write("You can ask the following things or freely type your question:")

# Sidebar with gardening topics
with st.sidebar:
    st.title("Gardening Help")
    if st.button("Composting Tips"):
        st.session_state.button_clicked = "composting"
    if st.button("Plant Recommendation"):
        st.session_state.button_clicked = "recommendation"
    if st.button("Plant Care Tips"):
        st.session_state.button_clicked = "care"

# Initialize the ChatGroq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly assistant for urban gardening enthusiasts. \
        Your role is to provide encouraging advice on plant recommendations, composting guides, \
        and plant care tips. If you don't know the answer, kindly say that you don't know."),
        *[(msg["role"], msg["content"]) for msg in st.session_state.messages]
    ]
)

# Define the chain
chain = prompt | llm

# Define helper functions
MAX_MESSAGES = 10  # Keep only the last 10 messages for LLM input

def trim_session_messages():
    """Limit the number of messages in the session to the rolling window size."""
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

def get_text_response(query):
    """Process text-based user input and return the LLM response."""
    st.session_state.messages.append({"role": "user", "content": query})
    trim_session_messages()
    response = chain.invoke({"messages": st.session_state.messages})
    answer = response.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer

# Process button clicks
if st.session_state.button_clicked == "composting":
    with st.chat_message("assistant"):
        st.write("Great! Composting is an excellent way to recycle organic waste. What kind of materials are you composting?")
elif st.session_state.button_clicked == "recommendation":
    with st.chat_message("assistant"):
        st.write("Sure! I can recommend plants for your home. Could you tell me more about your space—do you get a lot of sunlight?")
elif st.session_state.button_clicked == "care":
    with st.chat_message("assistant"):
        st.write("Plant care tips are on the way! Could you tell me which plant you’re looking to take care of?")

# Reset the button clicked state
if st.session_state.button_clicked:
    st.session_state.button_clicked = None

# Handle chat input
user_prompt = st.chat_input("Type your question below:")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            try:
                answer = get_text_response(user_prompt)
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Manage file uploads
uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.session_state.messages.append({"role": "user", "image": uploaded_image})
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.chat_message("assistant"):
        with st.spinner("Identifying plant and providing an answer..."):
            try:
                answer = get_text_response("Image uploaded. Please identify the plant.")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

#run in the terminal -- streamlit run filename.py
