import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Function to read URL content (re-used from HW2)
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        # Don't clutter UI with errors if field is empty/intermediary, but here we expect valid input
        # st.error(f"Error reading {url}: {e}") 
        return None

# Title and Description
st.title("HW3: Chatbot with Context & Memory")
st.write("""
**How this chatbot works:**
This chatbot can answer questions based on the content of up to two provided URLs.
It uses a **Conversation Buffer** memory system, retaining the last 3 user-assistant exchanges (6 messages total) to maintain context during the conversation.
The content from the URLs is injected into the system prompt and is never discarded.
""")

# Sidebar Configuration
st.sidebar.header("Configuration")

# 1. URL Inputs
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/1")
url2 = st.sidebar.text_input("URL 2", placeholder="https://example.com/2")

# 2. LLM Vendor and Model Selection
llm_provider = st.sidebar.selectbox("Select LLM Vendor", ["OpenAI", "Google (Gemini)"])

model_options = {
    "OpenAI": ["gpt-4o"], 
    "Google (Gemini)": ["gemini-2.5-pro"] 
}

selected_model = st.sidebar.selectbox("Select Model", model_options[llm_provider])

# API Key Handling
api_key = None
if llm_provider == "OpenAI":
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("OpenAI API key not found. Please configure it in .streamlit/secrets.toml")
elif llm_provider == "Google (Gemini)":
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("Google API key not found. Please configure it in .streamlit/secrets.toml")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question about the URLs..."):
    
    if not api_key:
        st.stop() # Stop if no API key

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Conversation Memory (Buffer of 6 messages)
    # We take the last 6 messages from the history (3 exchanges)
    # Note: st.session_state.messages includes ALL history.
    recent_history = st.session_state.messages[-6:]

    # 4. System Prompt with Context
    context_text = ""
    if url1:
        content1 = read_url_content(url1)
        if content1:
            context_text += f"\n\nContext from URL 1 ({url1}):\n{content1[:10000]}..." # Truncate to be safe
    if url2:
        content2 = read_url_content(url2)
        if content2:
            context_text += f"\n\nContext from URL 2 ({url2}):\n{content2[:10000]}..."

    system_prompt = {
        "role": "system",
        "content": f"You are a helpful assistant. Answer questions based on the following context if provided. If the answer is not in the context, use your general knowledge but mention that it wasn't in the provided text.\n\n{context_text}"
    }

    # Construct messages to send: System Prompt + Recent History
    messages_to_send = [system_prompt] + recent_history

    # Generate Response
    with st.chat_message("assistant"):
        response_text = ""
        
        try:
            if llm_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                stream = client.chat.completions.create(
                    model=selected_model,
                    messages=messages_to_send,
                    stream=True,
                )
                response_text = st.write_stream(stream)

            elif llm_provider == "Google (Gemini)":
                genai.configure(api_key=api_key)
                # Gemini requires a slightly different format (string or specific object list). 
                # Converting standard messages format to Gemini format if needed, but the library often handles simple lists.
                # However, Gemini 'chats' are usually stateful. For stateless/one-off with context, `generate_content` is easier, 
                # but for chat history we often use `start_chat`.
                # Given the requirements, we act stateless by sending the full context every time (System + History).
                
                # Adapting messages for Gemini
                gemini_messages = []
                # System instructions are usually set in model config or prepended.
                # Here we will prepend the system instruction to the first message or send as fresh context.
                
                # Simple approach: Combine everything into one prompt for Gemini or use the chat structure?
                # The `messages` for Gemini are usually `[{'role': 'user', 'parts': [...]}, {'role': 'model', 'parts': [...]}]`
                # System prompt is passed to `GenerativeModel(..., system_instruction=...)`
                
                system_instruction = system_prompt["content"]
                model = genai.GenerativeModel(selected_model, system_instruction=system_instruction)
                
                # Convert recent_history to Gemini format
                chat_history_for_gemini = []
                for msg in recent_history:
                    role = "user" if msg["role"] == "user" else "model"
                    chat_history_for_gemini.append({"role": role, "parts": [msg["content"]]})

                # Since `chat_history_for_gemini` ends with the user's latest prompt, 
                # and `start_chat` usually takes history *before* the new message,
                # we split it.
                
                history_input = chat_history_for_gemini[:-1]
                last_user_message = chat_history_for_gemini[-1]["parts"][0]
                
                chat = model.start_chat(history=history_input)
                response_stream = chat.send_message(last_user_message, stream=True)
                
                # Helper to yield text from Gemini stream
                def stream_gemini(response):
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text

                response_text = st.write_stream(stream_gemini(response_stream))

        except Exception as e:
            st.error(f"Error generating response: {e}")
            response_text = "I encountered an error."

    # Add assistant response to history
    if response_text:
        st.session_state.messages.append({"role": "assistant", "content": response_text})
