import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Function to read URL content
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# Show title and description
st.title("HW2: URL Summarizer")
st.write("Summarize a standard web page URL using different LLMs.")

# URL Input (Top of screen)
url = st.text_input("Enter a Web page URL", placeholder="https://example.com/article")

# Sidebar inputs
st.sidebar.header("Configuration")

# 1. Type of Summary
summary_type = st.sidebar.selectbox(
    "Type of Summary",
    ["Short Summary", "Detailed Summary", "Bullet Points", "ELI5 (Explain Like I'm 5)"]
)

# 2. Output Language
language = st.sidebar.selectbox(
    "Output Language",
    ["English", "French", "Spanish", "Chinese", "German"]
)

# 3. LLM Selection
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["OpenAI", "Google (Gemini)", "Anthropic (Claude)"]
)

model_options = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Google (Gemini)": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "Anthropic (Claude)": ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"]
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    model_options.get(llm_provider, [])
)

use_advanced_model = st.sidebar.checkbox("Use Advanced Model (if applicable/mapped logic needed)")

# API Key Handling
api_key = None

if llm_provider == "OpenAI":
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.sidebar.error("OpenAI API Key not found. Add `OPENAI_API_KEY` to `.streamlit/secrets.toml`.")

elif llm_provider == "Google (Gemini)":
    try:
        # User requested to know where to put the key:
        # Put "GOOGLE_API_KEY" in your .streamlit/secrets.toml file!
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.sidebar.warning("Google API Key not found. Please add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.")
        st.sidebar.info("You can get a key from: https://aistudio.google.com/app/apikey")

elif llm_provider == "Anthropic (Claude)":
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.sidebar.warning("Anthropic API Key not found. Please add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml`.")

# Main Action
if st.button("Summarize"):
    if not url:
        st.error("Please enter a URL.")
    elif not api_key:
        st.error(f"Please configure the API key for {llm_provider} in .streamlit/secrets.toml")
    else:
        text_content = read_url_content(url)
        
        if text_content:
            st.info(f"Successfully read content. Length: {len(text_content)} characters. Generating summary...")
            
            # Construct Prompt
            prompt = f"""
            Please provide a {summary_type} of the following text.
            Output the summary in {language}.
            
            Text:
            {text_content[:20000]}  # Truncate to avoid context window issues just in case
            """
            
            # Call LLM
            try:
                if llm_provider == "OpenAI":
                    client = OpenAI(api_key=api_key)
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True
                    )
                    st.write_stream(stream)
                    
                elif llm_provider == "Google (Gemini)":
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(selected_model)
                    response_stream = model.generate_content(prompt, stream=True)
                    
                    # Gemini streaming is a bit different, iterate and yield text
                    def stream_gemini(response):
                        for chunk in response:
                            if chunk.text:
                                yield chunk.text
                                
                    st.write_stream(stream_gemini(response_stream))

                elif llm_provider == "Anthropic (Claude)":
                    # Placeholder for valid library usage if user adds 'anthropic' to requirements
                    # For now just showing error or mocked if library not installed, 
                    # but I will assume it's not in requirements unless requested.
                    # I will modify requirements to include google-generativeai, but maybe not anthropic yet.
                    st.error("Claude implementation requires `anthropic` library. Please add it to requirements.txt.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
