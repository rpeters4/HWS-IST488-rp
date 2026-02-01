import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Read URL content
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# App title
st.title("HW2: URL Summarizer")
st.write("Summarize a standard web page URL using different LLMs.")

# Inputs
url = st.text_input("Enter a Web page URL", placeholder="https://example.com/article")

# Sidebar config
summary_type = st.sidebar.selectbox(
    "Type of Summary",
    ["Short Summary", "Detailed Summary", "Bullet Points", "ELI5 (Explain Like I'm 5)"]
)

# Output language
language = st.sidebar.selectbox(
    "Output Language",
    ["English", "French", "Spanish", "Chinese", "German"]
)

# Model selection
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["OpenAI", "Google (Gemini)"]
)

model_options = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Google (Gemini)": ["gemini-1.5-flash", "gemini-1.5-pro"]
}

selected_model = st.sidebar.selectbox(
    "Select Model",
    model_options.get(llm_provider, [])
)

use_advanced_model = st.sidebar.checkbox("Use Advanced Model (if applicable/mapped logic needed)")

# API Keys
api_key = None

if llm_provider == "OpenAI":
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.sidebar.error("OpenAI API Key not found. Add `OPENAI_API_KEY` to `.streamlit/secrets.toml`.")

elif llm_provider == "Google (Gemini)":
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.sidebar.warning("Google API Key not found. Please add `GOOGLE_API_KEY` to `.streamlit/secrets.toml`.")



# Summarize button logic
if st.button("Summarize"):
    if not url:
        st.error("Please enter a URL.")
    elif not api_key:
        st.error(f"Please configure the API key for {llm_provider} in .streamlit/secrets.toml")
    else:
        text_content = read_url_content(url)
        
        if text_content:
            st.info(f"Successfully read content. Length: {len(text_content)} characters. Generating summary...")
            
            # Build prompt
            prompt = f"""
            Please provide a {summary_type} of the following text.
            Output the summary in {language}.
            
            Text:
            {text_content[:20000]}  # Truncate to avoid context window issues just in case
            """
            
            # Generate summary
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
                    
                    # Stream logic for Gemini
                    def stream_gemini(response):
                        for chunk in response:
                            if chunk.text:
                                yield chunk.text
                                
                    st.write_stream(stream_gemini(response_stream))


                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
