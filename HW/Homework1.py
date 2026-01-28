import streamlit as st
from openai import OpenAI
import pymupdf as fitz


# Helper function for reading PDFs
def read_pdf(pdf_file):
    pdf_contents = pdf_file.read()
    document = fitz.open(stream=pdf_contents, filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    document.close()
    return text


# Show title and description.
st.title("MY (Ryan) Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.secrets`.
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please configure it in .streamlit/secrets.toml", icon="🗝️")
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

try:
    client.models.list()
    st.success("API Key valid, proceed...")

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Check file extension and process accordingly
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            document = None

        if document:
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
               # model="gpt-3.5-turbo", #huge paragraph, content sounded good
               # model="gpt-4o-mini", #good amount of content, but parsed it with numbers for good reading spacing
               # model="gpt-5-chat-latest", # more formatting on the output, made it more human readable for myself and fast response time
                model="gpt-5-nano", #similar to 5 chat latest, but doesn't use emojis and just uses periods or dots for highlighting stop points
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)

except Exception as e:
    st.error("Invalid API Key, please try again.")
