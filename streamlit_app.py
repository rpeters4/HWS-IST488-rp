
import streamlit as st
from openai import OpenAI
import fitz


#helper function for reading these PDFs
def read_pdf(pdf_file):
    pdf_contents = pdf_file.read()
    document = fitz.open(stream=pdf_contents, filetype="pdf")
    text = "" #empty buffer
    for page in document:
        text += page.get_text()
    document.close()
    return text
#---------------------------------------

# Show title and description.
st.title("MY (Ryan) Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

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

        #change below to allow text and pdf, pdf doesn't appear to be handled by this. 
        if uploaded_file and question:

            # Process the uploaded file and question.
            document = uploaded_file.read().decode()
            messages = [
                {
                    "role": "user",
                    "content": f"Here's a document: {document} \n\n---\n\n {question}",
                }
            ]

            # Generate an answer using the OpenAI API.
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo", ##might need to be changed with new formatting for adding a model selector. 
                messages=messages,
                stream=True,
            )

            # Stream the response to the app using `st.write_stream`.
            st.write_stream(stream)
    except Exception as e:
        st.error("Invalid API Key, please try again.") 
