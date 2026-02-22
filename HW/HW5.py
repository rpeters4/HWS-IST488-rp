import streamlit as st
from openai import OpenAI
import os
import json
import chromadb
from bs4 import BeautifulSoup

# ---------- Page Setup ----------
st.title("HW 5: Enhanced Student Org Chatbot")
st.write("Ask questions about Syracuse University student organizations.")

# ---------- API Key ----------
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("OpenAI API key not found. Please configure it in .streamlit/secrets.toml", icon="🗝️")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ---------- Vector DB helpers (reused from HW4) ----------

def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks


def extract_text_from_html(file_path):
    """Extracts visible text from an HTML file using BeautifulSoup."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return ""


def create_vector_db():
    """Creates and returns a ChromaDB collection with the HTML data."""
    html_dir = "su_orgs"

    if not os.path.exists(html_dir):
        st.error(f"Directory {html_dir} not found.")
        return None

    html_files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    if not html_files:
        st.error(f"No HTML files found in {html_dir}.")
        return None

    # Persistent ChromaDB — same path as HW4 so the DB is shared
    db_path = "./chroma_db_hw4"
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection_name = "HW4Collection"

    # If collection already exists and has data, reuse it
    try:
        collection = chroma_client.get_collection(name=collection_name)
        if collection.count() > 0:
            return collection
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=collection_name)

    documents = []
    ids = []
    metadatas = []

    progress_bar = st.progress(0, text="Processing HTML files...")
    for idx, filename in enumerate(html_files):
        file_path = os.path.join(html_dir, filename)
        full_text = extract_text_from_html(file_path)
        if full_text:
            chunks = chunk_text(full_text, chunk_size=2000, overlap=200)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                ids.append(f"{filename}_chunk_{i}")
                metadatas.append({"filename": filename, "chunk_id": i})
        if idx % 10 == 0:
            progress_bar.progress((idx + 1) / len(html_files),
                                  text=f"Processed {idx + 1}/{len(html_files)} files")
    progress_bar.empty()

    # Generate embeddings in batches
    embeddings = []
    batch_size = 100
    embedding_progress = st.progress(0, text="Generating embeddings...")
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        try:
            response = client.embeddings.create(input=batch_docs, model="text-embedding-3-small")
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            embedding_progress.progress((i + len(batch_docs)) / len(documents),
                                        text=f"Generated {i + len(batch_docs)}/{len(documents)} embeddings")
        except Exception as e:
            st.error(f"Error generating embeddings for batch {i}: {e}")
            return None
    embedding_progress.empty()

    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection


# ---------- Initialize Vector DB in session state ----------
if "HW5_VectorDB" not in st.session_state:
    with st.spinner("Initializing Vector Database... (first load may take a while)"):
        st.session_state.HW5_VectorDB = create_vector_db()

# ---------- The key function: relevant_club_info ----------

def relevant_club_info(query: str) -> str:
    """
    Takes a query string, performs a vector search against the ChromaDB
    collection of student organizations, and returns the top results as
    a formatted string that the LLM can use to answer the user's question.
    """
    collection = st.session_state.get("HW5_VectorDB")
    if not collection:
        return json.dumps({"error": "Vector database not available."})

    try:
        # Generate embedding for the query
        query_response = client.embeddings.create(input=query, model="text-embedding-3-small")
        query_embedding = query_response.data[0].embedding

        # Search ChromaDB
        results = collection.query(query_embeddings=[query_embedding], n_results=5)

        if not results['documents'] or not results['documents'][0]:
            return json.dumps({"results": "No relevant student organizations found."})

        # Format results
        formatted = []
        retrieved_docs = []
        for i, doc in enumerate(results['documents'][0]):
            filename = results['metadatas'][0][i]['filename']
            retrieved_docs.append(filename)
            formatted.append(f"--- Source: {filename} ---\n{doc}")

        # Store debug info in session state
        st.session_state.last_retrieved_docs = retrieved_docs
        st.session_state.last_context_length = sum(len(d) for d in results['documents'][0])

        return json.dumps({"results": "\n\n".join(formatted)})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------- OpenAI Tool Definition ----------
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the Syracuse University student organizations database "
                "for information relevant to the user's query. Call this whenever "
                "the user asks about clubs, organizations, student groups, or "
                "related topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant student organizations.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

# ---------- Short-term Memory ----------
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = []

# Keep last 5 interaction pairs (10 messages)
MAX_HISTORY = 10
if len(st.session_state.hw5_messages) > MAX_HISTORY:
    st.session_state.hw5_messages = st.session_state.hw5_messages[-MAX_HISTORY:]

# Display chat history
for message in st.session_state.hw5_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Chat Input & Function-Calling Loop ----------
if prompt := st.chat_input("Ask about Student Organizations at Syracuse University"):
    # Add & display user message
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages list for the LLM
    system_prompt = (
        "You are a helpful assistant for Syracuse University students. "
        "You answer questions about student organizations. "
        "When the user asks about clubs, organizations, or student groups, "
        "use the relevant_club_info tool to search the database first. "
        "Base your answers on the information returned by the tool. "
        "If the tool returns no results, let the user know that you couldn't "
        "find matching organizations in the database."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(st.session_state.hw5_messages)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # --- First LLM call: let the model decide to call the tool ---
                first_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )

                response_message = first_response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:
                    # The model wants to call relevant_club_info
                    messages.append(response_message)

                    for tool_call in tool_calls:
                        fn_args = json.loads(tool_call.function.arguments)
                        query_arg = fn_args.get("query", prompt)

                        # Execute the function
                        tool_result = relevant_club_info(query=query_arg)

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "relevant_club_info",
                            "content": tool_result,
                        })

                    # --- Second LLM call: generate the final answer ---
                    second_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        stream=True,
                    )
                    response = st.write_stream(second_response)
                else:
                    # Model answered without calling the tool
                    response = response_message.content
                    st.markdown(response)

            except Exception as e:
                response = f"An error occurred: {e}"
                st.error(response)

    st.session_state.hw5_messages.append({"role": "assistant", "content": response})

# ---------- Debug Sidebar ----------
with st.sidebar:
    st.header("Debug Tool")

    if "last_context_length" in st.session_state:
        st.write(f"**Context Length:** {st.session_state.last_context_length} chars")

    if "last_retrieved_docs" in st.session_state:
        st.write("**Retrieved Documents:**")
        seen = set()
        for doc in st.session_state.last_retrieved_docs:
            if doc not in seen:
                seen.add(doc)
                st.write(f"- {doc}")

    if st.button("Clear Conversation"):
        st.session_state.hw5_messages = []
        st.rerun()
