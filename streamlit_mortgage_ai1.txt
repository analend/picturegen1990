
import streamlit as st
import openai
import faiss
import numpy as np
import tiktoken

# Set your OpenAI API key
openai.api_key = "API"  # Replace with your key

# Helper functions
def chunk_text(text, max_tokens=500):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [enc.decode(chunk) for chunk in chunks]

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]

# UI setup
st.set_page_config(page_title="Mortgage AI Assistant")
st.title("📄 Mortgage AI Assistant")
st.write("Upload a document and ask questions about it!")

# Memory for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload and embed
uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    chunks = chunk_text(text)
    vectors = [get_embedding(chunk) for chunk in chunks]

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    st.success("Document uploaded and processed!")

    question = st.text_input("Ask a question about your document:")

    if question:
        query_vector = np.array([get_embedding(question)]).astype("float32")
        _, result = index.search(query_vector, k=1)
        matched_chunk = chunks[result[0][0]]

        # Prepare prompt with memory
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.chat_history])
        prompt = f"{history}\n\nUse this to answer:\nContext: {matched_chunk}\nQuestion: {question}"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a helpful mortgage assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["choices"][0]["message"]["content"]
        st.session_state.chat_history.append((question, answer))
        st.markdown("**Answer:** " + answer)
import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text
uploaded_file = st.file_uploader("Upload a .pdf or .txt", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

