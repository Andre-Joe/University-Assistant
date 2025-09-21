import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceApi
import os

# ------------------ Load embeddings ------------------
@st.cache_data
def load_embeddings(pkl_file="course_embeddings.pkl"):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

data = load_embeddings("course_embeddings.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Retrieve top chunks ------------------
def retrieve_chunks(query, data, top_k=5):
    query_emb = model.encode([query])
    chunk_embs = np.array([d["embedding"] for d in data])
    sims = cosine_similarity(query_emb, chunk_embs)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [data[i] for i in top_indices]

# ------------------ Generate response ------------------
HF_TOKEN = st.secrets["HF_TOKEN"]  # Hugging Face token from Streamlit secrets
inference = InferenceApi(
    repo_id="google/flan-t5-small",  # free model for online inference
    token=HF_TOKEN
)

def generate_response(query, chunks):
    context = "\n".join([c["text"] for c in chunks])
    prompt = (
        f"Answer the question based only on the following course content:\n\n"
        f"{context}\n\nQuestion: {query}\nAnswer:"
    )
    result = inference(prompt)
    # Hugging Face returns dict with 'generated_text'
    if isinstance(result, dict) and "generated_text" in result:
        return result["generated_text"]
    return str(result)

# ------------------ Streamlit UI ------------------
st.title("University Courses Chatbot")
query = st.text_input("Ask a question about your courses:")

if st.button("Get Answer") and query:
    with st.spinner("Searching for relevant content..."):
        top_chunks = retrieve_chunks(query, data, top_k=5)
    with st.spinner("Generating answer..."):
        answer = generate_response(query, top_chunks)
    st.markdown("**Answer:**")
    st.write(answer)
