import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceApi

# ------------------ Load embeddings ------------------
@st.cache_data
def load_embeddings(pkl_file="course_embeddings.pkl"):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

data = load_embeddings("course_embeddings.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Helper: retrieve top chunks ------------------
def retrieve_chunks(query, data, top_k=5):
    query_emb = model.encode([query])
    chunk_embs = np.array([d["embedding"] for d in data])
    sims = cosine_similarity(query_emb, chunk_embs)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [data[i] for i in top_indices]

# ------------------ Helper: generate response ------------------
HF_TOKEN = st.secrets["HF_TOKEN"]  # use secret instead of hardcoding
inference = InferenceApi(repo_id="google/flan-t5-small", token=HF_TOKEN)  # small free model

def generate_response(query, chunks):
    context = "\n".join([c["text"] for c in chunks])
    prompt = f"Answer the question based only on the following course content:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = inference(prompt)
    return response

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
