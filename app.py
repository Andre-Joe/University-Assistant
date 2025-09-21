import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceApi, hf_api

# ------------------ Load course chunks (text only) ------------------
@st.cache_data
def load_chunks(pkl_file="course_chunks.pkl"):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    return data

chunks = load_chunks("course_chunks.pkl")

# ------------------ Hugging Face API ------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
inference = InferenceApi(repo_id="google/flan-t5-small", token=HF_TOKEN)

def get_embedding(text):
    # Use Hugging Face embedding model
    response = hf_api().embed(
        model="sentence-transformers/all-MiniLM-L6-v2",
        input=text,
        token=HF_TOKEN
    )
    return np.array(response)

# ------------------ Retrieve top chunks ------------------
def retrieve_chunks(query, chunks, top_k=5):
    query_emb = get_embedding(query)
    chunk_embs = np.array([get_embedding(c["text"]) for c in chunks])
    sims = cosine_similarity([query_emb], chunk_embs)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ------------------ Generate response ------------------
def generate_response(query, top_chunks):
    context = "\n".join([c["text"] for c in top_chunks])
    prompt = f"Answer the question based only on the following course content:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = inference(prompt)
    if isinstance(response, dict) and "generated_text" in response:
        return response["generated_text"]
    return str(response)

# ------------------ Streamlit UI ------------------
st.title("University Courses Chatbot")
query = st.text_input("Ask a question about your courses:")

if st.button("Get Answer") and query:
    with st.spinner("Searching for relevant content..."):
        top_chunks = retrieve_chunks(query, chunks, top_k=5)
    with st.spinner("Generating answer..."):
        answer = generate_response(query, top_chunks)
    st.markdown("**Answer:**")
    st.write(answer)
