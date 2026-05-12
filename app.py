import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re

# ------------------ Load embeddings ------------------
@st.cache_data
def load_embeddings(pkl_file="course_embeddings.pkl"):
    with open(pkl_file, "rb") as f:
        return pickle.load(f)

data = load_embeddings("course_embeddings.pkl")

# ------------------ Embedding model ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ------------------ Retrieval ------------------
def retrieve_chunks(query, data, top_k=3, full_course_weight=1.5):
    query_emb = embed_model.encode([query])
    chunk_embs = np.array([d["embedding"] for d in data])

    sims = cosine_similarity(query_emb, chunk_embs)[0]

    # boost full summary if exists
    for i, d in enumerate(data):
        if d.get("file_name") == "full_course_summary":
            sims[i] *= full_course_weight

    top_indices = sims.argsort()[-top_k:][::-1]
    return [data[i] for i in top_indices]

# ------------------ Generator ------------------
@st.cache_resource
def load_generator():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

generator = load_generator()

# ------------------ Safe generation ------------------
def generate_response(query, chunks):

    # HARD LIMIT: prevent token overflow
    chunks = chunks[:2]

    context = "\n\n".join([c["text"] for c in chunks])
    context = context[:2000]  # prevents 13k token crash

    prompt = f"""
You are a strict university assistant.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer briefly and clearly:
"""

    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    return result[0]["generated_text"]

# ------------------ Cleaning ------------------
def clean_answer(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if "." in text:
        text = text.rsplit(".", 1)[0] + "."

    return text

# ------------------ UI ------------------
st.title("🤖 University Courses Chatbot")

query = st.text_input("Ask a question about AI courses:")

if st.button("Get Answer") and query:

    with st.spinner("Retrieving relevant content..."):
        top_chunks = retrieve_chunks(query, data, top_k=3)

    with st.spinner("Generating answer..."):
        answer = generate_response(query, top_chunks)
        cleaned = clean_answer(answer)

    st.markdown("### Answer")
    st.write(cleaned)

    st.markdown("### Sources")
    for c in top_chunks:
        st.write(f"- {c['file_name']} (page/slide: {c['page_or_slide']})")
