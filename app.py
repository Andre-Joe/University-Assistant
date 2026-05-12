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

# ------------------ Retrieval (STRICT LIMIT) ------------------
def retrieve_chunks(query, data, top_k=2):
    query_emb = embed_model.encode([query])
    chunk_embs = np.array([d["embedding"] for d in data])

    sims = cosine_similarity(query_emb, chunk_embs)[0]

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

# ------------------ Generate response (HARD SAFE LIMITS) ------------------
def generate_response(query, chunks):

    # HARD LIMIT 1: max chunks
    chunks = chunks[:2]

    # HARD LIMIT 2: truncate each chunk aggressively
    safe_context_parts = []
    for c in chunks:
        text = c["text"]

        # remove excessive size early
        text = text[:800]

        safe_context_parts.append(text)

    context = "\n\n".join(safe_context_parts)

    # HARD LIMIT 3: total context cap
    context = context[:2000]

    prompt = f"""
You are a strict university assistant.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
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

# ------------------ Streamlit UI ------------------
st.title("🤖 University Courses Chatbot")

query = st.text_input("Ask a question about AI courses:")

if st.button("Get Answer") and query:

    with st.spinner("Retrieving relevant content..."):
        top_chunks = retrieve_chunks(query, data, top_k=2)

    # DEBUG (optional but useful)
    # for c in top_chunks:
    #     st.write(len(c["text"]))

    with st.spinner("Generating answer..."):
        answer = generate_response(query, top_chunks)
        cleaned = clean_answer(answer)

    st.markdown("### Answer")
    st.write(cleaned)

    st.markdown("### Sources")
    for c in top_chunks:
        st.write(f"- {c['file_name']} (page/slide: {c['page_or_slide']})")
