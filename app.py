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

    # keep 3 chunks (not 2)
    chunks = chunks[:3]

    processed_chunks = []

    for c in chunks:
        text = c["text"]

        # keep more signal, less aggression
        text = text[:1200]

        processed_chunks.append(text)

    context = "\n\n".join(processed_chunks)

    # allow slightly larger context but still safe
    context = context[:3500]

    prompt = f"""
You are a university professor.

If the context is not enough, explain using general knowledge from the course topic.

Always give a clear explanation.

Structure:
- Definition
- Explanation
- Example

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=220,
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
