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
        data = pickle.load(f)
    return data

data = load_embeddings("course_embeddings.pkl")

# ------------------ Embedding model (online) ------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# ------------------ Retrieve top chunks ------------------
def retrieve_chunks(query, data, top_k=5, full_course_weight=1.5):
    query_emb = embed_model.encode([query])
    chunk_embs = np.array([d["embedding"] for d in data])
    sims = cosine_similarity(query_emb, chunk_embs)[0]
    
    # Boost full course summaries
    for i, d in enumerate(data):
        if d.get("file_name") == "full_course_summary":
            sims[i] *= full_course_weight
    
    top_indices = sims.argsort()[-top_k:][::-1]
    return [data[i] for i in top_indices]

# ------------------ Hugging Face text generation pipeline ------------------
@st.cache_resource
def load_generator():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
    )
    return generator

generator = load_generator()

# ------------------ Generate answer ------------------
def generate_response(query, chunks):
    answers = []
    for c in chunks:
        prompt = f"Answer the question based only on the following course content:\n\n{c['text']}\n\nQuestion: {query}\nAnswer:"
        result = generator(prompt, max_new_tokens=256, repetition_penalty=2.0)  # repetition penalty added
        if isinstance(result, list) and "generated_text" in result[0]:
            answers.append(result[0]["generated_text"])
        else:
            answers.append(str(result))
    
    # Combine answers and remove duplicates
    combined_text = " ".join(answers)
    seen = set()
    unique_sentences = []
    for s in combined_text.split('. '):
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            unique_sentences.append(s_clean)
            seen.add(s_clean)
    final_answer = '. '.join(unique_sentences)
    
    # Ensure final full stop
    if not final_answer.endswith('.'):
        final_answer += '.'
        
    return final_answer

# ------------------ Clean generated answer ------------------
def remove_links(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_names_emails(text):
    # rough removal: names in format First Last or emails
    text = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    return text

def truncate_after_last_fullstop(text):
    if '.' in text:
        return text.rsplit('.', 1)[0] + '.'
    return text

def clean_answer(text):
    text = remove_links(text)
    text = remove_names_emails(text)
    text = truncate_after_last_fullstop(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------ Streamlit UI ------------------
st.title("🤖 University Courses Chatbot")

query = st.text_input("Ask a question about AI courses:")

if st.button("Get Answer") and query:
    with st.spinner("Searching for relevant content..."):
        top_chunks = retrieve_chunks(query, data, top_k=5)
    with st.spinner("Generating answer..."):
        answer = generate_response(query, top_chunks)
        cleaned_answer = clean_answer(answer)
    
    st.markdown("**Answer:**")
    st.write(cleaned_answer)

    # Optional: show source of each chunk
    st.markdown("**Sources of information used:**")
    for i, c in enumerate(top_chunks):
        st.write(f"- {c['file_name']} (page/slide: {c['page_or_slide']})")
