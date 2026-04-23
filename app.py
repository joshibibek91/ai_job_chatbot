import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
genai.configure(api_key="AIzaSyCbr5SOHnI_1ImJon7MGRMvQGn7PV5UStI")

# Load models (cached to avoid reload)
# @st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = genai.GenerativeModel("gemini-2.5-flash-lite")
    return embed_model, llm

# Load documents and FAISS index
# @st.cache_resource
def load_index(_embed_model):
    with open("job.txt", "r", encoding="utf-8") as f:
        documents = f.read().split("\n")

    embeddings = _embed_model.encode(documents)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return documents, index

# -----------------------------
# UI
# -----------------------------
st.title("Job Fit Analyzer")
st.write("Enter your query. System retrieves relevant job info and evaluates fit.")

query = st.text_input("Enter your query")

# -----------------------------
# MAIN LOGIC
# -----------------------------
if st.button("Analyze") and query:

    embed_model, llm = load_models()
    documents, index = load_index(embed_model)

    # Encode query
    query_vec = embed_model.encode([query])

    # Search
    D, I = index.search(np.array(query_vec), k=2)

    retrieved_text = documents[I[0][0]]

    # Prompt
    prompt = f"""First read the doument or text "{documents[I[0][0]]}" then search the user query "{query}". """

    response = llm.generate_content(prompt)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Top Retrieved Job Text")
    st.write(retrieved_text)

    st.subheader("AI Evaluation")
    st.write(response.text)