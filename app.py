import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
genai.configure(api_key="AIzaSyCUTQbJu8e_a4YYviLMDYb5kxBZSAvTQ24")

# load model
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# embedding model
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents and FAISS index
file = open("job.txt", "r", encoding="utf-8")
documents = file.read().split("\n")
file.close()


# convert text → vectors
embeddings = model_embed.encode(documents)


# -----------------------------
# UI
# -----------------------------
st.title("Job Fit Analyzer")
st.write("Enter your query. System retrieves relevant job info and evaluates fit.")

query = st.text_input("Enter your query")


# create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# -----------------------------
# MAIN LOGIC
# -----------------------------
if st.button("Analyze") and query:

    # Encode query
    query_vec = model_embed.encode([query])

    # Search
    D, I = index.search(np.array(query_vec), k=2)

    retrieved_text = documents[I[0][0]]

    # Prompt
    prompt = f"""First read the doument or text "{documents[I[0][0]]}" then search the user query "{query}". """

    response = model.generate_content(prompt)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("Top Retrieved Job Text")
    st.write(retrieved_text)

    st.subheader("AI Evaluation")
    st.write(response.text)