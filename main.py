from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------
# CONFIG
# -----------------------------
genai.configure(api_key="AIzaSyCUTQbJu8e_a4YYviLMDYb5kxBZSAvTQ24")

# load LLM
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# FastAPI app
app = FastAPI(title="Job Fit Analyzer API")

# -----------------------------
# LOAD MODELS + INDEX (ON START)
# -----------------------------
model_embed = SentenceTransformer("all-MiniLM-L6-v2")

with open("job.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n")

embeddings = model_embed.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class QueryRequest(BaseModel):
    query: str


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "Job Fit Analyzer API is running"}


@app.post("/analyze")
def analyze(req: QueryRequest):
    query = req.query

    # encode query
    query_vec = model_embed.encode([query])

    # search FAISS
    D, I = index.search(np.array(query_vec), k=2)

    retrieved_text = documents[I[0][0]]

    # prompt
    prompt = f"""
    First read the document: "{retrieved_text}"
    Then answer the user query: "{query}"
    """

    response = model.generate_content(prompt)

    return {
        "query": query,
        "retrieved_text": retrieved_text,
        "ai_evaluation": response.text
    }