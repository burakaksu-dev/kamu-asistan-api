from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()

# CORS ayarları (mobil uygulama erişebilsin diye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SoruModel(BaseModel):
    soru: str

# Embedder modeli yükle
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FAISS index ve metadata dosyaları
index = faiss.read_index("kamu.index")

with open("kamu_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

@app.post("/ask")
async def ask_question(payload: SoruModel):
    soru_embedding = model.encode([payload.soru])
    D, I = index.search(np.array(soru_embedding).astype("float32"), k=1)
    idx = int(I[0][0])
    return metadata[idx]
