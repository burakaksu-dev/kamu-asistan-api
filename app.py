import json
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("kamu.index")

with open("kamu_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

app = FastAPI()

class Soru(BaseModel):
    soru: str

@app.post("/ask")
def ask(soru: Soru):
    vector = model.encode([soru.soru])[0].astype("float32")
    D, I = index.search(np.array([vector]), k=1)
    sonuc = metadata[I[0][0]]

    return {
        "soru": sonuc["soru"],
        "cevap": sonuc["cevap"],
        "kategori": sonuc["kategori"]
    }
