import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("AssistantData.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["soru"] + " " + item["cevap"] for item in data]
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "kamu.index")

with open("kamu_metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Embedding işlemi tamamlandı.")
