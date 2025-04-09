import os
import json
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import extract_resume_text

SOURCE_PATH = os.path.join("data", "cv_10k.json")
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")
MODEL_NAME = "cointegrated/rubert-tiny2"

os.makedirs("models", exist_ok=True)

def build_index():
    print("[Построение индекса] Загрузка модели...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = []
    resume_map = []

    with open(SOURCE_PATH, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Обработка резюме"):
            resume = json.loads(line)
            text = extract_resume_text(resume)
            if text:
                emb = model.encode(text)
                embeddings.append(emb)
                resume_map.append(resume)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings, f)
    with open(MAP_PATH, "wb") as f:
        pickle.dump(resume_map, f)

    print(f"Индекс построен: {len(embeddings)} резюме")

if __name__ == "__main__":
    build_index()
