from sentence_transformers import SentenceTransformer
from utils import extract_resume_text
import pickle
import os
import json
from tqdm import tqdm

DATA_PATH = os.path.join("data", "cv_10k.json")
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")

def read_json_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                yield json.loads(line.strip())
            except json.JSONDecodeError:
                continue

def build_index_10k():
    print("Загрузка модели...")
    model = SentenceTransformer("cointegrated/rubert-tiny2")

    all_embeddings = []
    resume_id_map = []

    print("Чтение и обработка резюме...")
    texts = []
    records = []

    for resume in tqdm(read_json_lines(DATA_PATH), desc="Обработка резюме"):
        text = extract_resume_text(resume)
        if text.strip():
            texts.append(text)
            records.append(resume)

    print(f"Генерация эмбеддингов для {len(texts)} резюме...")
    embeddings = model.encode(texts, show_progress_bar=True)
    all_embeddings.extend(embeddings)
    resume_id_map.extend(records)

    print("Сохранение файлов...")
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(all_embeddings, f)

    with open(MAP_PATH, "wb") as f:
        pickle.dump(resume_id_map, f)

    print("Готово! Индекс создан.")

if __name__ == "__main__":
    build_index_10k()
