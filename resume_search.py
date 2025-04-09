import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import os

# Пути
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings.pkl")
MAP_PATH = os.path.join("models", "resume_id_map.pkl")

# Загрузка данных
print("Загрузка модели и данных...")
model = SentenceTransformer("cointegrated/rubert-tiny", device="cuda")

with open(EMBEDDINGS_PATH, "rb") as f:
    all_embeddings = pickle.load(f)

with open(MAP_PATH, "rb") as f:
    resume_id_map = pickle.load(f)

# Инициализация поиска
print("Создание индекса...")
nn = NearestNeighbors(n_neighbors=10, metric='cosine')
if not all_embeddings:
    print("Массив эмбеддингов пуст. Проверьте обработку данных.")
    exit(1)
import numpy as np

all_embeddings = np.array(all_embeddings)
if len(all_embeddings.shape) == 1:
    all_embeddings = all_embeddings.reshape(-1, 1)
nn.fit(all_embeddings)

# Поиск
while True:
    query = input("\nВведите описание кандидата (или 'exit'): ")
    if query.lower() == 'exit':
        break

    query_vec = model.encode([query])
    distances, indices = nn.kneighbors([query_vec], n_neighbors=10)

    print("\nТоп-10 подходящих резюме:")
    for i, idx in enumerate(indices[0]):
        print(f"\n--- Кандидат {i+1} ---")
        print(resume_id_map[idx])