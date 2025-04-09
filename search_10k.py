import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# Пути
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")

# Загрузка
print("Загрузка модели и индекса...")
model = SentenceTransformer("cointegrated/rubert-tiny2")

with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)

with open(MAP_PATH, "rb") as f:
    resumes = pickle.load(f)

# Индексация
nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(embeddings)

# Запуск поиска
while True:
    query = input("\nВведите описание кандидата (или 'exit'): ")
    if query.lower() == 'exit':
        break

    query_vec = model.encode([query])
    distances, indices = nn.kneighbors([query_vec], n_neighbors=10)

    print("\nТоп-10 подходящих резюме:")
    for i, idx in enumerate(indices[0]):
        print(f"\n--- Кандидат {i+1} ---")
        print(f"Должность: {resumes[idx].get('positionName', '-')}")
        print(f"Опыт: {resumes[idx].get('experience', '-')}")
        print(f"Город: {resumes[idx].get('localityName', '-')}")
        print(f"Образование: {resumes[idx].get('education', '-')}")
        print(f"Желаемая зарплата: {resumes[idx].get('salary', '-')}")
