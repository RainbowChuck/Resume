import pickle

with open("models/resume_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Количество эмбеддингов: {len(data)}")
print(f"Пример (если есть): {data[0] if data else 'пусто'}")