import os
import pickle
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from utils import extract_resume_text

EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")
MODEL_NAME = "cointegrated/rubert-tiny2"

def search_resumes(query_text, model, resumes, embeddings, top_k=10):
    query = {
        "positionName": query_text,
        "experience": 3,
        "hardSkills": query_text.lower().split(),
        "salary": 100000
    }

    text = query["positionName"] + " " + " ".join(query["hardSkills"])
    query_vec = model.encode(text)

    nn = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors([query_vec], n_neighbors=50)

    ranked = []
    keywords = set(query["hardSkills"])
    strong_keywords = {"история", "учитель", "обществознание"}

    for idx in indices[0]:
        resume = resumes[idx]
        score = 0
        pos_tokens = set(resume.get("positionName", "").lower().split())
        pos_match = len(keywords & pos_tokens)
        strong_match = len(strong_keywords & pos_tokens)
        skills = set(resume.get("hardSkills", []))
        skill_match = len(keywords & skills)
        experience_score = max(0, 10 - abs(query["experience"] - resume.get("experience", 0)))
        salary_score = max(0, 10 - abs(query["salary"] - resume.get("salary", 0)) // 10000)

        score = pos_match * 5 + skill_match * 3 + strong_match * 7 + experience_score + salary_score
        ranked.append((score, resume))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]

def main():
    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(MAP_PATH):
        print("Файлы индекса не найдены. Сначала запусти build_index.py")
        return

    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings = pickle.load(f)
    with open(MAP_PATH, "rb") as f:
        resumes = pickle.load(f)

    model = SentenceTransformer(MODEL_NAME)

    while True:
        query_text = input("\nВведите описание кандидата (или 'exit'): ")
        if query_text.lower() == "exit":
            break

        top10 = search_resumes(query_text, model, resumes, embeddings)

        print("\nТоп-10 подходящих резюме:")
        for i, res in enumerate(top10):
            print(f"\n--- Кандидат {i+1} ---")
            print(f"Должность: {res.get('positionName', '-')}")
            print(f"Опыт: {res.get('experience', '-')}")
            print(f"Город: {res.get('localityName', '-')}")
            print(f"Образование: {res.get('education', '-')}")
            print(f"Зарплата: {res.get('salary', '-')}")
            print(f"Навыки: {', '.join(res.get('hardSkills', []))}")

if __name__ == "__main__":
    main()
