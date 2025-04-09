import os
import json
import pickle
import ijson
import re
import pymorphy2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# === Пути ===
SOURCE_PATH = os.path.join("data", "cv.json")
EXTRACTED_PATH = os.path.join("data", "cv_10k.json")
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")
MODEL_NAME = "cointegrated/rubert-tiny2"
LIMIT = 10000

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)



def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^а-яА-Яa-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Извлечение текста из резюме ===
def extract_resume_text(resume):
    parts = [
        resume.get("positionName", ""),
        resume.get("education", ""),
        str(resume.get("experience", "")),
        resume.get("scheduleType", ""),
        str(resume.get("salary", "")),
        resume.get("retrainingCapability", ""),
        resume.get("relocation", ""),
        resume.get("businessTrip", ""),
        resume.get("gender", ""),
        resume.get("localityName", "")
    ]

    for edu in resume.get("educationList", []):
        parts.append(edu.get("instituteName", ""))
        parts.append(str(edu.get("graduateYear", "")))

    for prof in resume.get("professionList", []):
        parts.append(prof.get("codeProfessionalSphere", ""))

    for lang in resume.get("languageKnowledge", []):
        parts.append(lang.get("codeLanguage", ""))
        parts.append(lang.get("level", ""))

    for country in resume.get("country", []):
        parts.append(country.get("countryName", ""))

    raw = " ".join(str(p).strip() for p in parts if p)
    return clean_text(raw)

# === 1. Извлечение 10 000 резюме ===
def extract_10k():
    print("[1] Извлечение 10 000 резюме из большого файла...")
    count = 0
    with open(SOURCE_PATH, 'r', encoding='utf-8') as f, \
         open(EXTRACTED_PATH, 'w', encoding='utf-8') as out:
        for item in ijson.items(f, "cvs.item"):
            if count >= LIMIT:
                break
            json.dump(item, out, ensure_ascii=False)
            out.write('\n')
            count += 1
    print(f"    → Сохранено {count} резюме в {EXTRACTED_PATH}")

# === 2. Построение эмбеддингов ===
def build_index():
    print("[2] Построение эмбеддингов и индекса...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = []
    resume_map = []

    with open(EXTRACTED_PATH, 'r', encoding='utf-8') as f:
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

    print(f"    → Сохранено: {len(embeddings)} эмбеддингов")

def search_loop(query_text, model, resumes, embeddings, top_k=10):
    # Подготовка запроса
    query = {
        "positionName": query_text,
        "experience": 3,
        "hardSkills": query_text.lower().split(),
        "salary": 100000
    }

    text = query["positionName"] + " " + " ".join(query["hardSkills"])
    query_vec = model.encode(text)

    # Поиск ближайших соседей
    nn = NearestNeighbors(n_neighbors=50, metric='cosine')
    nn.fit(embeddings)
    _, indices = nn.kneighbors([query_vec], n_neighbors=50)

    # Ранжирование по более продвинутым критериям
    ranked = []
    keywords = set(query["hardSkills"])
    strong_keywords = {"история", "учитель", "обществознание"}

    for idx in indices[0]:
        resume = resumes[idx]
        score = 0

        # Название позиции
        pos_tokens = set(resume.get("positionName", "").lower().split())
        pos_match = len(keywords & pos_tokens)
        strong_match = len(strong_keywords & pos_tokens)

        # Навыки
        skills = set(resume.get("hardSkills", []))
        skill_match = len(keywords & skills)

        # Опыт
        experience_score = max(0, 10 - abs(query.get("experience", 0) - resume.get("experience", 0)))

        # Зарплата
        salary_score = max(0, 10 - abs(query.get("salary", 0) - resume.get("salary", 0)) // 10000)

        # Общая оценка
        score = (
            pos_match * 3 +
            skill_match * 2 +
            strong_match * 5 +
            experience_score +
            salary_score
        )

        ranked.append((score, resume))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]


    print("\nТоп-10 подходящих резюме:")
    for i, res in enumerate(top10):
         print(f"\n--- Кандидат {i+1} ---")
         print(f"Должность: {res.get('positionName', '-')}")
         print(f"Опыт: {res.get('experience', '-')}")
         print(f"Город: {res.get('localityName', '-')}")
         print(f"Образование: {res.get('education', '-')}")
         print(f"Зарплата: {res.get('salary', '-')}")
         print(f"Навыки: {', '.join(res.get('hardSkills', []))}")

# === Запуск всех шагов ===
if __name__ == "__main__":
    extract_10k()
    build_index()
    search_loop()
