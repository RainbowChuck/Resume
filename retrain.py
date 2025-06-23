import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sqlalchemy.orm import sessionmaker, joinedload
from database import engine
import models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATABASE_URL = "sqlite:///./app.db"
MODEL_NAME = "cointegrated/rubert-tiny2"
RETRAINED_MODEL_PATH = "models/retrained_rubert"
BATCH_SIZE = 16
EPOCHS = 1

# --- Database Setup ---
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_training_data():
    """
    Fetches approved and rejected candidates from the database
    and prepares them as training examples.
    """
    db = SessionLocal()
    try:
        # Fetch candidates that have a definitive status and a link to a search query
        approved_candidates = db.query(models.Candidate).options(joinedload(models.Candidate.search_history)).filter(
            models.Candidate.search_history_id.isnot(None),
            models.Candidate.status.in_(['approved', 'approved_by_dean'])
        ).all()

        rejected_candidates = db.query(models.Candidate).options(joinedload(models.Candidate.search_history)).filter(
            models.Candidate.search_history_id.isnot(None),
            models.Candidate.status.in_(['rejected', 'rejected_by_dean'])
        ).all()

        print(f"Found {len(approved_candidates)} positive examples.")
        print(f"Found {len(rejected_candidates)} negative examples.")

        train_examples = []
        for candidate in approved_candidates:
            if candidate.search_history:
                # Positive example: query and resume are a good match (label 1.0)
                train_examples.append(InputExample(texts=[candidate.search_history.query, candidate.resume_text], label=1.0))

        for candidate in rejected_candidates:
            if candidate.search_history:
                 # Negative example: query and resume are a bad match (label 0.0)
                train_examples.append(InputExample(texts=[candidate.search_history.query, candidate.resume_text], label=0.0))
        
        return train_examples

    finally:
        db.close()

def run_retraining():
    """
    Основная функция для запуска процесса дообучения модели.
    """
    print("--- Запуск дообучения модели ---")

    # 1. Загрузка предобученной модели
    print(f"Загрузка базовой модели: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # 2. Получение обучающих данных
    print("Получение обучающих данных из базы...")
    train_examples = get_training_data()

    if not train_examples:
        print("Не найдено обучающих данных. Выход.")
        return

    # Разделение на обучающую и валидационную выборки
    train_data, val_data = train_test_split(train_examples, test_size=0.2, random_state=42)
    print(f"Обучение на {len(train_data)} примерах, валидация на {len(val_data)} примерах.")

    # 3. DataLoader
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

    # 4. Функция потерь
    train_loss = losses.CosineSimilarityLoss(model)

    # 5. Дообучение
    print(f"Старт дообучения на {EPOCHS} эпох(и)...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=EPOCHS,
              warmup_steps=100,
              output_path=RETRAINED_MODEL_PATH,
              show_progress_bar=True)

    # Явное сохранение модели
    model.save(RETRAINED_MODEL_PATH)

    print(f"--- Дообучение завершено ---")
    print(f"Модель сохранена в: {RETRAINED_MODEL_PATH}")

    # --- Оценка ---
    def evaluate(model, val_data):
        queries = [ex.texts[0] for ex in val_data]
        docs = [ex.texts[1] for ex in val_data]
        labels = [ex.label for ex in val_data]

        query_emb = model.encode(queries, convert_to_numpy=True)
        doc_emb = model.encode(docs, convert_to_numpy=True)
        sims = np.array([np.dot(q, d) / (np.linalg.norm(q) * np.linalg.norm(d)) for q, d in zip(query_emb, doc_emb)])

        preds = (sims > 0.5).astype(float)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        print(f"Валидация: Точность={acc:.3f}, F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")

    evaluate(model, val_data)

if __name__ == "__main__":
    run_retraining() 