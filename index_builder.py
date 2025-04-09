from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from utils import read_large_json, extract_resume_text
from tqdm import tqdm
import pickle
import os

# # Пути
# DATA_PATH = os.path.join("data", "cv.json")
# EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings.pkl")
# MAP_PATH = os.path.join("models", "resume_id_map.pkl")
# CHUNK_SIZE = 100000
#
# # Модель
# print("Загрузка модели...")
# model = SentenceTransformer("cointegrated/rubert-tiny2")
#
# # Хранилища
# all_embeddings = []
# resume_id_map = []
#
# # Обработка JSON по частямрфкь
# print("Обработка данных...")
# for i, chunk in enumerate(read_large_json(DATA_PATH, CHUNK_SIZE)):
#     print(f"Обрабатываем чанк {i + 1}")
#     texts = [extract_resume_text(r) for r in chunk]
#     print(f"Количество текстов в чанке: {len(texts)}")
#     texts = [text for text in texts if text.strip()]
#     if not texts:
#         print("Все тексты пустые, пропускаем чанк...")
#         continue
#     embeddings = model.encode(texts, show_progress_bar=True)
#     all_embeddings.extend(embeddings)
#     resume_id_map.extend(chunk)
#
# # Сохраняем эмбеддинги и карту резюме
# print("Сохраняем данные...")
# with open(EMBEDDINGS_PATH, "wb") as f:
#     pickle.dump(all_embeddings, f)
#
# with open(MAP_PATH, "wb") as f:
#     pickle.dump(resume_id_map, f)
#
# print("Готово.")

#################################################################################################

# from sentence_transformers import SentenceTransformer
# from utils import read_large_json, extract_resume_text
# from tqdm import tqdm
# import pickle
# import os
# import numpy as np
#
# # Пути
# DATA_PATH = os.path.join("data", "cv.json")
# EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings.pkl")
# MAP_PATH = os.path.join("models", "resume_id_map.pkl")
# CHUNK_SIZE = 100000
#
# # Создаем директории
# os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
# os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
#
# # Загрузка модели
# print("Загрузка модели...")
# try:
#     model = SentenceTransformer("cointegrated/rubert-tiny2", device='cuda')
# except Exception as e:
#     print(f"Ошибка при загрузке модели: {e}")
#     exit(1)
#
# # Хранилища
# all_embeddings = []
# resume_id_map = []
#
# # Обработка JSON по частям
# print("Обработка данных...")
# for i, chunk in enumerate(read_large_json(DATA_PATH, CHUNK_SIZE)):
#     print(f"Обрабатываем чанк {i + 1}")
#     if not chunk:
#         print("Чанк пустой, пропускаем...")
#         continue
#
#     # Извлечение текста
#     texts = []
#     for r in chunk:
#         try:
#             text = extract_resume_text(r)
#             texts.append(text)
#         except Exception as e:
#             print(f"Ошибка при обработке резюме: {e}")
#             texts.append("")  # Добавляем пустую строку вместо сломанного резюме
#
#     # Логирование
#     print(f"Количество текстов до фильтрации: {len(texts)}")
#     texts = [text for text in texts if text.strip()]
#     print(f"Количество текстов после фильтрации: {len(texts)}")
#
#     if not texts:
#         print("Все тексты пустые, пропускаем чанк...")
#         continue
#
#     # Генерация эмбеддингов
#     embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True).to('cuda')
#     print(f"Количество сгенерированных эмбеддингов: {len(embeddings)}")
#     all_embeddings.extend(embeddings)
#     resume_id_map.extend(chunk)
#
# # Проверка на пустой массив
# if not all_embeddings:
#     print("Массив эмбеддингов пуст. Проверьте обработку данных.")
#     exit(1)
#
# # Преобразование в NumPy массив
# all_embeddings = np.array(all_embeddings)
# if len(all_embeddings.shape) == 1:
#     all_embeddings = all_embeddings.reshape(-1, 1)
#
# # Сохранение данных
# print("Сохраняем данные...")
# try:
#     temp_embeddings_path = EMBEDDINGS_PATH + ".tmp"
#     with open(temp_embeddings_path, "wb") as f:
#         pickle.dump(all_embeddings, f)
#     os.replace(temp_embeddings_path, EMBEDDINGS_PATH)
#
#     temp_map_path = MAP_PATH + ".tmp"
#     with open(temp_map_path, "wb") as f:
#         pickle.dump(resume_id_map, f)
#     os.replace(temp_map_path, MAP_PATH)
# except Exception as e:
#     print(f"Ошибка при сохранении данных: {e}")
#     exit(1)
#
# print("Готово.")


######################################################################



import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
from sentence_transformers import SentenceTransformer
from utils import read_large_json, extract_resume_text
import pickle
import os
import torch
import numpy as np

# Пути к файлам
DATA_PATH = os.path.join("data", "cv.json")
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings.pkl")
MAP_PATH = os.path.join("models", "resume_id_map.pkl")
CHUNK_SIZE = 100000

# Глобальные переменные для хранения данных
all_embeddings = []
resume_id_map = []

# Функция для создания индексов
def create_index(progress_callback):
    global all_embeddings, resume_id_map

    # Загрузка модели
    model = SentenceTransformer("cointegrated/rubert-tiny2")

    # Чтение JSON по частям
    total_chunks = sum(1 for _ in read_large_json(DATA_PATH, CHUNK_SIZE))  # Подсчет общего количества чанков
    chunk_count = 0

    for i, chunk in enumerate(read_large_json(DATA_PATH, CHUNK_SIZE)):
        chunk_count += 1
        progress = (chunk_count / total_chunks) * 100  # Вычисление прогресса в процентах
        progress_callback(progress)  # Обновление прогресса

        # Извлечение текста
        texts = [extract_resume_text(r) for r in chunk]
        texts = [text for text in texts if text.strip()]

        # Генерация эмбеддингов
        embeddings = model.encode(texts, show_progress_bar=False)
        all_embeddings.extend(embeddings)
        resume_id_map.extend(chunk)

    # Сохранение данных
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(all_embeddings, f)
    with open(MAP_PATH, "wb") as f:
        pickle.dump(resume_id_map, f)

    progress_callback(100)  # Установка прогресса в 100%

# Функция для запуска процесса в отдельном потоке
def start_index_creation():
    def run_task():
        create_index(update_progress)

    thread = Thread(target=run_task)
    thread.start()

# Функция для обновления прогресса
def update_progress(value):
    progress_bar["value"] = value
    root.update_idletasks()  # Обновление интерфейса

# Создание графического интерфейса
root = tk.Tk()
root.title("Создание индексов")

# Прогресс-бар
progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=20)

# Кнопка "Старт"
start_button = tk.Button(root, text="Начать создание индексов", command=start_index_creation)
start_button.pack(pady=10)

# Запуск интерфейса
root.mainloop()