import os
import pickle
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer

# --- Шаблоны ---
templates = Jinja2Templates(directory="templates")

# --- Загрузка модели ---
BASE_MODEL_NAME = "cointegrated/rubert-tiny2"
RETRAINED_MODEL_PATH = "models/retrained_rubert"
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")

def load_search_model():
    """Загружает дообученную модель, если есть веса, иначе базовую модель."""
    def has_model_weights(path):
        if not os.path.exists(path):
            return False
        # Проверка наличия весов или подпапок
        for fname in ["pytorch_model.bin", "model.safetensors", "tf_model.h5", "model.ckpt.index", "flax_model.msgpack"]:
            for root, dirs, files in os.walk(path):
                if fname in files:
                    return True
        # Также проверяем подпапку 0_Transformer
        if os.path.isdir(os.path.join(path, "0_Transformer")):
            return True
        return False

    if has_model_weights(RETRAINED_MODEL_PATH):
        model_path = RETRAINED_MODEL_PATH
    else:
        model_path = BASE_MODEL_NAME
    print(f"Загрузка модели из: {model_path}")
    return SentenceTransformer(model_path)

model = load_search_model()
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)
with open(MAP_PATH, "rb") as f:
    resumes_data = pickle.load(f)

# In-memory store for last search results
user_last_search_results = {} 