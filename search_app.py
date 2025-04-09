from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import pandas as pd
from datetime import datetime
import csv

# === Пути ===
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")
HISTORY_PATH = os.path.join("results", "history.csv")
EXCEL_PATH = os.path.join("results", "last_result.xlsx")

# === Подготовка ===
os.makedirs("results", exist_ok=True)

model = SentenceTransformer("cointegrated/rubert-tiny2")
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)
with open(MAP_PATH, "rb") as f:
    resumes = pickle.load(f)

nn = NearestNeighbors(n_neighbors=10, metric='cosine')
nn.fit(embeddings)

# === FastAPI ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search_resume(request: Request, query: str = Form(...)):
    query_vec = model.encode([query])
    _, indices = nn.kneighbors([query_vec], n_neighbors=10)

    results = []
    for idx in indices[0]:
        res = resumes[idx]
        results.append({
            "position": res.get("positionName", "-"),
            "experience": res.get("experience", "-"),
            "city": res.get("localityName", "-"),
            "education": res.get("education", "-"),
            "salary": res.get("salary", "-")
        })

    # === Сохраняем историю ===
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORY_PATH, "a", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        for res in results:
            writer.writerow([timestamp, query, res["position"], res["experience"], res["city"], res["education"], res["salary"]])

    # === Сохраняем Excel ===
    df = pd.DataFrame(results)
    df.to_excel(EXCEL_PATH, index=False)

    return templates.TemplateResponse("form.html", {"request": request, "results": results, "query": query})


@app.get("/download", response_class=FileResponse)
async def download_excel():
    return FileResponse(EXCEL_PATH, filename="резюме.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
