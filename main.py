import ast
from collections import defaultdict
from sqlalchemy import func

from fastapi import FastAPI, Request, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session

import models
from database import engine, get_db
from routers import authentication, candidates, search, admin, users, dean
from dependencies import get_current_user
from shared import templates
from utils import update_user_activity

# Создать все таблицы базы данных
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# --- Фильтры шаблонов ---
def parse_resume_data(resume_text):
    """Парсинг данных резюме из строки для отображения в шаблонах."""
    if not resume_text: return None
    try:
        # Обрабатывает как JSON, так и строковое представление dict
        return ast.literal_eval(resume_text)
    except (ValueError, SyntaxError):
        return {"raw_text": resume_text} # На случай некорректной строки
templates.env.filters["parse_resume"] = parse_resume_data

# --- Root and Dashboard ---
@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/login")

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)
    if current_user.role == "admin":
        return RedirectResponse(url="/admin/dashboard")
    
    status_counts = db.query(models.Candidate.status, func.count(models.Candidate.id)).filter(models.Candidate.user_id == current_user.id).group_by(models.Candidate.status).all()
    funnel_data = defaultdict(int, status_counts)
    
    template_name = f"{current_user.role}_dashboard.html"
    return templates.TemplateResponse(template_name, {"request": request, "user": current_user, "funnel_data": funnel_data})

# --- Include Routers ---
app.include_router(authentication.router)
app.include_router(candidates.router)
app.include_router(search.router)
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(dean.router)
