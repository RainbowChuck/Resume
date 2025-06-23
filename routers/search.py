from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session, joinedload

from database import get_db
import models
from dependencies import get_current_user
from shared import templates, model, resumes_data, embeddings, user_last_search_results
from utils import log_user_action, update_user_activity

router = APIRouter(
    dependencies=[Depends(get_current_user)]
)

@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, current_user: models.User = Depends(get_current_user)):
    db = next(get_db())
    update_user_activity(db, current_user)
    settings = db.query(models.SystemSettings).first()
    if not settings:
        settings = models.SystemSettings()
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return templates.TemplateResponse("search.html", {"request": request, "search_engine": settings.search_engine})

@router.post("/search", response_class=HTMLResponse)
def perform_search(request: Request, query: str = Form(...), city: str = Form(None), gender: str = Form(None), salary_from: str = Form(None), salary_to: str = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    from search_10k import search_resumes
    update_user_activity(db, current_user)
    
    salary_from_int = int(salary_from) if salary_from and salary_from.strip() else None
    salary_to_int = int(salary_to) if salary_to and salary_to.strip() else None

    settings = db.query(models.SystemSettings).first()
    results = search_resumes(query, model, resumes_data, embeddings)

    if settings.search_engine == "advanced":
        filtered_results = []
        for res in results:
            if city and city.strip() and (not res.get('localityName') or city.lower() not in res.get('localityName', '').lower()): continue
            salary_value = res.get('salary')
            if isinstance(salary_value, int):
                result_salary = salary_value
            elif isinstance(salary_value, str):
                result_salary = int(salary_value.replace(' ', '').replace('₽', '')) if salary_value else 0
            else:
                result_salary = 0
            if salary_from_int is not None and result_salary < salary_from_int: continue
            if salary_to_int is not None and result_salary > salary_to_int: continue
            filtered_results.append(res)
        results = filtered_results

    search_entry = models.SearchHistory(query=query, city=city, results=len(results), user_id=current_user.id)
    db.add(search_entry)
    db.commit()
    db.refresh(search_entry)
    
    log_user_action(db, current_user.id, "search", f"Performed search for '{query}', found {len(results)} results.")
    user_last_search_results[current_user.id] = results
    
    return templates.TemplateResponse("search.html", {"request": request, "results": results, "search_engine": settings.search_engine, "search_history_id": search_entry.id})

@router.get("/history", response_class=HTMLResponse)
def get_history(request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user), city: str = "", start: str = "", end: str = ""):
    update_user_activity(db, current_user)
    query = db.query(models.SearchHistory).options(joinedload(models.SearchHistory.user))
    if current_user.role != "dean":
        query = query.filter(models.SearchHistory.user_id == current_user.id)
    if city: query = query.filter(models.SearchHistory.city.ilike(f"%{city}%"))
    if start: query = query.filter(models.SearchHistory.date >= start)
    if end: query = query.filter(models.SearchHistory.date <= end)
    
    results = query.order_by(models.SearchHistory.date.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "history": results, "current_user": current_user})

@router.get("/history/download")
def download_history(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user), city: str = "", start: str = "", end: str = ""):
    import pandas as pd
    query = db.query(models.SearchHistory)
    if current_user.role != "dean":
        query = query.filter(models.SearchHistory.user_id == current_user.id)
    # ... filtering logic from get_history ...
    results = query.order_by(models.SearchHistory.date.desc()).all()
    data = [{"Запрос": r.query, "Дата": r.date.strftime("%Y-%m-%d %H:%M:%S"), "Город": r.city, "Результатов": r.results, "Пользователь": r.user.username if r.user else ""} for r in results]
    df = pd.DataFrame(data)
    file_path = f"/tmp/history_{current_user.id}.xlsx"
    df.to_excel(file_path, index=False)
    return FileResponse(file_path, filename="history.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@router.get("/history/result/{history_id}", response_class=HTMLResponse)
def history_result(history_id: int, request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    from search_10k import search_resumes
    history = db.query(models.SearchHistory).filter(models.SearchHistory.id == history_id, models.SearchHistory.user_id == current_user.id).first()
    if not history: raise HTTPException(status_code=404, detail="История не найдена")
    
    settings = db.query(models.SystemSettings).first()
    results = search_resumes(history.query, model, resumes_data, embeddings)
    
    return templates.TemplateResponse("search.html", {"request": request, "results": results, "history_query": history.query, "history_city": history.city, "search_engine": settings.search_engine}) 