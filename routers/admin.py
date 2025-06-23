from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from collections import defaultdict
from datetime import datetime
import subprocess
import sys
import json

from database import get_db
import models
from dependencies import get_current_user
from shared import templates, load_search_model
from utils import log_user_action, update_user_activity

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_user)]
)

def is_admin(user: models.User = Depends(get_current_user)):
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail="Нет прав для этого действия")

@router.get("/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # This endpoint is also the main dashboard for admins
    update_user_activity(db, current_user)
    
    # Simplified stats for the main view
    total_users = db.query(models.User).count()
    total_candidates = db.query(models.Candidate).count()
    total_searches = db.query(models.SearchHistory).count()
    approved_candidates = db.query(models.Candidate).filter(models.Candidate.status == "approved").count()
    users = db.query(models.User).all()

    for user in users:
        time_diff = (datetime.utcnow() - user.last_activity).total_seconds() if user.last_activity else float('inf')
        user.is_active = time_diff < 15 * 60

    recent_actions = db.query(models.UserAction).options(joinedload(models.UserAction.user)).order_by(models.UserAction.created_at.desc()).limit(10).all()

    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request, "user": current_user, "users": users, "total_users": total_users,
        "total_candidates": total_candidates, "total_searches": total_searches,
        "approved_candidates": approved_candidates, "recent_actions": recent_actions
    })

@router.get("/statistics", response_class=HTMLResponse, dependencies=[Depends(is_admin)])
def statistics(request: Request, start_date: str = None, end_date: str = None, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)

    # Base queries
    actions_query = db.query(models.UserAction)
    logins_query = db.query(func.date(models.UserAction.created_at), func.count(models.UserAction.id)).filter(models.UserAction.action_type == 'login')

    # Date filtering
    if start_date:
        actions_query = actions_query.filter(models.UserAction.created_at >= start_date)
        logins_query = logins_query.filter(models.UserAction.created_at >= start_date)
    if end_date:
        actions_query = actions_query.filter(models.UserAction.created_at <= end_date)
        logins_query = logins_query.filter(models.UserAction.created_at <= end_date)

    # User activity breakdown
    user_activity = defaultdict(lambda: defaultdict(int))
    for action in actions_query.all():
        username = action.user.username if action.user else "unknown"
        user_activity[username][action.action_type] += 1

    # Logins per day
    logins_by_day = logins_query.group_by(func.date(models.UserAction.created_at)).all()

    # Candidate status for chart
    status_counts = db.query(models.Candidate.status, func.count(models.Candidate.id)).group_by(models.Candidate.status).all()
    candidate_status_labels = [status for status, count in status_counts]
    candidate_status_data = [count for status, count in status_counts]

    # User activity for chart
    activity_data = db.query(func.date(models.UserAction.created_at).label('date'), func.count(models.UserAction.id).label('count')).group_by('date').all()
    activity_dates = [item.date for item in activity_data]
    activity_counts = [item.count for item in activity_data]
    
    total_users = db.query(models.User).count()
    total_candidates = db.query(models.Candidate).count()
    total_searches = db.query(models.SearchHistory).count()
    approved_candidates = db.query(models.Candidate).filter(models.Candidate.status == "approved").count()
    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "user_activity": dict(user_activity),
        "logins_by_day": logins_by_day,
        "candidate_status_labels": json.dumps(candidate_status_labels),
        "candidate_status_data": json.dumps(candidate_status_data),
        "activity_dates": json.dumps(activity_dates),
        "activity_counts": json.dumps(activity_counts),
        "total_users": total_users,
        "total_candidates": total_candidates,
        "total_searches": total_searches,
        "approved_candidates": approved_candidates
    })

@router.get("/settings", response_class=HTMLResponse, dependencies=[Depends(is_admin)])
def settings_page(request: Request, db: Session = Depends(get_db)):
    settings = db.query(models.SystemSettings).first()
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings})

@router.post("/settings", dependencies=[Depends(is_admin)])
def update_settings(system_name: str = Form(...), items_per_page: int = Form(...), search_engine: str = Form(...), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    settings = db.query(models.SystemSettings).first()
    settings.system_name = system_name
    settings.items_per_page = items_per_page
    settings.search_engine = search_engine
    db.commit()
    log_user_action(db, current_user.id, "update_settings", "System settings updated.")
    return RedirectResponse(url="/admin/settings", status_code=302)

@router.post("/retrain-model", status_code=202, dependencies=[Depends(is_admin)])
def trigger_retraining(current_user: models.User = Depends(get_current_user)):
    try:
        subprocess.Popen([sys.executable, "retrain.py"])
        log_user_action(get_db().__next__(), current_user.id, "retrain_start", "Started model retraining.")
        return {"message": "Model retraining process started."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reload-model", dependencies=[Depends(is_admin)])
def reload_model_endpoint(current_user: models.User = Depends(get_current_user)):
    import shared
    shared.model = shared.load_search_model()
    log_user_action(get_db().__next__(), current_user.id, "model_reload", "Search model reloaded.")
    return {"message": "Search model has been reloaded."} 