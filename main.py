from fastapi import FastAPI, Request, Form, Depends, status, HTTPException, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session, joinedload
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy import func
from collections import defaultdict
import pandas as pd

from search_10k import search_resumes
# from search_app import search_resume
from auth import verify_password, create_access_token, SECRET_KEY, ALGORITHM, hash_password
from database import get_db, engine
import models
from schemas import UserOut, UserCreate
import os
import pickle
from sentence_transformers import SentenceTransformer

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Load search model and data
EMBEDDINGS_PATH = os.path.join("models", "resume_embeddings_10k.pkl")
MAP_PATH = os.path.join("models", "resume_id_map_10k.pkl")
MODEL_NAME = "cointegrated/rubert-tiny2"

model = SentenceTransformer(MODEL_NAME)
with open(EMBEDDINGS_PATH, "rb") as f:
    embeddings = pickle.load(f)
with open(MAP_PATH, "rb") as f:
    resumes_data = pickle.load(f)

# Store last search results per user (for demo; in production use a better approach)
user_last_search_results = {}

def is_user_active(user: models.User) -> bool:
    """Check if user is currently active (active within last 15 minutes)"""
    if not user.last_activity:
        return False
    time_diff = datetime.utcnow() - user.last_activity
    return time_diff.total_seconds() < 15 * 60  # 15 minutes

def update_user_activity(db: Session, user: models.User):
    """Update user's last activity timestamp"""
    user.last_activity = datetime.utcnow()
    db.commit()

@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/login")
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/logout")
def logout(access_token: str = Cookie(None), db: Session = Depends(get_db)):
    if access_token:
        try:
            payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username:
                user = db.query(models.User).filter(models.User.username == username).first()
                if user:
                    # Mark user as inactive by setting last_activity to 20 minutes ago
                    user.last_activity = datetime.utcnow() - timedelta(minutes=20)
                    db.commit()
                    log_user_action(db, user.id, "logout", f"User {user.username} logged out")
        except JWTError:
            pass  # Invalid token, just continue with logout
    
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("access_token")
    return response

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": {}, "error": "Неверные данные"}, status_code=401)

    # Update last activity timestamp
    update_user_activity(db, user)
    log_user_action(db, user.id, "login", f"User {user.username} logged in")

    token = create_access_token({"sub": user.username})
    response = RedirectResponse(url="/dashboard", status_code=302)
    response.set_cookie(key="access_token", value=token, httponly=True)
    return response

@app.get("/register", response_class=HTMLResponse)
def show_register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db)
):
    existing = db.query(models.User).filter(models.User.username == username).first()
    if existing:
        return templates.TemplateResponse("register.html", {
            "request": {},
            "error": "Пользователь уже существует"
        }, status_code=400)

    user = models.User(
        username=username,
        hashed_password=hash_password(password),
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_user_action(db, user.id, "register", f"New user {username} registered with role {role}")
    response = RedirectResponse(url="/login", status_code=302)
    return response

def get_current_user(access_token: str = Cookie(None), db: Session = Depends(get_db)) -> models.User:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated (no cookie)")

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

@app.get("/history", response_class=HTMLResponse)
def get_history(
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    city: str = "",
    start: str = "",
    end: str = ""
):
    # Update user activity when they view history
    update_user_activity(db, current_user)
    
    if current_user.role == "dean":
        query = db.query(models.SearchHistory).options(joinedload(models.SearchHistory.user))
    else:
        query = db.query(models.SearchHistory).filter(models.SearchHistory.user_id == current_user.id).options(joinedload(models.SearchHistory.user))

    if city:
        query = query.filter(models.SearchHistory.city.ilike(f"%{city}%"))
    if start:
        query = query.filter(models.SearchHistory.date >= start)
    if end:
        query = query.filter(models.SearchHistory.date <= end)

    results = query.order_by(models.SearchHistory.date.desc()).all()

    return templates.TemplateResponse("history.html", {"request": request, "history": results, "current_user": current_user})

@app.get("/candidates", response_class=HTMLResponse)
def list_candidates(
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Update user activity when they view candidates
    update_user_activity(db, current_user)
    
    candidates = db.query(models.Candidate).filter(models.Candidate.user_id == current_user.id).all()
    return templates.TemplateResponse("candidates.html", {
        "request": request,
        "candidates": candidates
    })

@app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
def view_candidate(candidate_id: int, request: Request,
                   db: Session = Depends(get_db),
                   current_user: models.User = Depends(get_current_user)):
    # Update user activity when they view candidates
    update_user_activity(db, current_user)
    
    candidate = db.query(models.Candidate).filter(
        models.Candidate.id == candidate_id,
        models.Candidate.user_id == current_user.id
    ).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    return templates.TemplateResponse("candidate_detail.html", {
        "request": request,
        "candidate": candidate,
        "current_user": current_user
    })

@app.post("/candidates/{candidate_id}/update")
def update_candidate_status(
    candidate_id: int,
    action: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Update user activity when they modify candidates
    update_user_activity(db, current_user)
    
    candidate = db.query(models.Candidate).filter(
        models.Candidate.id == candidate_id,
        models.Candidate.user_id == current_user.id
    ).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    old_status = candidate.status
    if current_user.role == "hr":
        if action == "approve":
            candidate.status = "approved"
        elif action == "reject":
            candidate.status = "rejected"
    elif current_user.role == "manager":
        if action == "hire":
            candidate.status = "hired"
        elif action == "decline":
            candidate.status = "declined"
    db.commit()
    log_user_action(
        db, current_user.id, "status_change",
        f"Status for candidate {candidate_id} changed from {old_status} to {candidate.status}"
    )
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@app.post("/candidates/{candidate_id}/update_test_results")
def update_test_results(
    candidate_id: int,
    test_results: str = Form(None),  # Make it optional
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != "hr":
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    candidate = db.query(models.Candidate).filter(
        models.Candidate.id == candidate_id,
        models.Candidate.user_id == current_user.id
    ).first()
    
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    
    # Only update if new results are provided and not empty
    if test_results and test_results.strip():
        # If there are existing results, append the new ones
        if candidate.test_results:
            candidate.test_results = candidate.test_results + "\n\n" + test_results
        else:
            candidate.test_results = test_results
        
        db.commit()
        
        log_user_action(
            db, current_user.id, "update_test_results",
            f"Test results for candidate {candidate_id} updated."
        )
    
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@app.post("/candidates/{candidate_id}/update_video_notes")
def update_video_notes(
    candidate_id: int,
    video_interview_notes: str = Form(None),  # Make it optional
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != "hr":
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    candidate = db.query(models.Candidate).filter(
        models.Candidate.id == candidate_id,
        models.Candidate.user_id == current_user.id
    ).first()
    
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    
    # Only update if new notes are provided and not empty
    if video_interview_notes and video_interview_notes.strip():
        # If there are existing notes, append the new ones
        if candidate.video_interview_notes:
            candidate.video_interview_notes = candidate.video_interview_notes + "\n\n" + video_interview_notes
        else:
            candidate.video_interview_notes = video_interview_notes
        
        db.commit()
        
        log_user_action(
            db, current_user.id, "update_video_notes",
            f"Video interview notes for candidate {candidate_id} updated."
        )
    
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@app.get("/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    current_user: models.User = Depends(get_current_user)
):
    # Update user activity when they access search page
    db = next(get_db())
    update_user_activity(db, current_user)
    
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
def perform_search(
        request: Request,
        query: str = Form(...),
        city: str = Form(None),
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)
):
    # Update user activity when they perform searches
    update_user_activity(db, current_user)
    
    results = search_resumes(query, model, resumes_data, embeddings)
    # Save number of results
    search_entry = models.SearchHistory(
        query=query,
        city=city,
        results=len(results),
        user_id=current_user.id
    )
    db.add(search_entry)
    db.commit()
    log_user_action(
        db, current_user.id, "search",
        f"Performed search for '{query}' in '{city}', found {len(results)} results."
    )
    # Store results for add-to-candidates
    user_last_search_results[current_user.id] = results
    return templates.TemplateResponse("search.html", {
        "request": request,
        "results": results
    })

@app.post("/candidates/add")
def add_candidate(
    request: Request,
    positionName: str = Form(...),
    experience: str = Form(...),
    localityName: str = Form(...),
    salary: str = Form(...),
    hardSkills: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Compose resume text for storage
    resume_dict = {
        "positionName": positionName,
        "experience": experience,
        "localityName": localityName,
        "salary": salary,
        "hardSkills": hardSkills.split(",") if hardSkills else []
    }
    candidate = models.Candidate(
        name=positionName or "-",
        email="",  # No email in resume data
        phone="",  # No phone in resume data
        resume_text=str(resume_dict),
        status="new",
        user_id=current_user.id
    )
    db.add(candidate)
    db.commit()
    return RedirectResponse(url="/candidates", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
        request: Request,
        db: Session = Depends(get_db),
        current_user: models.User = Depends(get_current_user)
):
    # Update user activity when they access dashboard
    update_user_activity(db, current_user)
    
    # Данные для воронок
    status_counts = db.query(
        models.Candidate.status, func.count(models.Candidate.id)
    ).filter(models.Candidate.user_id == current_user.id).group_by(models.Candidate.status).all()

    funnel_data = defaultdict(int)
    for status, count in status_counts:
        funnel_data[status] = count

    # Статистика по действиям
    actions_stats = db.query(
        models.UserAction.action_type, func.count(models.UserAction.id)
    ).group_by(models.UserAction.action_type).all()

    # Динамика найма
    hiring_dynamics = db.query(
        func.date(models.Candidate.status_update_date),
        func.count(models.Candidate.id)
    ).filter(
        models.Candidate.status == 'hired',
        models.Candidate.user_id == current_user.id
    ).group_by(func.date(models.Candidate.status_update_date)).all()

    if current_user.role == "admin":
        total_users = db.query(models.User).count()
        total_candidates = db.query(models.Candidate).count()
        total_searches = db.query(models.SearchHistory).count()
        approved_candidates = db.query(models.Candidate).filter(models.Candidate.status == "approved").count()
        users = db.query(models.User).all()
        
        # Add active status to each user
        for user in users:
            user.is_active = is_user_active(user)
            
        recent_actions = db.query(models.UserAction).options(joinedload(models.UserAction.user)).order_by(models.UserAction.created_at.desc()).limit(10).all()
        return templates.TemplateResponse("admin_dashboard.html", {
            "request": request,
            "funnel_data": funnel_data,
            "actions_stats": actions_stats,
            "hiring_dynamics": hiring_dynamics,
            "user": current_user,
            "total_users": total_users,
            "total_candidates": total_candidates,
            "total_searches": total_searches,
            "approved_candidates": approved_candidates,
            "users": users,
            "recent_actions": recent_actions
        })
    elif current_user.role == "hr":
        return templates.TemplateResponse("hr_dashboard.html", {
            "request": request,
            "funnel_data": funnel_data,
            "actions_stats": actions_stats,
            "hiring_dynamics": hiring_dynamics,
            "user": current_user
        })
    elif current_user.role == "dean":
        return templates.TemplateResponse("dean_dashboard.html", {
            "request": request,
            "funnel_data": funnel_data,
            "actions_stats": actions_stats,
            "hiring_dynamics": hiring_dynamics,
            "user": current_user
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "funnel_data": funnel_data,
        "actions_stats": actions_stats,
        "hiring_dynamics": hiring_dynamics,
        "user": current_user
    })

@app.get("/dean/candidates", response_class=HTMLResponse)
def dean_candidates(
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'dean':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    # Update user activity when they view candidates
    update_user_activity(db, current_user)
    
    candidates = db.query(models.Candidate).filter(models.Candidate.status == 'approved').all()
    
    return templates.TemplateResponse("dean_candidates.html", {
        "request": request,
        "candidates": candidates
    })

@app.post("/dean/candidates/{candidate_id}/action")
def dean_candidate_action(
    candidate_id: int,
    action: str = Form(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'dean':
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")

    if action == "accept":
        candidate.status = "approved_by_dean"
    elif action == "reject":
        candidate.status = "rejected_by_dean"
    
    db.commit()
    
    return RedirectResponse(url="/dean/candidates", status_code=302)

@app.post("/users/", response_model=UserOut)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Только админы могут создавать
    # if current_user.role != 'admin':
    #     raise HTTPException(status_code=403, detail="Not enough permissions")
    existing = db.query(models.User).filter(models.User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = models.User(
        username=user_in.username,
        hashed_password=hash_password(user_in.password),
        role=user_in.role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.get("/users/{user_id}/edit", response_class=HTMLResponse)
def edit_user_form(
    user_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    user_to_edit = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_edit:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    success = request.query_params.get("success")
    return templates.TemplateResponse(
        "edit_user.html",
        {"request": request, "user": user_to_edit, "success": success}
    )

@app.post("/users/{user_id}/edit")
def edit_user(
    user_id: int,
    username: str = Form(...),
    role: str = Form(...),
    password: str = Form(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    user_to_edit = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_edit:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    user_to_edit.username = username
    user_to_edit.role = role
    if password:
        user_to_edit.hashed_password = hash_password(password)
    db.commit()
    log_user_action(db, current_user.id, "edit_user", f"User {username} (ID: {user_id}) updated.")
    # Redirect to edit page with success message
    return RedirectResponse(url=f"/users/{user_id}/edit?success=1", status_code=302)

@app.post("/users/{user_id}/delete")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    user_to_delete = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    db.delete(user_to_delete)
    db.commit()
    log_user_action(db, current_user.id, "delete_user", f"User ID {user_id} deleted.")
    # Redirect to admin dashboard after deletion with a success message
    return RedirectResponse(url="/admin/users?deleted=1", status_code=302)

def log_user_action(db: Session, user_id: int, action_type: str, description: str):
    action = models.UserAction(
        user_id=user_id,
        action_type=action_type,
        description=description
    )
    db.add(action)
    db.commit()

@app.get("/statistics", response_class=HTMLResponse)
def statistics(
    request: Request,
    start_date: str = None,
    end_date: str = None,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")

    # Update user activity when they view statistics
    update_user_activity(db, current_user)

    actions_query = db.query(models.UserAction)
    if start_date:
        actions_query = actions_query.filter(models.UserAction.created_at >= start_date)
    if end_date:
        actions_query = actions_query.filter(models.UserAction.created_at <= end_date)

    actions = actions_query.all()

    user_activity = defaultdict(lambda: defaultdict(int))
    for action in actions:
        username = action.user.username if action.user is not None else "unknown"
        user_activity[username][action.action_type] += 1

    logins_by_day = db.query(
        func.date(models.UserAction.created_at),
        func.count(models.UserAction.id)
    ).filter(models.UserAction.action_type == 'login')
    if start_date:
        logins_by_day = logins_by_day.filter(models.UserAction.created_at >= start_date)
    if end_date:
        logins_by_day = logins_by_day.filter(models.UserAction.created_at <= end_date)
    logins_by_day = logins_by_day.group_by(func.date(models.UserAction.created_at)).all()

    # Конверсия воронки
    funnel_stages = ["new", "review", "test", "interview", "offer", "hired"]
    funnel_conversion = {}
    for stage in funnel_stages:
        count = db.query(func.count(models.Candidate.id)).filter(models.Candidate.status == stage).scalar()
        funnel_conversion[stage] = count

    # Candidate status labels and data for chart
    status_counts = db.query(
        models.Candidate.status,
        func.count(models.Candidate.id)
    ).group_by(models.Candidate.status).all()
    candidate_status_labels = []
    candidate_status_data = []
    for status, count in status_counts:
        candidate_status_labels.append(status)
        candidate_status_data.append(count)

    # User activity data for chart
    activity_data = db.query(
        func.date(models.UserAction.created_at).label('date'),
        func.count(models.UserAction.id).label('count')
    ).group_by(func.date(models.UserAction.created_at)).all()
    activity_dates = []
    activity_counts = []
    for date, count in activity_data:
        activity_dates.append(date)
        activity_counts.append(count)

    # Overall statistics
    total_users = db.query(models.User).count()
    total_candidates = db.query(models.Candidate).count()
    total_searches = db.query(models.SearchHistory).count()
    approved_candidates = db.query(models.Candidate).filter(models.Candidate.status == "approved").count()
    users = db.query(models.User).all()
    recent_actions = db.query(models.UserAction).options(joinedload(models.UserAction.user)).order_by(models.UserAction.created_at.desc()).limit(10).all()

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "user_activity": dict(user_activity),
        "logins_by_day": logins_by_day,
        "funnel_conversion": funnel_conversion,
        "candidate_status_labels": candidate_status_labels,
        "candidate_status_data": candidate_status_data,
        "activity_dates": activity_dates,
        "activity_counts": activity_counts,
        "total_users": total_users,
        "total_candidates": total_candidates,
        "total_searches": total_searches,
        "approved_candidates": approved_candidates,
        "users": users,
        "recent_actions": recent_actions
    })

@app.get("/settings", response_class=HTMLResponse)
def settings_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    # Update user activity when they view settings
    update_user_activity(db, current_user)
    
    settings = db.query(models.SystemSettings).first()
    if not settings:
        settings = models.SystemSettings()
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return templates.TemplateResponse("settings.html", {"request": request, "settings": settings})

@app.post("/settings")
def update_settings(
    request: Request,
    system_name: str = Form(...),
    items_per_page: int = Form(...),
    search_engine: str = Form(...),
    enable_semantic_search: bool = Form(False),
    notify_new_candidates: bool = Form(False),
    notify_status_changes: bool = Form(False),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    settings = db.query(models.SystemSettings).first()
    settings.system_name = system_name
    settings.items_per_page = items_per_page
    settings.search_engine = search_engine
    settings.enable_semantic_search = enable_semantic_search
    settings.notify_new_candidates = notify_new_candidates
    settings.notify_status_changes = notify_status_changes
    db.commit()
    log_user_action(db, current_user.id, "update_settings", "System settings updated.")
    return RedirectResponse(url="/settings", status_code=302)

@app.get("/admin/users", response_class=HTMLResponse)
def manage_users(
    request: Request,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Недостаточно прав")
    
    # Update user activity when they manage users
    update_user_activity(db, current_user)
    
    users = db.query(models.User).all()
    
    # Add active status to each user
    for user in users:
        user.is_active = is_user_active(user)
    
    total_users = db.query(models.User).count()
    total_candidates = db.query(models.Candidate).count()
    total_searches = db.query(models.SearchHistory).count()
    approved_candidates = db.query(models.Candidate).filter(models.Candidate.status == "approved").count()
    recent_actions = db.query(models.UserAction).options(joinedload(models.UserAction.user)).order_by(models.UserAction.created_at.desc()).limit(10).all()
    deleted = request.query_params.get("deleted")
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "users": users,
        "user": current_user,
        "total_users": total_users,
        "total_candidates": total_candidates,
        "total_searches": total_searches,
        "approved_candidates": approved_candidates,
        "recent_actions": recent_actions,
        "deleted": deleted
    })

@app.get("/history/download")
def download_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
    city: str = "",
    start: str = "",
    end: str = ""
):
    # For dean, export all users' history; for others, only their own
    if current_user.role == "dean":
        query = db.query(models.SearchHistory)
    else:
        query = db.query(models.SearchHistory).filter(models.SearchHistory.user_id == current_user.id)
    if city:
        query = query.filter(models.SearchHistory.city.ilike(f"%{city}%"))
    if start:
        query = query.filter(models.SearchHistory.date >= start)
    if end:
        query = query.filter(models.SearchHistory.date <= end)
    results = query.order_by(models.SearchHistory.date.desc()).all()
    # Prepare data for Excel
    data = [
        {
            "Запрос": row.query,
            "Дата": row.date.strftime("%Y-%m-%d %H:%M:%S"),
            "Город": row.city,
            "Результатов": row.results,
            "Пользователь": row.user.username if hasattr(row, "user") and row.user else ""
        }
        for row in results
    ]
    import pandas as pd
    df = pd.DataFrame(data)
    file_path = f"/tmp/history_{current_user.id}.xlsx"
    df.to_excel(file_path, index=False)
    from fastapi.responses import FileResponse
    return FileResponse(file_path, filename="history.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
@app.get("/history/result/{history_id}", response_class=HTMLResponse)
def history_result(history_id: int, request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    history = db.query(models.SearchHistory).filter(models.SearchHistory.id == history_id, models.SearchHistory.user_id == current_user.id).first()
    if not history:
        raise HTTPException(status_code=404, detail="История не найдена")
    # Re-run the search with the stored query and city
    results = search_resumes(history.query, model, resumes_data, embeddings)
    return templates.TemplateResponse("search.html", {"request": request, "results": results, "history_query": history.query, "history_city": history.city})

@app.post("/candidates/{candidate_id}/update_details")
def update_candidate_details(
    candidate_id: int,
    name: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Update user activity when they modify candidates
    update_user_activity(db, current_user)
    
    candidate = db.query(models.Candidate).filter(
        models.Candidate.id == candidate_id,
        models.Candidate.user_id == current_user.id
    ).first()
    
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    
    candidate.name = name
    candidate.email = email or ""
    candidate.phone = phone or ""
    
    db.commit()
    
    log_user_action(
        db, current_user.id, "update_candidate_details",
        f"Details for candidate {candidate_id} updated."
    )
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)
