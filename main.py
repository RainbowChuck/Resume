# # main.py
# from fastapi import FastAPI, Depends, HTTPException, Request
# from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
# from sqlalchemy.orm import Session
# from starlette.templating import Jinja2Templates
#
# from database import get_db
# from models import User
# from schemas import UserCreate, UserOut, Token
# from auth import hash_password, verify_password, create_access_token, SECRET_KEY, ALGORITHM
# from jose import JWTError, jwt
#
# app = FastAPI()
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
#
# # Регистрация
# @app.post("/users/", response_model=UserOut)
# def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
#     existing = db.query(User).filter(User.username == user_in.username).first()
#     if existing:
#         raise HTTPException(status_code=400, detail="Username already taken")
#     user = User(
#         username=user_in.username,
#         hashed_password=hash_password(user_in.password)
#     )
#     db.add(user)
#     db.commit()
#     db.refresh(user)
#     return user
#
# # Получение токена
# @app.post("/token", response_model=Token)
# def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(),
#                            db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect credentials",
#                             headers={"WWW-Authenticate": "Bearer"})
#     token = create_access_token({"sub": user.username})
#     return {"access_token": token, "token_type": "bearer"}
#
# # Зависимость: получить текущего пользователя
# def get_current_user(token: str = Depends(oauth2_scheme),
#                      db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=401,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
#     user = db.query(User).filter(User.username == username).first()
#     if user is None:
#         raise credentials_exception
#     return user
#
# templates = Jinja2Templates(directory="templates")
# # Пример защищённого роута
# @app.get("/users/me", response_model=UserOut)
# def read_users_me(current_user: User = Depends(get_current_user)):
#     return current_user
# @app.get("/")
# def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/")
# def read_root(request: Request):
#     return templates.TemplateResponse("form.html", {"request": request}) #index.html

from fastapi import FastAPI, Request, Form, Depends, status, HTTPException, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from datetime import datetime, timedelta
from sqlalchemy import func
from collections import defaultdict
from auth import verify_password, create_access_token, SECRET_KEY, ALGORITHM, hash_password
from database import get_db
from models import User, SearchHistory, Candidate, , UserAction, SystemSettings
from schemas import UserOut, UserCreate
app = FastAPI()
templates = Jinja2Templates(directory="templates")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


@app.get("/", response_class=HTMLResponse)
def home():
    return RedirectResponse("/login")


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("access_token")
    return response

@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": {}, "error": "Неверные данные"}, status_code=401)

    token = create_access_token({"sub": user.username})
    log_user_action(db, user.id, "login", f"User {user.username} logged in")

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
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        return templates.TemplateResponse("register.html", {
            "request": {},
            "error": "Пользователь уже существует"
        }, status_code=400)

    user = User(
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



def get_current_user(access_token: str = Cookie(None), db: Session = Depends(get_db)) -> User:
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated (no cookie)")

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user




@app.get("/history", response_class=HTMLResponse)
def get_history(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    city: str = "",
    start: str = "",
    end: str = ""
):
    query = db.query(SearchHistory).filter(SearchHistory.user_id == current_user.id)

    if city:
        query = query.filter(SearchHistory.city.ilike(f"%{city}%"))
    if start:
        query = query.filter(SearchHistory.date >= start)
    if end:
        query = query.filter(SearchHistory.date <= end)

    results = query.order_by(SearchHistory.date.desc()).all()

    return templates.TemplateResponse("history.html", {"request": request, "history": results})


    # Фильтрация
    filtered = []
    for row in dummy_data:
        if city and city not in row["city"]:
            continue
        if start and row["date"] < start:
            continue
        if end and row["date"] > end:
            continue
        filtered.append(row)

    return templates.TemplateResponse("history.html", {"request": request, "history": filtered})
@app.get("/candidates", response_class=HTMLResponse)
def list_candidates(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    candidates = db.query(Candidate).filter(Candidate.user_id == current_user.id).all()
    return templates.TemplateResponse("candidates.html", {
        "request": request,
        "candidates": candidates
    })


# Просмотр кандидата
@app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
def view_candidate(candidate_id: int, request: Request,
                   db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id,
                                           Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    return templates.TemplateResponse("candidate_detail.html", {
        "request": request,
        "candidate": candidate
    })


# Обновление статуса кандидата
@app.post("/candidates/{candidate_id}/update")
def update_candidate_status(
        candidate_id: int,
        action: str = Form(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    candidate = db.query(Candidate).filter(
        Candidate.id == candidate_id,
        Candidate.user_id == current_user.id
    ).first()

    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")

    old_status = candidate.status
    if current_user.role == "hr":
        if action == "approve":
            candidate.status = "approved"
        elif action == "reject":
            candidate.status = "rejected"
        elif action == "screen":
            candidate.status = "screened"
    elif current_user.role == "dean" and action == "final_approve":
        if candidate.status == "approved":
            candidate.status = "final_approved"
        else:
            raise HTTPException(status_code=400, detail="Кандидат должен быть одобрен HR")

    db.commit()
    log_user_action(
        db,
        current_user.id,
        "update_candidate_status",
        f"Updated candidate {candidate.name} status from {old_status} to {candidate.status}"
    )
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)


@app.post("/candidates/{candidate_id}/update_test_results")
def update_test_results(
        candidate_id: int,
        test_results: str = Form(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if current_user.role != "hr":
        raise HTTPException(status_code=403, detail="Только HR может обновлять результаты тестов")

    candidate = db.query(Candidate).filter(
        Candidate.id == candidate_id,
        Candidate.user_id == current_user.id
    ).first()

    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")

    candidate.test_results = test_results
    db.commit()
    log_user_action(
        db,
        current_user.id,
        "update_test_results",
        f"Updated test results for candidate {candidate.name}"
    )
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)


@app.post("/candidates/{candidate_id}/update_video_notes")
def update_video_notes(
        candidate_id: int,
        video_interview_notes: str = Form(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if current_user.role != "hr":
        raise HTTPException(status_code=403, detail="Только HR может обновлять заметки по видео-интервью")

    candidate = db.query(Candidate).filter(
        Candidate.id == candidate_id,
        Candidate.user_id == current_user.id
    ).first()

    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")

    candidate.video_interview_notes = video_interview_notes
    db.commit()
    log_user_action(
        db,
        current_user.id,
        "update_video_notes",
        f"Updated video interview notes for candidate {candidate.name}"
    )
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@app.get("/search", response_class=HTMLResponse)
def search_page(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    if current_user.role not in ["hr", "admin"]:
        raise HTTPException(status_code=403, detail="Access denied")
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, current_user: User = Depends(get_current_user)):
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})


@app.post("/users/", response_model=UserOut)
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(
        username=user_in.username,
        hashed_password=hash_password(user_in.password),
        role="hr"  # можно изменить на "dean" или "admin"
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
