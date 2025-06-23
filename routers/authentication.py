from fastapi import APIRouter, Request, Form, Depends, status, HTTPException, Cookie
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import jwt, JWTError

from database import get_db
import models
from auth import verify_password, create_access_token, SECRET_KEY, ALGORITHM, hash_password
from shared import templates
from utils import log_user_action, update_user_activity

router = APIRouter()

@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": {}, "error": "Неверные данные"}, status_code=status.HTTP_401_UNAUTHORIZED)

    update_user_activity(db, user)
    log_user_action(db, user.id, "login", f"User {user.username} logged in")

    token = create_access_token({"sub": user.username})
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=token, httponly=True)
    return response

@router.get("/logout")
def logout(access_token: str = Cookie(None), db: Session = Depends(get_db)):
    if access_token:
        try:
            payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username:
                user = db.query(models.User).filter(models.User.username == username).first()
                if user:
                    user.last_activity = datetime.utcnow() - timedelta(minutes=20)
                    db.commit()
                    log_user_action(db, user.id, "logout", f"User {user.username} logged out")
        except JWTError:
            pass
    
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

@router.get("/register", response_class=HTMLResponse)
def show_register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@router.post("/register")
def register(username: str = Form(...), password: str = Form(...), role: str = Form(...), db: Session = Depends(get_db)):
    existing = db.query(models.User).filter(models.User.username == username).first()
    if existing:
        return templates.TemplateResponse("register.html", {"request": {}, "error": "Пользователь уже существует"}, status_code=status.HTTP_400_BAD_REQUEST)

    user = models.User(username=username, hashed_password=hash_password(password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    log_user_action(db, user.id, "register", f"New user {username} registered with role {role}")
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND) 