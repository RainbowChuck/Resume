from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session

from database import get_db
import models
from schemas import UserOut, UserCreate
from auth import hash_password
from dependencies import get_current_user
from shared import templates
from utils import log_user_action

router = APIRouter(
    prefix="/users",
    tags=["users"],
    dependencies=[Depends(get_current_user)]
)

def is_admin(user: models.User = Depends(get_current_user)):
    if user.role != 'admin':
        raise HTTPException(status_code=403, detail="Нет прав для этого действия")

@router.post("", response_model=UserOut, dependencies=[Depends(is_admin)])
def create_user(user_in: UserCreate, db: Session = Depends(get_db)):
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

@router.get("/{user_id}/edit", response_class=HTMLResponse, dependencies=[Depends(is_admin)])
def edit_user_form(user_id: int, request: Request, db: Session = Depends(get_db)):
    user_to_edit = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_edit:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return templates.TemplateResponse("edit_user.html", {"request": request, "user": user_to_edit, "success": request.query_params.get("success")})

@router.post("/{user_id}/edit", dependencies=[Depends(is_admin)])
def edit_user(user_id: int, username: str = Form(...), role: str = Form(...), password: str = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    user_to_edit = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_edit:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    user_to_edit.username = username
    user_to_edit.role = role
    if password:
        user_to_edit.hashed_password = hash_password(password)
    db.commit()
    log_user_action(db, current_user.id, "edit_user", f"User {username} (ID: {user_id}) updated.")
    return RedirectResponse(url=f"/users/{user_id}/edit?success=1", status_code=302)

@router.post("/{user_id}/delete", dependencies=[Depends(is_admin)])
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    user_to_delete = db.query(models.User).filter(models.User.id == user_id).first()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    db.delete(user_to_delete)
    db.commit()
    log_user_action(db, current_user.id, "delete_user", f"User ID {user_id} deleted.")
    return RedirectResponse(url="/admin/dashboard?deleted=1", status_code=302) 