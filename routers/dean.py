from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from database import get_db
import models
from dependencies import get_current_user
from shared import templates
from utils import update_user_activity

router = APIRouter(
    prefix="/dean",
    tags=["dean"],
    dependencies=[Depends(get_current_user)]
)

def is_dean(user: models.User = Depends(get_current_user)):
    if user.role != 'dean':
        raise HTTPException(status_code=403, detail="Нет прав для этого действия")
    return user

@router.get("/candidates", response_class=HTMLResponse, dependencies=[Depends(is_dean)])
def dean_candidates(request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)
    candidates = db.query(models.Candidate).filter(models.Candidate.status == "approved").all()
    return templates.TemplateResponse("dean_candidates.html", {"request": request, "candidates": candidates, "current_user": current_user}) 