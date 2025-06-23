from fastapi import APIRouter, Request, Form, Depends, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session

from database import get_db
import models
from dependencies import get_current_user
from shared import templates
from utils import log_user_action, update_user_activity

router = APIRouter(
    prefix="/candidates",
    tags=["candidates"],
    dependencies=[Depends(get_current_user)]
)

@router.get("", response_class=HTMLResponse)
def list_candidates(request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)
    candidates = db.query(models.Candidate).filter(models.Candidate.user_id == current_user.id).all()
    return templates.TemplateResponse("candidates.html", {"request": request, "candidates": candidates})

@router.post("/add")
def add_candidate(request: Request, positionName: str = Form(...), experience: str = Form(...), localityName: str = Form(...), salary: str = Form(...), hardSkills: str = Form(...), search_history_id: int = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    resume_dict = {
        "positionName": positionName,
        "experience": experience,
        "localityName": localityName,
        "salary": salary,
        "hardSkills": hardSkills.split(",") if hardSkills else []
    }
    candidate = models.Candidate(
        name=positionName or "-",
        email="",
        phone="",
        resume_text=str(resume_dict),
        status="new",
        user_id=current_user.id,
        search_history_id=search_history_id
    )
    db.add(candidate)
    db.commit()
    return RedirectResponse(url="/candidates", status_code=302)

@router.get("/{candidate_id}", response_class=HTMLResponse)
def view_candidate(candidate_id: int, request: Request, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)
    if current_user.role == "dean":
        candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id).first()
    else:
        candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id, models.Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    return templates.TemplateResponse("candidate_detail.html", {"request": request, "candidate": candidate, "current_user": current_user})

@router.post("/{candidate_id}/update")
def update_candidate_status(candidate_id: int, action: str = Form(...), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)
    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id, models.Candidate.user_id == current_user.id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Кандидат не найден")
    old_status = candidate.status
    if current_user.role == "hr":
        if action == "approve": candidate.status = "approved"
        elif action == "reject": candidate.status = "rejected"
    elif current_user.role == "manager":
        if action == "hire": candidate.status = "hired"
        elif action == "decline": candidate.status = "declined"
    db.commit()
    log_user_action(db, current_user.id, "status_change", f"Статус кандидата {candidate_id} изменён с {old_status} на {candidate.status}")
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@router.post("/{candidate_id}/update_test_results")
def update_test_results(candidate_id: int, test_results: str = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if current_user.role != "hr": raise HTTPException(status_code=403, detail="Недостаточно прав")
    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id, models.Candidate.user_id == current_user.id).first()
    if not candidate: raise HTTPException(status_code=404, detail="Кандидат не найден")
    if test_results and test_results.strip():
        candidate.test_results = (candidate.test_results + "\n\n" if candidate.test_results else "") + test_results
        db.commit()
        log_user_action(db, current_user.id, "update_test_results", f"Результаты тестирования кандидата {candidate_id} обновлены.")
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@router.post("/{candidate_id}/update_video_notes")
def update_video_notes(candidate_id: int, video_interview_notes: str = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    if current_user.role != "hr": raise HTTPException(status_code=403, detail="Недостаточно прав")
    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id, models.Candidate.user_id == current_user.id).first()
    if not candidate: raise HTTPException(status_code=404, detail="Кандидат не найден")
    if video_interview_notes and video_interview_notes.strip():
        candidate.video_interview_notes = (candidate.video_interview_notes + "\n\n" if candidate.video_interview_notes else "") + video_interview_notes
        db.commit()
        log_user_action(db, current_user.id, "update_video_notes", f"Заметки по видеоинтервью кандидата {candidate_id} обновлены.")
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302)

@router.post("/{candidate_id}/update_details")
def update_candidate_details(candidate_id: int, name: str = Form(...), email: str = Form(None), phone: str = Form(None), db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    update_user_activity(db, current_user)
    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id, models.Candidate.user_id == current_user.id).first()
    if not candidate: raise HTTPException(status_code=404, detail="Кандидат не найден")
    candidate.name = name
    candidate.email = email or ""
    candidate.phone = phone or ""
    db.commit()
    log_user_action(db, current_user.id, "update_candidate_details", f"Details for candidate {candidate_id} updated.")
    return RedirectResponse(url=f"/candidates/{candidate_id}", status_code=302) 