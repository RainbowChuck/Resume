import json
import re

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^а-яА-Яa-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_resume_text(resume):
    parts = [
        resume.get("positionName", ""),
        resume.get("education", ""),
        str(resume.get("experience", "")),
        resume.get("scheduleType", ""),
        str(resume.get("salary", "")),
        resume.get("retrainingCapability", ""),
        resume.get("relocation", ""),
        resume.get("businessTrip", ""),
        resume.get("gender", ""),
        resume.get("localityName", "")
    ]

    for edu in resume.get("educationList", []):
        parts.append(edu.get("instituteName", ""))
        parts.append(str(edu.get("graduateYear", "")))

    for prof in resume.get("professionList", []):
        parts.append(prof.get("codeProfessionalSphere", ""))

    for lang in resume.get("languageKnowledge", []):
        parts.append(lang.get("codeLanguage", ""))
        parts.append(lang.get("level", ""))

    for country in resume.get("country", []):
        parts.append(country.get("countryName", ""))

    raw = " ".join(str(p).strip() for p in parts if p)
    return clean_text(raw)
