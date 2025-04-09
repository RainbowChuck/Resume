import json
# def read_large_json(file_path, chunk_size=100000):
#     """Читает JSON построчно, возвращает чанки"""
#     with open(file_path, 'r', encoding='utf-8') as f:
#         chunk = []
#         for line in f:
#             try:
#                 data = json.loads(line.strip())
#                 chunk.append(data)
#             except json.JSONDecodeError:
#                 continue
#
#             if len(chunk) >= chunk_size:
#                 yield chunk
#                 chunk = []
#
#         if chunk:
#             yield chunk

def read_large_json(file_path, chunk_size=100000):
    """Читает JSON построчно и возвращает чанки"""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk.append(data)
            except json.JSONDecodeError as e:
                print(f"Ошибка при декодировании строки JSON: {line}. Ошибка: {e}")
                continue

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        if chunk:
            yield chunk
def extract_resume_text(resume):
    parts = []

    # Базовые поля
    parts.append(resume.get("positionName", ""))
    parts.append(resume.get("education", ""))
    parts.append(str(resume.get("experience", "")))
    parts.append(resume.get("scheduleType", ""))
    parts.append(str(resume.get("salary", "")))
    parts.append(resume.get("retrainingCapability", ""))
    parts.append(resume.get("relocation", ""))
    parts.append(resume.get("businessTrip", ""))
    parts.append(resume.get("gender", ""))
    parts.append(resume.get("localityName", ""))
    parts.append(resume.get("abilympicsParticipation", ""))
    parts.append(resume.get("volunteersParticipation", ""))
    parts.append(resume.get("worldskillsInspectionStatus", ""))
    parts.append(resume.get("abilympicsInspectionStatus", ""))
    parts.append(resume.get("volunteersInspectionStatus", ""))
    parts.append(resume.get("narkInspectionStatus", ""))

    # Образование
    for edu in resume.get("educationList", []):
        parts.append(edu.get("instituteName", ""))
        parts.append(str(edu.get("graduateYear", "")))
        parts.append(edu.get("type", ""))

    # Профессии
    for prof in resume.get("professionList", []):
        parts.append(prof.get("codeProfessionalSphere", ""))

    # Языки
    for lang in resume.get("languageKnowledge", []):
        parts.append(lang.get("codeLanguage", ""))
        parts.append(lang.get("level", ""))

    # Страна
    for country in resume.get("country", []):
        parts.append(country.get("countryName", ""))

    # Вложенное
    inner = resume.get("innerInfo", {})
    parts.append(inner.get("rfCitizen", ""))
    parts.append(inner.get("visibility", ""))

    raw_text = " ".join([str(p).strip() for p in parts if p])
    print(f"Извлеченный текст: {raw_text}")  # Логирование
    return raw_text
