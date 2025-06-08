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
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
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

from fastapi import FastAPI, Request, Form, Depends, status, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError

from auth import verify_password, create_access_token, SECRET_KEY, ALGORITHM, hash_password
from database import get_db
from models import User, SearchHistory
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

    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
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

    response = RedirectResponse(url="/login", status_code=302)
    return response


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


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except JWTError:
        return RedirectResponse("/login")

    return templates.TemplateResponse("dashboard.html", {"request": request, "username": username})

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
