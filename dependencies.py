from fastapi import Depends, HTTPException, status, Cookie
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from database import get_db
import models
from auth import SECRET_KEY, ALGORITHM

def get_current_user(access_token: str = Cookie(None), db: Session = Depends(get_db)) -> models.User:
    if not access_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Не авторизован (нет cookie)")

    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Некорректный payload токена")
    except JWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Некорректный токен: {str(e)}")

    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Пользователь не найден")
    
    return user 