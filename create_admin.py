# create_admin.py
from database import SessionLocal
from models import User
from auth import hash_password

db = SessionLocal()

user = User(
    username="dean1",
    hashed_password=hash_password("1234"),
    role="dean"
)

db.add(user)
db.commit()
print("Пользователь добавлен.")
