# init_db.py
from database import engine
from models import Base

Base.metadata.create_all(bind=engine)
print("База данных создана.")
