from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    role = Column(String, default="hr")  # hr, admin, dean
    last_activity = Column(DateTime, default=datetime.utcnow)

    history = relationship("SearchHistory", back_populates="user")
    candidates = relationship("Candidate", back_populates="user")
    actions = relationship("UserAction", back_populates="user")

class SearchHistory(Base):
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(String)
    city = Column(String)
    results = Column(Integer)
    date = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="history")

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    resume_text = Column(Text)
    status = Column(String, default="new")
    notes = Column(Text)
    test_results = Column(Text, nullable=True)
    video_interview_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="candidates")
    status_update_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    search_history_id = Column(Integer, ForeignKey("search_history.id"), nullable=True)
    search_history = relationship("SearchHistory")

class UserAction(Base):
    __tablename__ = "user_actions"
    
    id = Column(Integer, primary_key=True, index=True)
    action_type = Column(String)  # login, logout, create_candidate, update_candidate, etc.
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="actions")

class SystemSettings(Base):
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    system_name = Column(String, default="Resume Management System")
    items_per_page = Column(Integer, default=20)
    search_engine = Column(String, default="default")
    enable_semantic_search = Column(Boolean, default=True)
    notify_new_candidates = Column(Boolean, default=True)
    notify_status_changes = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
