from datetime import datetime
from sqlalchemy.orm import Session
import models

def update_user_activity(db: Session, user: models.User):
    """Update user's last activity timestamp."""
    user.last_activity = datetime.utcnow()
    db.commit()

def log_user_action(db: Session, user_id: int, action_type: str, description: str):
    """Log a user action to the database."""
    action = models.UserAction(user_id=user_id, action_type=action_type, description=description)
    db.add(action)
    db.commit()
