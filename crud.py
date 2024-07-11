from sqlalchemy.orm import Session
from models import HandLandmark

def get_hand_landmarks(db: Session, word_no: int):
    return db.query(HandLandmark).filter(HandLandmark.wordNo == word_no).all()
