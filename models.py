from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import relationship
from database import Base

class Word(Base):
    __tablename__ = "words"

    wordNo = Column(Integer, primary_key=True, index=True, autoincrement=True)
    wordName = Column(String(10), nullable=True)

    handlandmarks = relationship("HandLandmark", back_populates="word")

class HandLandmark(Base):
    __tablename__ = "handlandmark"

    handNo = Column(Integer, primary_key=True, index=True, autoincrement=True)
    wordNo = Column(Integer, ForeignKey('words.wordNo'), nullable=False)
    x = Column(Float, nullable=True)
    y = Column(Float, nullable=True)
    z = Column(Float, nullable=True)

    word = relationship("Word", back_populates="handlandmarks")
