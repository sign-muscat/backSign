from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
import random
from typing import List

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL 연결 설정
mydb = mysql.connector.connect(
    host="localhost",
    user="sign",
    password="sign",
    database="sign_db"
)

# MySQL 데이터베이스 커서 생성
mycursor = mydb.cursor(dictionary=True)

class WordResponse(BaseModel):
    wordDes: int
    wordName: str

@app.get("/get-words", response_model=List[WordResponse])
def get_words(difficulty: str = Query(..., regex="^(쉬움|보통|어려움)$"), count: int = 5):
    level_map = {
        "쉬움": "하",
        "보통": "중",
        "어려움": "상"
    }
    level = level_map.get(difficulty)

    if not level:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    # wordDes의 고유한 갯수를 제한하여 가져오는 쿼리
    sql = """
        SELECT DISTINCT w.wordDes
        FROM words w
        JOIN grade g ON w.wordNo = g.wordNo
        WHERE g.levels = %s
        LIMIT %s
    """
    mycursor.execute(sql, (level, count))
    word_des_list = mycursor.fetchall()

    if not word_des_list:
        raise HTTPException(status_code=404, detail="No words found for the given difficulty level")

    word_des_values = [word['wordDes'] for word in word_des_list]
    
    # 제한된 wordDes의 각 항목에 대해 wordName과 actions를 가져오는 쿼리
    placeholders = ', '.join(['%s'] * len(word_des_values))
    sql = f"""
        SELECT w.wordDes, w.wordName
        FROM words w
        JOIN handlandmark h ON w.wordNo = h.wordNo
        WHERE w.wordDes IN ({placeholders})
        GROUP BY w.wordDes, w.wordName
    """
    mycursor.execute(sql, word_des_values)
    words = mycursor.fetchall()

    # 가져온 단어 목록을 콘솔에 출력
    print("Fetched words from DB:", words)

    return words
