import random
import string
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mysql.connector
from typing import List

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
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
    poseCount: int

class NickNameResponse(BaseModel):
    myNo: int
    nickName: str

def generate_random_nickname(length: int = 20) -> str:
    characters = string.ascii_letters + string.digits + string.punctuation
    random_nickname = ''.join(random.choice(characters) for i in range(length))
    return random_nickname

def clean_word_name(word_name: str) -> str:
    # '_숫자' 부분 제거
    return '_'.join(word_name.split('_')[:-1])

# 난이도별 랜덤 문제 출제
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
        SELECT w.wordDes, COUNT(w.wordNo) as poseCount, MIN(w.wordName) as wordName
        FROM words w
        JOIN grade g ON w.wordNo = g.wordNo
        WHERE g.levels = %s
        GROUP BY w.wordDes
    """
    mycursor.execute(sql, (level,))
    word_des_list = mycursor.fetchall()

    if not word_des_list:
        raise HTTPException(status_code=404, detail="No words found for the given difficulty level")

    # wordDes 목록을 무작위로 섞음
    random.shuffle(word_des_list)

    # 제한된 수만큼 선택
    selected_words = word_des_list[:count]

    # wordName에서 '_숫자' 부분 제거 및 단어 목록 수정
    for word in selected_words:
        word['wordName'] = clean_word_name(word['wordName'])

    # 가져온 단어 목록을 콘솔에 출력
    print("Fetched words from DB:", selected_words)

    return selected_words

# 랜덤 닉네임 생성 및 저장 엔드포인트 추가
@app.post("/generate-nickname", response_model=NickNameResponse)
def generate_nickname():
    # 랜덤 닉네임 생성
    random_nickname = generate_random_nickname()

    # mypage 테이블에 데이터 삽입
    sql = "INSERT INTO mypage (nickName) VALUES (%s)"
    val = (random_nickname,)
    try:
        mycursor.execute(sql, val)
        mydb.commit()
        myNo = mycursor.lastrowid
        print(f"Inserted nickName: {random_nickname} with myNo: {myNo}")
        return NickNameResponse(myNo=myNo, nickName=random_nickname)
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=str(err))
