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

class VideoLinkResponse(BaseModel):
    wordNo: int
    wordDes: int
    wordName: str
    videoLink: str

@app.get("/get-video-link", response_model=VideoLinkResponse)
def get_video_link(wordDes: int = Query(...)):
    # 특정 wordDes에 대한 wordName 및 videoLink 가져오는 쿼리
    sql = """
        SELECT wordNo, wordDes, wordName, videoLink
        FROM words
        WHERE wordDes = %s
        ORDER BY wordNo
        LIMIT 1
    """
    mycursor.execute(sql, (wordDes,))
    word = mycursor.fetchone()

    if not word:
        raise HTTPException(status_code=404, detail="No video links found for the given wordDes")

    # 가져온 단어를 콘솔에 출력
    print("Fetched video link from DB:", word)

    return word
