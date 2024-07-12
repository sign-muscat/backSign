from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import io
from PIL import Image
import mysql.connector
import json
from scipy.spatial.distance import cosine
import logging
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL 연결 설정
db_config = {
    "host": "localhost",
    "user": "sign",
    "password": "sign",
    "database": "sign_db"
}

# wordDes와 poseStep을 받아서 words 테이블에서 wordDes가 같은 데이터들을 반환하는 엔드포인트
@app.post("/gameStart/")
async def get_words(wordDes: int = Form(...), poseStep: int = Form(...)):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
            SELECT wordNo, wordDes, wordImg
            FROM words
            WHERE wordDes = %s
        """
        
        cursor.execute(query, (wordDes,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="Words not found for the given wordDes")

        words = []
        for row in rows:
            words.append({
                "wordNo": row[0],
                "wordDes": row[1],
                "wordImg": row[2]
            })

        return {"words": words}
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
