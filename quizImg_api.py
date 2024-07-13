from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector
import logging

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
        
        # wordDes가 일치하는 데이터 조회
        query = """
            SELECT wordNo, wordDes, wordName, wordImg
            FROM words
            WHERE wordDes = %s
        """
        
        cursor.execute(query, (wordDes,))
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Words not found for the given wordDes")

        # poseStep 값과 wordName의 _ 뒤의 숫자가 일치하는 데이터 필터링
        for row in rows:
            wordNo, wordDes, wordName, wordImg = row
            # wordName에서 _ 뒤의 숫자를 추출하여 poseStep과 비교
            try:
                if int(wordName.split('_')[-1]) == poseStep:
                    cursor.close()
                    conn.close()
                    return {"wordNo": wordNo, "wordDes": wordDes, "wordImg": wordImg}
            except ValueError:
                continue  # 숫자가 아닌 경우 넘어가기

        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="No matching word found for the given poseStep")
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
