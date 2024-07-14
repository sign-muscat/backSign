# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import cv2
# import mediapipe as mp
# import numpy as np
# import io
# from PIL import Image

# app = FastAPI()

# # 손 찾기 관련 기능 불러오기
# mp_hands = mp.solutions.hands
# # 손 그려주는 기능 불러오기
# mp_drawing = mp.solutions.drawing_utils

# # 손 찾기 세부 설정
# hands = mp_hands.Hands(
#     max_num_hands=2,  # 탐지할 최대 손의 객수
#     min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
#     min_tracking_confidence=0.5  # 표시할 관찰의 최소 정확도
# )

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents))
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     # 손 탐지
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)

#     # 찾은 손 표시하기 및 랜드마크 좌표 추출
#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # 랜드마크 좌표 추출
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.append([lm.x, lm.y, lm.z])
#             landmarks = np.array(landmarks).flatten()

#             # 여기서 landmarks는 손의 각 관절의 3D 좌표를 포함하는 벡터입니다.
#             return JSONResponse(content={"landmarks": landmarks.tolist()})
    
#     return JSONResponse(content={"error": "No hand landmarks found"}, status_code=404)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# #springboot와 fastapi 연동
# from fastapi import FastAPI, File, UploadFile, HTTPException, Form
# from fastapi.responses import JSONResponse
# import cv2
# import mediapipe as mp
# import numpy as np
# import io
# from PIL import Image
# import mysql.connector
# import json
# from scipy.spatial.distance import cosine

# app = FastAPI()

# # 손 찾기 관련 기능 불러오기
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# hands = mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # MySQL 데이터베이스 설정
# db_config = {
#     'user': 'sign',
#     'password': 'sign',
#     'host': 'localhost',
#     'database': 'sign_db',
#     'port': '3306'
# }

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...), word_no: int = Form(...)):
#     contents = await file.read()
#     image = Image.open(io.BytesIO(contents))
#     img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#     # 손 탐지
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result = hands.process(img_rgb)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.append([lm.x, lm.y, lm.z])
#             landmarks = np.array(landmarks).flatten()

#             # MySQL 데이터베이스에 연결하고 단어 번호를 기준으로 정보 조회
#             try:
#                 conn = mysql.connector.connect(**db_config)
#                 cursor = conn.cursor()
#                 cursor.execute("SELECT vector FROM handlandmark WHERE wordNo = %s", (word_no,))
#                 rows = cursor.fetchall()
#             except mysql.connector.Error as err:
#                 return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)
#             finally:
#                 conn.close()

#             if not rows:
#                 raise HTTPException(status_code=404, detail="Hand landmarks not found for the word")

#             # 첫 번째 행의 벡터 값을 가져옴
#             db_landmark_array = np.array(json.loads(rows[0][0]))

#             similarity = calculate_cosine_similarity(landmarks, db_landmark_array)
#             threshold = 0.5  # 임계값 설정
#             is_correct = similarity > threshold

#             return JSONResponse(content={"similarity": similarity, "is_correct": is_correct})

#     return JSONResponse(content={"error": "No hand landmarks found"}, status_code=404)

# def calculate_cosine_similarity(landmarks1, landmarks2):
#     return 1 - cosine(landmarks1, landmarks2)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)





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
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_config = {
    'user': 'sign',
    'password': 'sign',
    'host': 'localhost',
    'database': 'sign_db',
    'port': '3306'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 사용자의 게임 결과를 데이터베이스에 저장하고 해당 사용자의 랭킹 정보를 조회하여 반환
from pydantic import BaseModel
class ResultRequest(BaseModel):
    nickname: str
    levels: str
    today: datetime
    wordDes: int
    word_name: str
    isCorrect: bool
    

@app.post("/result/")
async def save_result(request: ResultRequest):
    data = request.dict()
    nickname = data.get("nickname")
    levels = data.get("levels")
    today = data.get("today")
    wordDes = data.get("wordDes")
    word_name = data.get("word_name")
    isCorrect = data.get("isCorrect")

    if not nickname or levels is None or today is None or wordDes is None or word_name is None or isCorrect is None:
        raise HTTPException(status_code=400, detail="Invalid data")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # 여기에 로깅 코드 추가
        logger.info(f"Received wordDes: {wordDes}")
        logger.info(f"Type of wordDes: {type(wordDes)}")

        # Check if wordDes exists in words table
        cursor.execute("SELECT wordNo FROM words WHERE wordDes = %s", (wordDes,))
        result = cursor.fetchone()
        logger.info(f"Query result for wordDes {wordDes}: {result}")

        if not result:
            raise HTTPException(status_code=400, detail="Invalid wordDes: does not exist in words table")
        word_no = result[0]
        
        # 나머지 코드는 그대로 유지...

    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)
    
    return JSONResponse(content={"message": "Result saved successfully"})

# 전체 랭킹 조회
@app.get("/ranking/")
async def get_ranking():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            WITH user_stats AS (
                SELECT 
                    m.nickName, 
                    SUM(m.isCorrect) / COUNT(*) AS correct_ratio,
                    MAX(r.today) AS last_played_date
                FROM mypage m
                LEFT JOIN ranks r ON m.myNo = r.myNo
                GROUP BY m.nickName
            ),
            user_ranks AS (
                SELECT 
                    nickName, 
                    correct_ratio, 
                    last_played_date,
                    RANK() OVER (ORDER BY correct_ratio DESC) as `rank`
                FROM user_stats
            )
            SELECT 
                ur.nickName, 
                ur.correct_ratio, 
                ur.last_played_date, 
                g.levels AS last_played_level,
                ur.`rank`
            FROM user_ranks ur
            LEFT JOIN ranks r ON r.myNo = (
                SELECT myNo 
                FROM ranks 
                WHERE myNo IN (SELECT myNo FROM mypage WHERE nickName = ur.nickName) 
                ORDER BY today DESC 
                LIMIT 1
            )
            LEFT JOIN grade g ON r.gradeNo = g.gradeNo
            ORDER BY ur.`rank`
        """
        cursor.execute(query)
        rankings = cursor.fetchall()
        cursor.close()
        conn.close()

        ranked_results = [
            {
                "nickName": rank[0], 
                "correct_ratio": float(rank[1]), 
                "today": rank[2],
                "levels": rank[3] if rank[3] else None,
                "rank": rank[4]
            } 
            for rank in rankings
        ]

        return {"rankings": ranked_results}
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    
# 주어진 word_no에 대한 단어 이미지 URL을 데이터베이스에서 조회하여 반환
@app.get("/word_step/")
async def get_word_step(word_des: int, pose_num: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Retrieve the images for the given wordDes
        cursor.execute("SELECT wordImg FROM words WHERE wordDes = %s", (word_des,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows or pose_num < 1 or pose_num > len(rows):
            raise HTTPException(status_code=404, detail="Step image not found")

        # Select the image based on pose_num
        selected_image = rows[pose_num - 1][0]

        return {"imageUrl": selected_image}
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")


# word_no에 해당하는 단어의 상세 정보를 데이터베이스에서 조회하여 반환하는 엔드포인트
@app.get("/word/")
async def get_word(word_no: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            SELECT wordNo, wordDes, wordImg
            FROM words
            WHERE wordNo = %s
        """
        cursor.execute(query, (word_no,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Word not found")

        return {
            "wordNo": row[0],
            "wordDes": row[1],
            "wordImg": row[2]
        }
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")
    
# 사용자의 게임 결과와 날짜, 난이도, 랭킹을 조회하는 기능
@app.get("/user_ranking/")
async def get_user_ranking(nickname: str):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
        WITH user_rankings AS (
            SELECT 
                nickName,
                SUM(isCorrect) / COUNT(*) AS correct_ratio,
                RANK() OVER (ORDER BY SUM(isCorrect) / COUNT(*) DESC) as overall_rank
            FROM mypage
            GROUP BY nickName
        ),
        last_game AS (
            SELECT 
                m.nickName,
                r.today as last_played_date,
                g.levels as last_played_level
            FROM mypage m
            JOIN ranks r ON m.myNo = r.myNo
            JOIN grade g ON r.gradeNo = g.gradeNo
            WHERE m.nickName = %s
            ORDER BY r.today DESC
            LIMIT 1
        )
        SELECT 
            ur.correct_ratio,
            lg.last_played_date,
            lg.last_played_level,
            ur.overall_rank
        FROM user_rankings ur
        JOIN last_game lg ON ur.nickName = lg.nickName
        WHERE ur.nickName = %s
        """
        
        cursor.execute(query, (nickname, nickname))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="User not found or no game results")

        user_ranking = {
            "correct_ratio": float(result[0]),
            "today": result[1],
            "levels": result[2],
            "rank": result[3]
        }

        return user_ranking
    
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    

class QuestionResult(BaseModel):
    word_name: str
    is_correct: bool

class GameResult(BaseModel):
    correct_percentage: str
    questions: List[QuestionResult]


@app.get("/user_game_result/{nickname}", response_model=GameResult)
async def get_user_game_result(nickname: str):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            SUBSTRING_INDEX(w.wordName, '_', 1) as word_name,
            MIN(m.isCorrect) as is_correct
        FROM mypage m
        JOIN words w ON m.wordNo = w.wordNo
        WHERE m.nickName = %s
        GROUP BY SUBSTRING_INDEX(w.wordName, '_', 1)
        ORDER BY MIN(m.myNo)
        """
        
        cursor.execute(query, (nickname,))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()

        if not results:
            raise HTTPException(status_code=404, detail="User not found or no game results")

        # Calculate correct and total counts
        correct_count = sum(result['is_correct'] for result in results)
        total_count = len(results)
        
        # Calculate correct percentage as a string
        correct_percentage = f"{correct_count}/{total_count}"
        
        questions = [QuestionResult(**result) for result in results]

        return GameResult(
            correct_percentage=correct_percentage,
            questions=questions
        )
    
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)














