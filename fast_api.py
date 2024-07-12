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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
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

def calculate_cosine_similarity(landmarks1, landmarks2):
    return 1 - cosine(landmarks1, landmarks2)

# 사용자가 업로드한 이미지를 처리하고, 이미지에서 손의 랜드마크를 추출하여 데이터베이스에 저장된 손 랜드마크와 비교하여 유사성을 계산
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), word_no: int = Form(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Error reading the image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).flatten()

            try:
                conn = mysql.connector.connect(**db_config)
                cursor = conn.cursor()
                cursor.execute("SELECT vector FROM handlandmark WHERE wordNo = %s", (word_no,))
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
            except mysql.connector.Error as err:
                logger.error(f"Database error: {err}")
                return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)

            if not rows:
                logger.info(f"Hand landmarks not found for the word {word_no}")
                raise HTTPException(status_code=404, detail="Hand landmarks not found for the word")

            db_landmark_array = np.array(json.loads(rows[0][0]))

            similarity = calculate_cosine_similarity(landmarks, db_landmark_array)
            threshold = 0.5
            is_correct = similarity > threshold

            return JSONResponse(content={"similarity": similarity, "is_correct": is_correct})

    logger.info("No hand landmarks found")
    return JSONResponse(content={"error": "No hand landmarks found"}, status_code=404)

# 사용자의 게임 결과를 데이터베이스에 저장하고 해당 사용자의 랭킹 정보를 조회하여 반환
# 결과 저장 엔드포인트
@app.post("/result/")
async def save_result(request: Request):
    data = await request.json()
    nickname = data.get("nickname")
    correct_answers = data.get("correct_answers")
    word_no = data.get("word_no")

    if not nickname or correct_answers is None or word_no is None:
        raise HTTPException(status_code=400, detail="Invalid data")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO mypage (wordNo, isCorrect, nickName) VALUES (%s, %s, %s)",
            (word_no, correct_answers, nickname)
        )
        my_no = cursor.lastrowid

        cursor.execute(
            "INSERT INTO ranks (gradeNo, myNo, today) VALUES (%s, %s, %s)",
            (1, my_no, datetime.now())
        )
        conn.commit()

        cursor.execute("SELECT wordName, isCorrect FROM mypage m JOIN words w ON m.wordNo = w.wordNo WHERE nickName=%s", (nickname,))
        ranks = cursor.fetchall()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)

    return JSONResponse(content={"message": "Result saved successfully", "ranks": ranks})

# 랭킹 조회 엔드포인트
@app.get("/ranking/")
async def get_ranking():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            SELECT nickName, SUM(isCorrect) / COUNT(*) AS correct_ratio
            FROM mypage
            GROUP BY nickName
            ORDER BY correct_ratio DESC
        """
        cursor.execute(query)
        rankings = cursor.fetchall()
        cursor.close()
        conn.close()

        ranked_results = [{"nickName": rank[0], "correct_ratio": rank[1], "rank": index + 1} for index, rank in enumerate(rankings)]

        return {"rankings": ranked_results}
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")

#result를 테스트 하기 위한 endpoint
@app.get("/test_result/")
async def test_result(nickname: str, correct_answers: int, word_no: int):
    if not nickname or correct_answers is None or word_no is None:
        raise HTTPException(status_code=400, detail="Invalid data")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO mypage (wordNo, isCorrect, nickName) VALUES (%s, %s, %s)",
            (word_no, correct_answers, nickname)
        )
        my_no = cursor.lastrowid

        cursor.execute(
            "INSERT INTO ranks (gradeNo, myNo, today) VALUES (%s, %s, %s)",
            (1, my_no, datetime.now())
        )
        conn.commit()

        cursor.execute("SELECT wordName, isCorrect FROM mypage m JOIN words w ON m.wordNo = w.wordNo WHERE nickName=%s", (nickname,))
        ranks = cursor.fetchall()
        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)

    return JSONResponse(content={"message": "Result saved successfully", "ranks": ranks})

# 주어진 word_no에 대한 단어 이미지 URL을 데이터베이스에서 조회하여 반환
@app.get("/word_step/")
async def get_word_step(word_no: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT wordImg FROM words WHERE wordNo = %s", (word_no,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Step image not found")

        return {"imageUrl": row[0]}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)















