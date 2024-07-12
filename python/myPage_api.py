import json
import numpy as np
import mediapipe as mp
import cv2
import io
from PIL import Image
from scipy.spatial.distance import cosine
import mysql.connector
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 도메인 추가
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# Mediapipe 사용하기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# MySQL 연결 설정
mydb = mysql.connector.connect(
    host="localhost",
    user="sign",
    password="sign",
    database="sign_db"
)

# MySQL 데이터베이스 커서 생성
mycursor = mydb.cursor()

# 손의 랜드마크를 벡터화하여 반환하는 함수
def get_hand_landmarks_vector(hand_landmarks):
    landmarks_vector = []
    for lm in hand_landmarks.landmark:
        landmarks_vector.extend([lm.x, lm.y, lm.z])
    return landmarks_vector

@app.post("/answerfile/")
async def create_upload_file(file: UploadFile = File(...), wordNo: int = Form(...), wordDes: int = Form(...)):
    contents = await file.read()
    
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Failed to process image data: {str(e)}"}

    image = np.array(pil_img)

    # Mediapipe 이미지 처리
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        # OpenCV로 이미지 변환 (RGB에서 BGR로 변환)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Mediapipe로 처리
        results = hands.process(image_bgr)

        hand_landmarks_vectors = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # 이미지를 BGR 형식으로 그리기
                landmarks_vector = get_hand_landmarks_vector(hand_landmarks)
                hand_landmarks_vectors.append(landmarks_vector)

    counts = len(hand_landmarks_vectors)
    isPerson = counts > 0  # 손이 감지되었는지 여부

    # MySQL에서 데이터 가져오기
    query = f"SELECT vector FROM handlandmark WHERE wordNo = {wordNo}"
    mycursor.execute(query)
    rows = mycursor.fetchall()

    # 가져온 데이터와 비교하여 코사인 유사도 계산
    similarity_threshold = 0.8
    is_similar = False

    for row in rows:
        db_landmarks_vector = json.loads(row[0])  # MySQL에서 가져온 벡터를 JSON에서 파싱
        db_landmarks_vector = np.array(db_landmarks_vector, dtype=np.float32).flatten()  # 리스트를 NumPy 배열로 변환 후 평평하게 만듦
        for detected_vector in hand_landmarks_vectors:
            detected_vector = np.array(detected_vector, dtype=np.float32).flatten()  # 리스트를 NumPy 배열로 변환 후 평평하게 만듦
            similarity = 1 - cosine(db_landmarks_vector, detected_vector)
            if similarity >= similarity_threshold:
                is_similar = True
                break
        if is_similar:
            break

    # MySQL에서 words 테이블의 wordNo를 가져오기
    query_wordNo = f"SELECT wordNo FROM words WHERE wordNo = {wordNo}"
    mycursor.execute(query_wordNo)
    wordNo_result = mycursor.fetchone()

    if wordNo_result:
        wordNo = wordNo_result[0]
    else:
        return {"error": "Invalid wordNo. Word does not exist in database."}

    # 데이터베이스에 isCorrect 값 저장
    sql = "INSERT INTO mypage (wordNo, isCorrect) VALUES (%s, %s)"
    val = (wordNo, is_similar)
    mycursor.execute(sql, val)
    mydb.commit()

    return {"isSimilar": is_similar, "image": image_base64}
