import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import mediapipe as mp
import mysql.connector

# MySQL 연결 설정
mydb = mysql.connector.connect(
    host="localhost",
    user="sign",
    password="sign",
    database="sign_db"
)

# MySQL 데이터베이스 커서 생성
mycursor = mydb.cursor()

# Mediapipe 사용하기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손 찾기 관련 세부 설정
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def capture_hand_landmarks(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 이미지 변환

    # 손 탐지
    result = hands.process(img)

    # 좌표값 출력 및 저장
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                cx, cy, cz = lm.x, lm.y, lm.z
                landmarks.append((cx, cy, cz))
    
    return landmarks

def vectorize_coordinates(landmarks):
    # 각 좌표를 하나의 벡터로 변환
    vector = np.array(landmarks).reshape(1, -1)
    return vector

def fetch_hand_landmarks_from_db(wordNo):
    # 데이터베이스에서 wordNo에 해당하는 손동작 랜드마크를 가져옴
    sql = "SELECT x, y, z FROM handlandmark WHERE wordNo = %s"
    val = (wordNo,)
    mycursor.execute(sql, val)
    result = mycursor.fetchall()

    # 결과를 리스트 형태로 반환
    landmarks = [(float(x), float(y), float(z)) for x, y, z in result]
    return landmarks

def calculate_cosine_similarity(vector1, vector2):
    # 코사인 유사도 계산
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0][0]

if __name__ == "__main__":
    # 사용자 입력 이미지 파일 경로
    image_path = 'python/test_img/test1.png'

    # 단어 번호 (프론트엔드에서 전송될 예정)
    wordNo = 1

    # 손동작 랜드마크 좌표 캡처
    landmarks_from_image = capture_hand_landmarks(image_path)
    vector_from_image = vectorize_coordinates(landmarks_from_image)

    # 데이터베이스에서 손동작 랜드마크 좌표 가져오기
    landmarks_from_db = fetch_hand_landmarks_from_db(wordNo)
    vector_from_db = vectorize_coordinates(landmarks_from_db)

    # 코사인 유사도 계산
    similarity_score = calculate_cosine_similarity(vector_from_image, vector_from_db)
    print(f"Cosine Similarity Score: {similarity_score}")
