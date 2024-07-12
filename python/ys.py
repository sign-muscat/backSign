import os
import glob
import cv2
import mediapipe as mp
import mysql.connector
import json
import numpy as np  # numpy import 추가
from PIL import Image

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
    max_num_hands=2,  # 탐지할 최대 손의 개수
    min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
    min_tracking_confidence=0.5  # 표시할 관찰의 최소 정확도
)

def capture_hand_landmarks(image_path):
    # PIL을 사용하여 이미지 열기
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise FileNotFoundError(f"Cannot open/read file: {image_path}. Error: {e}")
    
    # 손 탐지
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # 좌표값 출력 및 저장
    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            hand_vector = []
            for lm in hand_landmarks.landmark:
                hand_vector.extend([lm.x, lm.y, lm.z])
            landmarks.append(hand_vector)
    
    return landmarks

def save_to_database(wordNo, vector):
    # 벡터 데이터를 JSON 형식으로 변환
    vector_json = json.dumps(vector)
    print(f"Saving to database wordNo: {wordNo}, vector: {vector_json}")
    
    # handlandmark 테이블에 데이터 삽입
    sql = "INSERT INTO handlandmark (wordNo, vector) VALUES (%s, %s)"
    val = (wordNo, vector_json)
    try:
        mycursor.execute(sql, val)
        mydb.commit()
        print("Data committed to database")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

def add_word_to_database(word_name, level):
    # words 테이블에 데이터 삽입
    sql = "INSERT INTO words (wordName) VALUES (%s)"
    val = (word_name,)
    try:
        mycursor.execute(sql, val)
        mydb.commit()
        wordNo = mycursor.lastrowid
        print(f"Inserted wordNo: {wordNo}")
        
        # grade 테이블에 데이터 삽입
        grade_sql = "INSERT INTO grade (wordNo, levels) VALUES (%s, %s)"
        grade_val = (wordNo, level)
        mycursor.execute(grade_sql, grade_val)
        mydb.commit()
        print(f"Inserted grade with wordNo: {wordNo} and level: {level}")
        
        return wordNo
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def process_images_in_folder(folder_path, level):
    # 폴더 내 모든 이미지 파일 경로 가져오기 (대소문자 구분 없이 처리)
    image_files = glob.glob(os.path.join(folder_path, '*.[pP][nN][gG]'))
    if not image_files:
        print(f"No images found in {folder_path}")
    for image_path in image_files:
        # 파일 경로를 정규화하여 인코딩 문제를 방지
        image_path = os.path.normpath(image_path)
        
        # 이미지 파일 이름에서 단어 이름 추출
        full_word_name = os.path.splitext(os.path.basename(image_path))[0]
        word_name = '_'.join(full_word_name.split('_')[1:])
        print(f"Processing {image_path} with word name: {word_name}")

        # 단어 추가 및 wordNo 가져오기
        wordNo = add_word_to_database(word_name, level)
        if wordNo is None:
            continue

        # 손동작 랜드마크 좌표 캡처
        try:
            landmarks = capture_hand_landmarks(image_path)
        except FileNotFoundError as e:
            print(e)
            continue
        print(f"Captured landmarks: {landmarks}")

        if landmarks:  # landmarks가 비어있지 않을 때만 데이터베이스에 저장
            # 데이터베이스에 저장
            save_to_database(wordNo, landmarks)
            print("Landmarks saved to database")
        else:
            print(f"No landmarks captured for image: {image_path}")

if __name__ == "__main__":
    # easy 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder(os.path.abspath('image/easy'), '하')

    # medium 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder(os.path.abspath('image/medium'), '중')

    # hard 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder(os.path.abspath('image/hard'), '상')

    mycursor.close()
    mydb.close()
    print("Database connection closed")
