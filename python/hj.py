import os
import glob
import cv2
import mediapipe as mp
import mysql.connector
import json
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
    max_num_hands=2,  # 탐지할 최대 손의 갯수
    min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
    min_tracking_confidence=0.5  # 표시할 관찰의 최소 정확도
)

def capture_hand_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open/read file: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 이미지 변환

    # 손 탐지
    result = hands.process(img)

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
    
    # handlandmark 테이블에 데이터 삽입
    sql = "INSERT INTO handlandmark (wordNo, vector) VALUES (%s, %s)"
    val = (wordNo, vector_json)
    mycursor.execute(sql, val)
    
    mydb.commit()

def add_word_to_database(word_name):
    # 이미 존재하는 단어인지 확인하고, 존재하지 않으면 추가
    sql_check = "SELECT wordNo FROM words WHERE wordName = %s"
    val_check = (word_name,)
    mycursor.execute(sql_check, val_check)
    existing_word = mycursor.fetchone()
    
    if existing_word:
        wordNo = existing_word[0]
    else:
        # words 테이블에 데이터 삽입
        sql_insert = "INSERT INTO words (wordName) VALUES (%s)"
        val_insert = (word_name,)
        mycursor.execute(sql_insert, val_insert)
        mydb.commit()

        # 삽입된 단어의 wordNo 가져오기
        wordNo = mycursor.lastrowid

    return wordNo

def extract_word_name(filename):
    # 파일 이름에서 확장자를 제외한 부분을 추출
    name_without_extension = os.path.splitext(filename)[0]
    
    # 파일 이름에서 첫 번째 '_' 이후의 문자열을 추출 (단어 이름)
    parts = name_without_extension.split('_')
    if len(parts) > 2:  # '_'가 두 개 이상 있는 경우
        return '_'.join(parts[1:])  # 첫 번째 '_' 이후의 모든 문자열을 추출하여 단어 이름으로 반환
    elif len(parts) == 2:  # '_'가 하나 있는 경우
        return parts[1]  # 두 번째 부분이 단어 이름
    else:  # '_'가 없는 경우 (예외 처리)
        return name_without_extension  # 전체 파일 이름 반환

def process_images_in_folder(folder_path):
    # 폴더 내 모든 이미지 파일 경로 가져오기
    image_files = glob.glob(os.path.join(folder_path, '*.PNG'))
    if not image_files:
        print(f"No images found in {folder_path}")
        
    for image_path in image_files:
        # 이미지 파일 이름에서 단어 이름 추출
        filename = os.path.basename(image_path)
        word_name = extract_word_name(filename)
        print(f"Processing {image_path} with word name: {word_name}")

        # wordNo 추출 (첫 번째 '_' 전의 숫자)
        try:
            wordNo = int(filename.split('_')[0])
        except ValueError:
            print(f"Invalid filename format for {filename}. Skipping.")
            continue

        # 단어 추가 및 wordNo 가져오기
        wordNo = add_word_to_database(word_name)
        print(f"Inserted wordNo: {wordNo}")

        # 손동작 랜드마크 좌표 캡처
        try:
            landmarks = capture_hand_landmarks(image_path)
        except FileNotFoundError as e:
            print(e)
            continue
        print(f"Captured landmarks: {landmarks}")

        # 데이터베이스에 저장
        save_to_database(wordNo, landmarks)
        print("Landmarks saved to database")

if __name__ == "__main__":
    # easy 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder('image/easy')

    # medium 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder('image/medium')
    
    # hard 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder('image\hard')

    mycursor.close()
    mydb.close()
    print("Database connection closed")
