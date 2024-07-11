import os
import glob
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
            for lm in hand_landmarks.landmark:
                cx, cy, cz = lm.x, lm.y, lm.z
                landmarks.append((cx, cy, cz))
    
    return landmarks

def save_to_database(wordNo, landmarks):
    for x, y, z in landmarks:
        # handlandmark 테이블에 데이터 삽입
        sql = "INSERT INTO handlandmark (wordNo, x, y, z) VALUES (%s, %s, %s, %s)"
        val = (wordNo, float(x), float(y), float(z))
        mycursor.execute(sql, val)
    
    mydb.commit()

def add_word_to_database(word_name):
    # words 테이블에 데이터 삽입
    sql = "INSERT INTO words (wordName) VALUES (%s)"
    val = (word_name,)
    mycursor.execute(sql, val)
    mydb.commit()

    # 삽입된 단어의 wordNo 가져오기
    wordNo = mycursor.lastrowid

    return wordNo

def process_images_in_folder(folder_path):
    # 폴더 내 모든 이미지 파일 경로 가져오기
    image_files = glob.glob(os.path.join(folder_path, '*.PNG'))  # 확장자 수정
    if not image_files:
        print(f"No images found in {folder_path}")
    for image_path in image_files:
        # 이미지 파일 이름에서 단어 이름 추출
        word_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing {image_path} with word name: {word_name}")

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
    process_images_in_folder('C:/d드라이브/codingProject/pythonProject_02/backSign/python/easy')

    # medium 폴더의 이미지 처리 (절대 경로 사용)
    process_images_in_folder('C:/d드라이브/codingProject/pythonProject_02/backSign/python/medium')

    mycursor.close()
    mydb.close()
    print("Database connection closed")
