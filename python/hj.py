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

def add_word_to_database():

    # 사용자로부터 단어 이름 입력 받기
    word_name = input("단어의 이름은? ")

    # words 테이블에 데이터 삽입
    sql = "INSERT INTO words (wordName) VALUES (%s)"
    val = (word_name,)
    mycursor.execute(sql, val)
    mydb.commit()

    # 삽입된 단어의 wordNo 가져오기
    wordNo = mycursor.lastrowid

    return wordNo

if __name__ == "__main__":

    # 사용자 입력 이미지 파일 경로
    image_path = 'python/test_img/test1.png'

    # 단어 추가 및 wordNo 가져오기
    wordNo = add_word_to_database()
    print(f"Inserted wordNo: {wordNo}")

    # 손동작 랜드마크 좌표 캡처
    landmarks = capture_hand_landmarks(image_path)
    print(f"Captured landmarks: {landmarks}")

    # 데이터베이스에 저장
    save_to_database(wordNo, landmarks)
    print("Landmarks saved to database")

    mycursor.close()
    mydb.close()
    print("Database connection closed")
