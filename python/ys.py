import cv2 
import mediapipe as mp

from mediapipe.tasks import python

#mediapipe 사용하기
#손찾기 관련 기능 불러오기
mp_hands = mp.solutions.hands
#손 그려주는 기능 불러오기
mp_drawing = mp.solutions.drawing_utils

#손 찾기 관연 세부 설정
hands = mp_hands.Hands(
    max_num_hands = 2,# 탐지할 최대 손의 갯수
    min_detection_confidence = 0.5, # 표시할 손의 최소 정확도
    min_tracking_confidence = 0.5 # 표시할 관찰의 최소 정확도

)

# 이미지 파일 읽어오기
image_path = 'python\\test_img\\test1.png'  # 여기에 이미지 파일 경로를 입력하세요
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR에서 RGB로 이미지 바꿔주기

# 손 탐지
result = hands.process(img)

# 좌표값 출력하기
if result.multi_hand_landmarks is not None:
    for hand_landmarks in result.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy, cz = lm.x * w, lm.y * h, lm.z
            print(f"ID: {id}, Vector: ({cx}, {cy}, {cz})")
