import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import os

# 손 찾기 관련 기능 불러오기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
    
# 손 찾기 세부 설정
hands = mp_hands.Hands(
    max_num_hands=2,  # 탐지할 최대 손의 개수
    min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
    min_tracking_confidence=0.5  # 표시할 관찰의 최소 정확도
)

# 이미지 파일 읽어오기
image_path = 'image/easy/2_살찌다_1.PNG'  # 여기에 이미지 파일 경로를 입력하세요

# 경로가 유효한지 확인
if not os.path.exists(image_path):
    print(f"Error: Path {image_path} does not exist")
else:
    print(f"Path {image_path} exists")
    
    # PIL을 사용하여 이미지 읽기
    try:
        image = Image.open(image_path)
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # PIL 이미지는 RGB이므로 BGR로 변환
    except Exception as e:
        print(f"Error: Could not open the image with PIL. {e}")
        img = None

    if img is None:
        print(f"Error: Could not read the image at path {image_path}")
    else:
        # BGR에서 RGB로 이미지 바꿔주기
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

        # 손 탐지
        result = hands.process(img)

        # 찾은 손 표시하기 및 랜드마크 좌표 추출
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 랜드마크 좌표 추출
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                landmarks = np.array(landmarks).flatten()

                # 여기서 landmarks는 손의 각 관절의 3D 좌표를 포함하는 벡터입니다.
                print("Landmarks:", landmarks)

        # RGB에서 BGR로 이미지 바꿔주기 (cv2.imshow를 위해)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Tracking', img)
        cv2.waitKey(0)  # 키 입력을 기다립니다.
        cv2.destroyAllWindows()
