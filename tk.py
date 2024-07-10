import cv2
import mediapipe as mp

from mediapipe.tasks import python

#캠 연결하기
import cv2

#mediapipe 사용하기
#손찾디 관련 기능 불러오기
mp_hands = mp.solutions.hands
#손 그려주는 기능 불러오기
mp_drawing = mp.solutions.drawing_utils
#손 찾기 관연 세부 설정
hands = mp_hands.Hands(
    max_num_hands = 2,# 탐지할 최대 손의 객수
    min_detection_confidence = 0.5, # 표시할 손의 최소 정확도
    min_tracking_confidence = 0.5 # 표시할 관잘의 최소 정확도

)

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, img = video.read()
    img = cv2.flip(img,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img) # 손 탐지하기

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not ret: 
        break
    #찾은 손 표시하기
    if result.multi_hand_landmarks is not None:
       # print(result.multi_hand_landmarks)
        for res in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    k = cv2.waitKey(30)
    if k==49: # 1번 누르면 기능 : break 되는거
        break
    cv2.imshow('hand', img)
video.release()
cv2.destroyAllWindows() 