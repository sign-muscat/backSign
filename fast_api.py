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





# from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
# from fastapi.responses import JSONResponse
# from sqlalchemy.orm import Session
# import cv2
# import mediapipe as mp
# import numpy as np
# import io
# from PIL import Image

# from database import engine, SessionLocal, Base
# import models
# import crud

# models.Base.metadata.create_all(bind=engine)

# app = FastAPI()

# # 손 찾기 관련 기능 불러오기
# mp_hands = mp.solutions.hands
# # 손 그려주는 기능 불러오기
# mp_drawing = mp.solutions.drawing_utils

# # 손 찾기 세부 설정
# hands = mp_hands.Hands(
#     max_num_hands=2,  # 탐지할 최대 손의 개수
#     min_detection_confidence=0.5,  # 표시할 손의 최소 정확도
#     min_tracking_confidence=0.5  # 표시할 관찰의 최소 정확도
# )

# # 데이터베이스 세션을 생성하고 종료하는 종속성을 정의합니다.
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# @app.post("/upload/")
# async def upload_image(file: UploadFile = File(...), word_no: int = Form(...), db: Session = Depends(get_db)):
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

#             # DB에서 정답 값 불러오기
#             db_landmarks = crud.get_hand_landmarks(db, word_no)
#             if not db_landmarks:
#                 raise HTTPException(status_code=404, detail="Hand landmarks not found for the word")

#             db_landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in db_landmarks])

#             similarity = calculate_cosine_similarity(landmarks, db_landmark_array)
#             threshold = 0.8  # 임계값 설정
#             is_correct = similarity > threshold

#             return JSONResponse(content={"similarity": similarity, "is_correct": is_correct})

#     return JSONResponse(content={"error": "No hand landmarks found"}, status_code=404)

# def calculate_cosine_similarity(landmarks1, landmarks2):
#     landmarks1_flat = landmarks1.flatten()
#     landmarks2_flat = landmarks2.flatten()
#     return 1 - cosine(landmarks1_flat, landmarks2_flat)

# @app.on_event("startup")
# async def startup():
#     await database.connect()

# @app.on_event("shutdown")
# async def shutdown():
#     await database.disconnect()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import io
from PIL import Image
import mysql.connector
from scipy.spatial.distance import cosine


app = FastAPI()

# 손 찾기 관련 기능 불러오기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MySQL 데이터베이스 설정
db_config = {
    'user': 'sign',
    'password': 'sign',
    'host': 'localhost',
    'database': 'sign_db',
    'port': '3306'
}

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...), word_no: int = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 손 탐지
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).flatten()

            # MySQL 데이터베이스에 연결하고 단어 번호를 기준으로 정보 조회
            try:
                conn = mysql.connector.connect(**db_config)
                cursor = conn.cursor()
                cursor.execute("SELECT x, y, z FROM handlandmark WHERE wordNo = %s", (word_no,))
                rows = cursor.fetchall()
            except mysql.connector.Error as err:
                return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)
            finally:
                conn.close()

            if not rows:
                raise HTTPException(status_code=404, detail="Hand landmarks not found for the word")

            db_landmark_array = np.array(rows)

            similarity = calculate_cosine_similarity(landmarks, db_landmark_array)
            threshold = 0.5  # 임계값 설정
            is_correct = similarity > threshold

            return JSONResponse(content={"similarity": similarity, "is_correct": is_correct})

    return JSONResponse(content={"error": "No hand landmarks found"}, status_code=404)

def calculate_cosine_similarity(landmarks1, landmarks2):
    landmarks1_flat = landmarks1.flatten()
    landmarks2_flat = landmarks2.flatten()
    return 1 - cosine(landmarks1_flat, landmarks2_flat)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


