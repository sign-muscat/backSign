# import random
# import string
# import json
# import cv2
# import io
# import numpy as np
# import mediapipe as mp
# import mysql.connector
# import logging
# import base64
# from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List
# from PIL import Image
# from pydantic import BaseModel
# from scipy.spatial.distance import cosine


# app = FastAPI()

# # CORS 설정
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 모든 도메인 허용
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # MySQL 연결 설정
# mydb = mysql.connector.connect(
#     host="localhost",
#     user="sign",
#     password="sign",
#     database="sign_db"
# )

# # MySQL 데이터베이스 커서 생성
# mycursor = mydb.cursor(dictionary=True)

# class WordResponse(BaseModel):
#     wordDes: int
#     wordName: str
#     poseCount: int

# class NickNameResponse(BaseModel):
#     myNo: int
#     nickName: str

# def generate_random_nickname(length: int = 20) -> str:
#     characters = string.ascii_letters + string.digits + string.punctuation
#     random_nickname = ''.join(random.choice(characters) for i in range(length))
#     return random_nickname

# def clean_word_name(word_name: str) -> str:
#     # '_숫자' 부분 제거
#     return '_'.join(word_name.split('_')[:-1])

# # 난이도별 랜덤 문제 출제
# @app.get("/get-words", response_model=List[WordResponse])
# def get_words(difficulty: str = Query(..., regex="^(쉬움|보통|어려움)$"), count: int = 5):
#     level_map = {
#         "쉬움": "하",
#         "보통": "중",
#         "어려움": "상"
#     }
#     level = level_map.get(difficulty)

#     if not level:
#         raise HTTPException(status_code=400, detail="Invalid difficulty level")

#     # wordDes의 고유한 갯수를 제한하여 가져오는 쿼리
#     sql = """
#         SELECT w.wordDes, COUNT(w.wordNo) as poseCount, MIN(w.wordName) as wordName
#         FROM words w
#         JOIN grade g ON w.wordNo = g.wordNo
#         WHERE g.levels = %s
#         GROUP BY w.wordDes
#     """
#     mycursor.execute(sql, (level,))
#     word_des_list = mycursor.fetchall()

#     if not word_des_list:
#         raise HTTPException(status_code=404, detail="No words found for the given difficulty level")

#     # wordDes 목록을 무작위로 섞음
#     random.shuffle(word_des_list)

#     # 제한된 수만큼 선택
#     selected_words = word_des_list[:count]

#     # wordName에서 '_숫자' 부분 제거 및 단어 목록 수정
#     for word in selected_words:
#         word['wordName'] = clean_word_name(word['wordName'])

#     # 가져온 단어 목록을 콘솔에 출력
#     print("Fetched words from DB:", selected_words)

#     return selected_words



# # MySQL 데이터베이스 커서 생성
# mycursor = mydb.cursor(dictionary=True)

# class VideoLinkResponse(BaseModel):
#     wordNo: int
#     wordDes: int
#     wordName: str
#     videoLink: str

# @app.get("/get-video-link", response_model=VideoLinkResponse)
# def get_video_link(wordDes: int = Query(...)):
#     # 특정 wordDes에 대한 wordName 및 videoLink 가져오는 쿼리
#     sql = """
#         SELECT wordNo, wordDes, wordName, videoLink
#         FROM words
#         WHERE wordDes = %s
#         ORDER BY wordNo
#         LIMIT 1
#     """
#     mycursor.execute(sql, (wordDes,))
#     word = mycursor.fetchone()

#     if not word:
#         raise HTTPException(status_code=404, detail="No video links found for the given wordDes")

#     # 가져온 단어를 콘솔에 출력
#     print("Fetched video link from DB:", word)

#     return word




# # wordDes와 poseStep을 받아서 words 테이블에서 wordDes가 같은 데이터들을 반환하는 엔드포인트
# @app.post("/gameStart/")
# async def get_words(wordDes: int = Form(...), poseStep: int = Form(...)):
#     try:
#         conn = mydb
#         cursor = conn.cursor()
        
#         # wordDes가 일치하는 데이터 조회
#         query = """
#             SELECT wordNo, wordDes, wordName, wordImg
#             FROM words
#             WHERE wordDes = %s
#         """
        
#         cursor.execute(query, (wordDes,))
#         rows = cursor.fetchall()

#         if not rows:
#             cursor.close()
#             conn.close()
#             raise HTTPException(status_code=404, detail="Words not found for the given wordDes")

#         # poseStep 값과 wordName의 _ 뒤의 숫자가 일치하는 데이터 필터링
#         for row in rows:
#             wordNo, wordDes, wordName, wordImg = row
#             # wordName에서 _ 뒤의 숫자를 추출하여 poseStep과 비교
#             try:
#                 if int(wordName.split('_')[-1]) == poseStep:
#                     cursor.close()
#                     conn.close()
#                     return {"wordNo": wordNo, "wordDes": wordDes, "wordImg": wordImg}
#             except ValueError:
#                 continue  # 숫자가 아닌 경우 넘어가기

#         cursor.close()
#         conn.close()
#         raise HTTPException(status_code=404, detail="No matching word found for the given poseStep")
#     except mysql.connector.Error as err:
#         logging.error(f"Database error: {err}")
#         raise HTTPException(status_code=500, detail="Database error")



# # Mediapipe 사용하기
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # 손의 랜드마크를 벡터화하여 반환하는 함수
# def get_hand_landmarks_vector(hand_landmarks):
#     landmarks_vector = []
#     for lm in hand_landmarks.landmark:
#         landmarks_vector.extend([lm.x, lm.y, lm.z])
#     return landmarks_vector

# class VideoLinkResponse(BaseModel):
#     wordNo: int
#     wordDes: int
#     wordName: str
#     videoLink: str

# @app.post("/answerfile/")
# async def create_upload_file(file: UploadFile = File(...), wordNo: int = Form(...), wordDes: int = Form(...)):
#     contents = await file.read()
    
#     try:
#         pil_img = Image.open(io.BytesIO(contents))
#     except Exception as e:
#         return {"error": f"Failed to process image data: {str(e)}"}

#     image = np.array(pil_img)

#     # Mediapipe 이미지 처리
#     with mp_hands.Hands(
#         max_num_hands=2,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:
        
#         # OpenCV로 이미지 변환 (RGB에서 BGR로 변환)
#         image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Mediapipe로 처리
#         results = hands.process(image_bgr)

#         hand_landmarks_vectors = []
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # 이미지를 BGR 형식으로 그리기
#                 landmarks_vector = get_hand_landmarks_vector(hand_landmarks)
#                 hand_landmarks_vectors.append(landmarks_vector)

#     counts = len(hand_landmarks_vectors)
#     isPerson = counts > 0  # 손이 감지되었는지 여부

#     # MySQL에서 데이터 가져오기
#     query = f"SELECT vector FROM handlandmark WHERE wordNo = {wordNo}"
#     mycursor.execute(query)
#     rows = mycursor.fetchall()

#     # 가져온 데이터와 비교하여 코사인 유사도 계산
#     similarity_threshold = 0.8
#     is_similar = False

#     for row in rows:
#         db_landmarks_vector = json.loads(row['vector'])  # MySQL에서 가져온 벡터를 JSON에서 파싱
#         db_landmarks_vector = np.array(db_landmarks_vector, dtype=np.float32).flatten()  # 리스트를 NumPy 배열로 변환 후 평평하게 만듦
#         for detected_vector in hand_landmarks_vectors:
#             detected_vector = np.array(detected_vector, dtype=np.float32).flatten()  # 리스트를 NumPy 배열로 변환 후 평평하게 만듦
#             similarity = 1 - cosine(db_landmarks_vector, detected_vector)
#             if similarity >= similarity_threshold:
#                 is_similar = True
#                 break
#         if is_similar:
#             break

#     # MySQL에서 words 테이블의 wordNo를 가져오기
#     query_wordNo = f"SELECT wordNo FROM words WHERE wordNo = {wordNo}"
#     mycursor.execute(query_wordNo)
#     wordNo_result = mycursor.fetchone()

#     if wordNo_result:
#         wordNo = wordNo_result['wordNo']
#     else:
#         return {"error": "Invalid wordNo. Word does not exist in database."}

#     # 데이터베이스에 isCorrect 값 저장
#     sql = "INSERT INTO mypage (wordNo, isCorrect) VALUES (%s, %s)"
#     val = (wordNo, is_similar)
#     mycursor.execute(sql, val)
#     mydb.commit()

#     # 이미지를 Base64로 인코딩
#     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     _, img_encoded = cv2.imencode('.jpg', image_bgr)
#     image_base64 = base64.b64encode(img_encoded).decode('utf-8')

#     return {"isSimilar": is_similar, "image": image_base64}


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




import random
import string
import json
import cv2
import io
import numpy as np
import mediapipe as mp
import mysql.connector
import logging
import base64
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from PIL import Image
from scipy.spatial.distance import cosine
from datetime import datetime

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MySQL 연결 설정
db_config = {
    'user': 'sign',
    'password': 'sign',
    'host': 'localhost',
    'database': 'sign_db',
    'port': '3306'
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL 데이터베이스 커서 생성
mydb = mysql.connector.connect(**db_config)
mycursor = mydb.cursor(dictionary=True)

class WordResponse(BaseModel):
    wordDes: int
    wordName: str
    poseCount: int

class NickNameResponse(BaseModel):
    myNo: int
    nickName: str

def generate_random_nickname(length: int = 20) -> str:
    characters = string.ascii_letters + string.digits + string.punctuation
    random_nickname = ''.join(random.choice(characters) for i in range(length))
    return random_nickname

def clean_word_name(word_name: str) -> str:
    return '_'.join(word_name.split('_')[:-1])

@app.get("/get-words", response_model=List[WordResponse])
def get_words(difficulty: str = Query(..., regex="^(쉬움|보통|어려움)$"), count: int = 5):
    level_map = {
        "쉬움": "하",
        "보통": "중",
        "어려움": "상"
    }
    level = level_map.get(difficulty)

    if not level:
        raise HTTPException(status_code=400, detail="Invalid difficulty level")

    sql = """
        SELECT w.wordDes, COUNT(w.wordNo) as poseCount, MIN(w.wordName) as wordName
        FROM words w
        JOIN grade g ON w.wordNo = g.wordNo
        WHERE g.levels = %s
        GROUP BY w.wordDes
    """
    mycursor.execute(sql, (level,))
    word_des_list = mycursor.fetchall()

    if not word_des_list:
        raise HTTPException(status_code=404, detail="No words found for the given difficulty level")

    random.shuffle(word_des_list)
    selected_words = word_des_list[:count]

    for word in selected_words:
        word['wordName'] = clean_word_name(word['wordName'])

    return selected_words

class VideoLinkResponse(BaseModel):
    wordNo: int
    wordDes: int
    wordName: str
    videoLink: str

@app.get("/get-video-link", response_model=VideoLinkResponse)
def get_video_link(wordDes: int = Query(...)):
    sql = """
        SELECT wordNo, wordDes, wordName, videoLink
        FROM words
        WHERE wordDes = %s
        ORDER BY wordNo
        LIMIT 1
    """
    mycursor.execute(sql, (wordDes,))
    word = mycursor.fetchone()

    if not word:
        raise HTTPException(status_code=404, detail="No video links found for the given wordDes")

    return word

@app.post("/gameStart/")
async def get_words(wordDes: int = Form(...), poseStep: int = Form(...)):
    try:
        conn = mydb
        cursor = conn.cursor()
        
        query = """
            SELECT wordNo, wordDes, wordName, wordImg
            FROM words
            WHERE wordDes = %s
        """
        
        cursor.execute(query, (wordDes,))
        rows = cursor.fetchall()

        if not rows:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Words not found for the given wordDes")

        for row in rows:
            wordNo, wordDes, wordName, wordImg = row
            try:
                if int(wordName.split('_')[-1]) == poseStep:
                    cursor.close()
                    conn.close()
                    return {"wordNo": wordNo, "wordDes": wordDes, "wordImg": wordImg}
            except ValueError:
                continue

        cursor.close()
        conn.close()
        raise HTTPException(status_code=404, detail="No matching word found for the given poseStep")
    except mysql.connector.Error as err:
        logging.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_hand_landmarks_vector(hand_landmarks):
    landmarks_vector = []
    for i in range(21):  # 21개의 랜드마크
        if i < len(hand_landmarks.landmark):
            lm = hand_landmarks.landmark[i]
            landmarks_vector.extend([lm.x, lm.y, lm.z])
        else:
            landmarks_vector.extend([0, 0, 0])  # 부족한 랜드마크는 0으로 채웁니다
    return landmarks_vector

@app.post("/answerfile/")
async def create_upload_file(file: UploadFile = File(...), wordNo: int = Form(...), wordDes: int = Form(...)):
    contents = await file.read()
    
    try:
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        return {"error": f"Failed to process image data: {str(e)}"}

    image = np.array(pil_img)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = hands.process(image_bgr)

        hand_landmarks_vectors = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks_vector = get_hand_landmarks_vector(hand_landmarks)
                hand_landmarks_vectors.append(landmarks_vector)

    counts = len(hand_landmarks_vectors)
    isPerson = counts > 0

    query = f"SELECT vector FROM handlandmark WHERE wordNo = {wordNo}"
    mycursor.execute(query)
    rows = mycursor.fetchall()

    similarity_threshold = 0.8
    is_similar = False

    for row in rows:
        db_landmarks_vector = json.loads(row['vector'])
        db_landmarks_vector = np.array(db_landmarks_vector, dtype=np.float32).flatten()
        
        # 데이터베이스 벡터의 크기를 63으로 조정
        if len(db_landmarks_vector) < 63:
            db_landmarks_vector = np.pad(db_landmarks_vector, (0, 63 - len(db_landmarks_vector)), 'constant')
        elif len(db_landmarks_vector) > 63:
            db_landmarks_vector = db_landmarks_vector[:63]
        
        print(f"DB vector size: {len(db_landmarks_vector)}")
        
        for detected_vector in hand_landmarks_vectors:
            detected_vector = np.array(detected_vector, dtype=np.float32).flatten()
            
            # 감지된 벡터의 크기를 63으로 조정
            if len(detected_vector) < 63:
                detected_vector = np.pad(detected_vector, (0, 63 - len(detected_vector)), 'constant')
            elif len(detected_vector) > 63:
                detected_vector = detected_vector[:63]
            
            print(f"Detected vector size: {len(detected_vector)}")
            
            try:
                similarity = 1 - cosine(db_landmarks_vector, detected_vector)
                if similarity >= similarity_threshold:
                    is_similar = True
                    break
            except ValueError as e:
                print(f"Vector size mismatch: db_vector size = {len(db_landmarks_vector)}, detected_vector size = {len(detected_vector)}")
                raise HTTPException(status_code=500, detail=f"Vector size mismatch: {str(e)}")
        
        if is_similar:
            break

    query_wordNo = f"SELECT wordNo FROM words WHERE wordNo = {wordNo}"
    mycursor.execute(query_wordNo)
    wordNo_result = mycursor.fetchone()

    if wordNo_result:
        wordNo = wordNo_result['wordNo']
    else:
        return {"error": "Invalid wordNo. Word does not exist in database."}

    sql = "INSERT INTO mypage (wordNo, isCorrect) VALUES (%s, %s)"
    val = (wordNo, is_similar)
    mycursor.execute(sql, val)
    mydb.commit()

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.jpg', image_bgr)
    image_base64 = base64.b64encode(img_encoded).decode('utf-8')

    return {"isSimilar": is_similar, "image": image_base64}

class ResultRequest(BaseModel):
    nickname: str
    levels: str
    today: datetime
    wordDes: int
    word_name: str
    isCorrect: bool

@app.post("/result/")
async def save_result(request: ResultRequest):
    data = request.dict()
    nickname = data.get("nickname")
    levels = data.get("levels")
    today = data.get("today")
    wordDes = data.get("wordDes")
    word_name = data.get("word_name")
    isCorrect = data.get("isCorrect")

    if not nickname or levels is None or today is None or wordDes is None or word_name is None or isCorrect is None:
        raise HTTPException(status_code=400, detail="Invalid data")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        logger.info(f"Received wordDes: {wordDes}")
        logger.info(f"Type of wordDes: {type(wordDes)}")

        cursor.execute("SELECT wordNo FROM words WHERE wordDes = %s", (wordDes,))
        result = cursor.fetchone()
        logger.info(f"Query result for wordDes {wordDes}: {result}")

        if not result:
            raise HTTPException(status_code=400, detail="Invalid wordDes: does not exist in words table")
        word_no = result[0]

        # 결과 저장 로직 추가 필요...

    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return JSONResponse(content={'error': f'Database error: {err}'}, status_code=500)
    
    return JSONResponse(content={"message": "Result saved successfully"})

@app.get("/ranking/")
async def get_ranking():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            WITH user_stats AS (
                SELECT 
                    m.nickName, 
                    SUM(m.isCorrect) / COUNT(*) AS correct_ratio,
                    MAX(r.today) AS last_played_date
                FROM mypage m
                LEFT JOIN ranks r ON m.myNo = r.myNo
                GROUP BY m.nickName
            ),
            user_ranks AS (
                SELECT 
                    nickName, 
                    correct_ratio, 
                    last_played_date,
                    RANK() OVER (ORDER BY correct_ratio DESC) as `rank`
                FROM user_stats
            )
            SELECT 
                ur.nickName, 
                ur.correct_ratio, 
                ur.last_played_date, 
                g.levels AS last_played_level,
                ur.`rank`
            FROM user_ranks ur
            LEFT JOIN ranks r ON r.myNo = (
                SELECT myNo 
                FROM ranks 
                WHERE myNo IN (SELECT myNo FROM mypage WHERE nickName = ur.nickName) 
                ORDER BY today DESC 
                LIMIT 1
            )
            LEFT JOIN grade g ON r.gradeNo = g.gradeNo
            ORDER BY ur.`rank`
        """
        cursor.execute(query)
        rankings = cursor.fetchall()
        cursor.close()
        conn.close()

        ranked_results = [
            {
                "nickName": rank[0], 
                "correct_ratio": float(rank[1]), 
                "today": rank[2],
                "levels": rank[3] if rank[3] else None,
                "rank": rank[4]
            } 
            for rank in rankings
        ]

        return {"rankings": ranked_results}
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    
@app.get("/word_step/")
async def get_word_step(word_des: int, pose_num: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT wordImg FROM words WHERE wordDes = %s", (word_des,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        if not rows or pose_num < 1 or pose_num > len(rows):
            raise HTTPException(status_code=404, detail="Step image not found")

        selected_image = rows[pose_num - 1][0]

        return {"imageUrl": selected_image}
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/word/")
async def get_word(word_no: int):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = """
            SELECT wordNo, wordDes, wordImg
            FROM words
            WHERE wordNo = %s
        """
        cursor.execute(query, (word_no,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Word not found")

        return {
            "wordNo": row[0],
            "wordDes": row[1],
            "wordImg": row[2]
        }
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail="Database error")

@app.get("/user_ranking/")
async def get_user_ranking(nickname: str):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
        WITH user_rankings AS (
            SELECT 
                nickName,
                SUM(isCorrect) / COUNT(*) AS correct_ratio,
                RANK() OVER (ORDER BY SUM(isCorrect) / COUNT(*) DESC) as overall_rank
            FROM mypage
            GROUP BY nickName
        ),
        last_game AS (
            SELECT 
                m.nickName,
                r.today as last_played_date,
                g.levels as last_played_level
            FROM mypage m
            JOIN ranks r ON m.myNo = r.myNo
            JOIN grade g ON r.gradeNo = g.gradeNo
            WHERE m.nickName = %s
            ORDER BY r.today DESC
            LIMIT 1
        )
        SELECT 
            ur.correct_ratio,
            lg.last_played_date,
            lg.last_played_level,
            ur.overall_rank
        FROM user_rankings ur
        JOIN last_game lg ON ur.nickName = lg.nickName
        WHERE ur.nickName = %s
        """
        
        cursor.execute(query, (nickname, nickname))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            raise HTTPException(status_code=404, detail="User not found or no game results")

        user_ranking = {
            "correct_ratio": float(result[0]),
            "today": result[1],
            "levels": result[2],
            "rank": result[3]
        }

        return user_ranking
    
    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        raise HTTPException(status_code=500, detail=f"Database error: {err}")
    
class QuestionResult(BaseModel):
    word_name: str
    is_correct: bool

class GameResult(BaseModel):
    correct_percentage: str
    questions: List[QuestionResult]

@app.get("/user_game_result/{nickname}", response_model=GameResult)
async def get_user_game_result(nickname: str):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT 
            SUBSTRING_INDEX(w.wordName, '_', 1) as word_name,
            MIN(m.isCorrect) as is_correct
        FROM mypage m
        JOIN words w ON m.wordNo = w.wordNo
        WHERE m.nickName = %s
        GROUP BY SUBSTRING_INDEX(w.wordName, '_', 1)
        ORDER BY MIN(m.myNo)
        """
        
        cursor.execute(query, (nickname,))
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()

        if not results:
            raise HTTPException(status_code=404, detail="User not found or no game results")

        correct_count = sum(result['is_correct'] for result in results)
        total_count = len(results)
        
        correct_percentage = f"{correct_count}/{total_count}"
        
        questions = [QuestionResult(**result) for result in results]

        return GameResult(
            correct_percentage=correct_percentage,
            questions=questions
        )
    
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {err}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

