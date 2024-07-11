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

def is_valid_wordNo(wordNo):
    # words 테이블에서 해당 wordNo가 존재하는지 확인
    sql = "SELECT wordNo FROM words WHERE wordNo = %s"
    val = (wordNo,)
    mycursor.execute(sql, val)
    result = mycursor.fetchone()
    return result is not None

def record_result(wordNo, result):
    # wordNo가 유효한지 확인
    if not is_valid_wordNo(wordNo):
        print(f"Error: 단어 번호 {wordNo}는 존재하지 않습니다.")
        return
    
    # mypage 테이블에 성공/실패 기록 삽입
    sql = "INSERT INTO mypage (wordNo, isCorrect) VALUES (%s, %s)"
    val = (wordNo, result)
    mycursor.execute(sql, val)
    mydb.commit()

def fetch_results(wordNos):
    # 입력된 단어 번호들에 대한 성공/실패 기록 가져오기
    format_strings = ','.join(['%s'] * len(wordNos))
    sql = f"SELECT wordNo, isCorrect FROM mypage WHERE wordNo IN ({format_strings})"
    mycursor.execute(sql, tuple(wordNos))
    results = mycursor.fetchall()
    return results

if __name__ == "__main__":
    wordNos = []
    results = []

    # 10개의 단어 번호와 성공/실패 입력 받기
    for i in range(10):
        while True:
            try:
                wordNo = int(input(f"{i+1}번째 단어 번호를 입력하세요: "))
                if is_valid_wordNo(wordNo):
                    wordNos.append(wordNo)
                    break
                else:
                    print(f"단어 번호 {wordNo}는 존재하지 않습니다. 다시 입력해주세요.")
            except ValueError:
                print("올바른 숫자를 입력하세요.")
        
        while True:
            result = input(f"{i+1}번째 시도 결과를 입력하세요 (성공/실패): ")
            if result == "성공":
                is_correct = 1
                break
            elif result == "실패":
                is_correct = 0
                break
            else:
                print("올바른 입력이 아닙니다. 성공 또는 실패로 입력해주세요.")
        
        # 결과 기록
        record_result(wordNo, is_correct)
        results.append((wordNo, is_correct))
    
    # 입력된 단어 번호들에 대한 기록 가져오기
    db_results = fetch_results(wordNos)
    word_results = {wordNo: [] for wordNo in wordNos}
    
    for wordNo, isCorrect in db_results:
        word_results[wordNo].append(isCorrect)

    print("\n단어별 성공/실패 기록")
    print("----------------------")
    for wordNo in wordNos:
        correct_count = sum(word_results[wordNo])
        total_count = len(word_results[wordNo])
        success_rate = (correct_count / total_count * 100) if total_count > 0 else 0
        print(f"단어 번호 {wordNo}: {'성공' if word_results[wordNo][-1] == 1 else '실패'}, 정답률: {success_rate:.2f}%")

    total_attempts = sum(len(word_results[wordNo]) for wordNo in wordNos)
    total_correct = sum(sum(word_results[wordNo]) for wordNo in wordNos)
    overall_success_rate = (total_correct / total_attempts * 100) if total_attempts > 0 else 0
    
    print(f"\n전체 정답률: {overall_success_rate:.2f}%")

    # 연결 종료
    mycursor.close()
    mydb.close()
