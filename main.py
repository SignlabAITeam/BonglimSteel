from MainProcess.database.schema.models import *
from MainProcess.data_prep.src.functions import get_abno_value, get_corr_value
from MainProcess.data_prep.src.preprocessing import slicing_data
from MainProcess.database.base import SessionLocal
from MainProcess.util.src.my_utils import make_report
from dotenv import load_dotenv
from datetime import datetime

import pandas as pd
import requests
import schedule
import logging
import json
import time
import os

load_dotenv()

def main_process(session):
    """DB에 있는 state값을 확인해서 0인 값들의 AMS_IDX 값을 모두 가져와 데이터를 날짜에 맞게 슬라이싱한 뒤 state값 변경
    
    ## Input:
    - session : 시스템에 로그인 한 이후의 세션
    
    """
    
    db = SessionLocal()  # db객체 생성
    query = db.query(AI_MODEL_STATE).all()  # status 테이블 값 전부 가져와서
    origin_path = os.getcwd()
    
    columns = [col.name for col in AI_MODEL_STATE.__table__.columns]  # 컬럼명 추출
    data = {col : [] for col in columns}  # 데이터프레임화 하기 위한 딕셔너리 생성
    
    for row in query:  # state 테이블 값들을 순회하면서
        for col in columns:
            data[col].append(getattr(row, col))  # 딕셔너리에 값 추가
    
    ams_index = 0  # 작업 대상의 AMS_IDX값을 할당할 기본값 생성
    df = pd.DataFrame(data)  # state테이블 데이터프레임화
    state = df['MODEL_STATE']  # model_state컬럼을 대상으로
    for index, value in state.items():  # index와 value값(state)을 순회하면서
        if value == 0:  # value(state)가 0이라면
            ams_index = df['AMS_IDX'].loc[index]  # ams_idx값을 idx에 할당
            break

    save_dir = os.getenv('SHARE_FOLDER_IP')
    save_target = "anlsDoc"
    
    save_at = os.path.join(save_dir, save_target)
    os.chdir(save_at)
    
    if ams_index != 0:  # state가 0인 인덱스가 할당되었을때만 작업
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] : 작업 대상 인덱스 -> {ams_index}")
        start_date = df[df['AMS_IDX'] == ams_index]['MODEL_START_DTE'].values[0]  # start date와
        end_date = df[df['AMS_IDX'] == ams_index]['MODEL_END_DTE'].values[0]  # end date를 들고와서
        
        sliced_df = slicing_data("CONT_COMP_FAC_DATA_ONE", start_date, end_date)  # 데이터를 슬라이싱하고
        update_target = db.query(AI_MODEL_STATE).filter(AI_MODEL_STATE.AMS_IDX == ams_index).first()  # ams_idx가 동일한 객체를 가져오고
        if sliced_df.shape[0] != 0:  # 슬라이싱된 데이터의 건수가 있으면
            make_report(sliced_df, ams_index, save_at)  # 리포트 생성 후
            
            # AI_DATA_ANLS 테이블 업데이트 준비
            anls_corr = get_corr_value(df=sliced_df, target='보링 가공 전,후 속도-왼팔')
            anls_abno_cnt = get_abno_value(df=sliced_df)
            anls_data_filename = f"AMS_IDX_{ams_index}.html"
            db.add(
                AI_DATA_ANLS(
                    ANLS_CORR=anls_corr, 
                    ANLS_ABNO_CNT=anls_abno_cnt, 
                    ANLS_START_DTE=start_date, 
                    ANLS_END_DTE=end_date,
                    ANLS_DATA_FILENAME=anls_data_filename,
                    ANLS_REG_DTE=now
                    )
                )
            
        # 값들을 업데이트한다
        update_target.MODEL_STATE = 1
        update_target.MODEL_FIN_DTE = now
        
        db.commit()
        
        # 예측 완료를 알리기위해서 idx값을 전송하고
        res = session.get(f"{os.getenv('BACKEND_URL')}/api/aimodel/finish?amsIdx={ams_index}")
        if res.status_code == 200:  # 전송이 완료되었다면 
            print(f"[{now}] : 예측 프로세스 완료, 전달된 idx값 : {ams_index}")
        else:
            print(res.status_code)
        
        print(f"[{now}] : 데이터 업데이트 완료.......")
    else:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] : Data is already up-to-date...")
        
    # 사용 종료 이후에는 세션 누수를 방지하기 위해서 close
    db.close()
    os.chdir(origin_path)
    
def login():
    """다른 api들에 요청을 보낼 때 필요한 권한을 챙기기 위해 로그인 진행 후 세션을 리턴
    
    ## Input:
    - None
    
    ## Output:
    - session : 로그인 한 이후의 세션을 리턴
    """
    url = F"{os.getenv('BACKEND_URL')}/api/loginCheck"
    data = {
        "username":f"{os.getenv('API_USER')}",
        "password":f"{os.getenv('API_PASSWORD')}"
    }
    headers = {
        "Content-Type":"application/json"
    }

    session = requests.Session()
    res = session.post(url, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        return session
    else:
        return
    
def check_system_connection(session):
    """예측 프로세스가 실행중임을 알릴 수 있는 API 주소에 지속적으로 ok sign 전달
    
    ## Input:
    - session : 로그인 한 이후의 세션
    
    ## Output:
    - return None
    """
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        res = session.get(F"{os.getenv('BACKEND_URL')}/api/util/modelcheck?msg=ok")
        print(f"[{now}] : Program is Running...")
    except:
        print("I'm Die!")
        
def set_scheduler():
    """
    3초마다 main process를 실행하는 스케줄러와 15초마다 프로그램이 실행중임을 알리는 스케줄러 설정
    """
    
    schedule.every(3).seconds.do(main_process, session)
    schedule.every(15).seconds.do(check_system_connection, session)
        

if __name__ == "__main__":
    logging.basicConfig(
        filename="./logging.log",
        level=logging.ERROR,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        encoding="cp949"
    )
    
    session = login()
    if session is not None:
        set_scheduler()
        print(f"[Login Status] : 로그인 성공")
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                print("[Program Status] : 오류 발생..")
                logging.error("예외 발생", exc_info=True)
                
                session.close()
                schedule.clear()
                time.sleep(2)
                
                session = login()
                if session is not None:
                    set_scheduler()
                    print(f"[Login Status] : 재로그인 성공")