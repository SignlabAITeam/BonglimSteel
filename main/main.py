from schema.models import *
from src.utils.data_preprocessing import get_abno_value, get_corr_value
from src.utils.funcs import plotting_hist_subplots
from plotly.subplots import make_subplots
from database.base import SessionLocal
from dotenv import load_dotenv
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
import datapane as dp
import pandas as pd
import numpy as np
import warnings
import requests
import schedule
import logging
import json
import time
import os

warnings.filterwarnings("ignore")
load_dotenv(r"C:\Users\signlab_039\Desktop\projects\bonglim\backend\database\.env")

def main_process(session):
    """
    DB에 있는 state값을 확인해서 0인 값들의 AMS_IDX 값을 모두 가져와 데이터를 날짜에 맞게 슬라이싱한 뒤 state값 변경
    
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
        
        sliced_df = slicing_data(start_date, end_date)  # 데이터를 슬라이싱하고
        update_target = db.query(AI_MODEL_STATE).filter(AI_MODEL_STATE.AMS_IDX == ams_index).first()  # ams_idx가 동일한 객체를 가져오고
        if sliced_df.shape[0] != 0:  # 슬라이싱된 데이터의 건수가 있으면
            make_report(sliced_df, ams_index, save_at)  # 리포트 생성 후
            
            # AI_DATA_ANLS 테이블 업데이트 준비
            anls_corr = get_corr_value(df=sliced_df, target='FSO_BOR_ST_ED_SPEED_L')
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
    
def slicing_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    시작 일자와 끝 일자를 기준으로 데이터를 슬라이싱한다.
    
    ## Input:
    - start_date : 슬라이싱 하려는 데이터 시작점
    - end_date : 슬라이싱 하려는 데이터의 끝 점
    
    ## Output:
    - df : 슬라이싱된 데이터프레임을 리턴
    """
    
    db = SessionLocal()
    query = db.query(CONT_COMP_FAC_DATA_ONE).all()
    columns = [col.name for col in CONT_COMP_FAC_DATA_ONE.__table__.columns]
    data = {col : [] for col in columns}
    
    for row in query:
        for col in columns:
            data[col].append(getattr(row, col))
            
    df = pd.DataFrame(data)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df['FSO_DATETIME'] = pd.to_datetime(df['FSO_DATETIME'])
    df = df[(start_date <= df['FSO_DATETIME']) & (df['FSO_DATETIME'] <= end_date)]
    
    return df

def make_report(data: pd.DataFrame, idx: int, save_dir: str):
    """
    간단한 시각화 자료들을 report로 생성하여 공유 폴더에 저장
    
    ## Input:
    - data : 시각화 대상 dataframe
    - idx : 작업 대상 index
    - save_dir : report.html 파일이 저장될 경로
    
    ## Output:
    - return none
    """
    
    target_data = data.drop(["FSO_DATETIME", "FSO_IDX"], axis=1)
    rename_cols = [col.info.get("comment") for col in CONT_COMP_FAC_DATA_ONE.__table__.columns if col.name not in ["FSO_DATETIME", "FSO_IDX"]]
    origin_cols = target_data.columns
    
    target_data.rename(columns=dict(zip(origin_cols, rename_cols)), inplace=True)
    df = target_data
    
    left_data = df[[col for col in df.columns if "왼팔" in col]]
    right_data = df[[col for col in df.columns if "오른팔" in col]]
    
    left_corr_data = left_data.corr().round(2)  # 상관계수 데이터 준비
    right_corr_data = right_data.corr().round(2)  # 상관계수 데이터 준비

    # 정의한 함수를 기반으로 모든 변수에 대한 히스토그램 시각화
    hist_plot = plotting_hist_subplots(df)
    hist_plot.update_layout(title="각 변수별 히스토그램")

    # 히트맵 시각화
    left_txt = np.round(left_corr_data, 2).astype(str)
    right_txt = np.round(right_corr_data, 2).astype(str)
    corr_subplot = make_subplots(1, 2, subplot_titles=("왼팔 데이터", "오른팔 데이터"), horizontal_spacing=0.19)
    corr_subplot.add_trace(
        go.Heatmap(
            z=left_corr_data.values, 
            x=left_data.columns, 
            y=left_data.columns,
            text=left_txt,
            texttemplate="%{text}"
            ),
        row=1, col=1
    )
    corr_subplot.add_trace(
        go.Heatmap(
            z=right_corr_data.values, 
            x=right_data.columns, 
            y=right_data.columns,
            text=right_txt,
            texttemplate="%{text}"
            ),
        row=1, col=2
    )
    
    corr_subplot.update_layout(width=1600, height=700, title="상관관계 히트맵")
    corr_subplot.update_xaxes(tickangle=45)

    # 박스플롯 시각화
    boxplot = go.Figure()
    for col in df.columns:
        boxplot.add_trace(
            go.Box(y=target_data[col].values, name=col)
        )
    boxplot.update_layout(width=600, height=600, title='각 변수 별 박스플롯')

    # 라인플롯 시각화
    line_plot = go.Figure()
    for col in df.columns:
        line_plot.add_trace(go.Scatter(x=list(range(df.shape[0])), y=df[col], mode='lines', name=col))
    line_plot.update_layout(width=600, height=400, title="각 변수 별 추세선")

    # select로 report의 각 페이지에 실릴 내용들 나열
    view = dp.Blocks(
        dp.Page(
            title="데이터 원본",
            blocks=[
                dp.DataTable(data, label="데이터 원본"),  # 1페이지 : 데이터 테이블
            ]
        ),
        dp.Page(
            title="그래프 시각화",
            blocks=[
                dp.Select(
                    dp.Plot(hist_plot, label="데이터 분포 확인하기"),  # 2페이지 : 히스토그램 그래프
                    dp.Plot(corr_subplot, label="상관관계 히트맵"),  # 3페이지 : 히트맵 그래프
                    dp.Plot(boxplot, label="박스 플롯으로 이상치 확인하기"),  # 4페이지 : 박스 플롯
                    dp.Plot(line_plot, label="라인 플롯 확인하기"),  # 5페이지 : 라인 플롯
                )
            ]
        )
    )

    # html 파일로 리포트 파일 생성
    dp.save_report(view, os.path.join(save_dir, f"AMS_IDX_{idx}.html"))
    
def login():
    """
    다른 api들에 요청을 보낼 때 필요한 권한을 챙기기 위해 로그인 진행 후 세션을 리턴
    
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
    """
    예측 프로세스가 실행중임을 알릴 수 있는 API 주소에 지속적으로 ok sign 전달
    
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