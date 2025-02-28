import numpy as np
import pandas as pd
import pingouin as pg
import plotly.graph_objects as go

from importlib import import_module
from plotly.subplots import make_subplots
from ...database.base import SessionLocal

def adjust_r2_score(n, k, r2):
    """r2 스코어와 표본의 수, 특성의 수를 받아 adjust r2 score를 계산합니다.
    
    ## Input:
    - n : 표본의 수
    - k : feature의 개수
    - r2 : 기존의 r2 score
    
    ## Output:
    - adjust r2 score : 계산된 adjust r2 score를 리턴합니다.
    
    """
    return 1 - (((1 - r2)*(n - 1)) / n - k - 1)

def is_sorted_ascending(lst: list) -> bool:
    """리스트를 받아 오름차순으로 정리되지 않았다면 False를 반환합니다.
    
    ## Input:
    - lst : 오름차순 점검 대상 리스트입니다.
    
    ## Output:
    - True or False : 오름차순 정렬 여부를 bool type으로 return합니다.
    """
    return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))

def db_table_to_csv(table_name: str, including_index=True) -> pd.DataFrame:
    """DB 테이블로부터 데이터를 가져와 데이터프레임으로 리턴
    
    ## Input:
    - table_name : DB에 등재된 테이블 이름. string 형태로 작성할 것
    - including_index : index 컬럼 포함 여부
    
    ## Output:
    - dataframe : 해당 테이블로 만든 데이터프레임
    """
    
    # prevent circular import problem
    table_module = import_module("MainProcess.database.schema.models")
    # 테이블을 schema.models 라는 모듈에서 가져온다.
    table_instance = getattr(table_module, table_name if table_name.isupper() else table_name.upper())
    
    db = SessionLocal()
    
    if including_index:
        columns = table_instance.__table__.columns
    else:
        columns = table_instance.__table__.columns[1:]
    
    # 영어 컬럼명이 아닌, 코멘트가 달린 한글 명으로 컬럼명을 세팅한다.
    all_data = db.query(table_instance).all()
    data = {col.info.get('comment'):[] for col in columns}
    for row in all_data:
        for col in columns:
            data[col.info.get("comment")].append(getattr(row, col.name))
    
    # cont_comp_fac_data_one 테이블이나 cont_comp_fac_data_two 테이블의 경우 단위 변환이 필요하기에
    if table_instance.__tablename__ in ["cont_comp_fac_data_one", "cont_comp_fac_data_two"]:
        factors = [10, 10, 10, 10, 1000, 100, 10, 1000, 100, 10, 10]*2
        keys = [key for key in list(data.keys()) if key not in ["카운트-왼팔", "카운트-오른팔", "row index", "등록일"]]
        for factor, key in zip(factors*2, keys):
            new_values = []
            for val in data[key]:
                new_values.append(int(val/factor))
            data[key] = new_values

    # 데이터 프레임 형태로 만들어서 반환
    df = pd.DataFrame(data)
    db.close()
    
    return df

def plotting_hist_subplots(data: pd.DataFrame):
    """데이터프레임의 각 열에 대한 히스토그램을 subplot으로 시각화
    
    ## Input:
    - data : 시각화할 데이터프레임을 받습니다.
    
    ## Return:
    - hist plot이 할당된 subplot 객체를 return합니다.
    """

    cols = data.select_dtypes(exclude='datetime64[ns]').columns.tolist()
    n_row_col = int(np.sqrt(data.shape[1])) + 1
    fig = make_subplots(
        rows=n_row_col, 
        cols=n_row_col,
        row_heights=[0.3]*n_row_col,
        column_widths=[0.3]*n_row_col,
        subplot_titles=tuple(f'{title}' for title in cols),
        )

    column_index = 0
    for row_num in range(1, n_row_col + 1):
        for col_num in range(1, n_row_col + 1):
            if column_index < len(cols):
                fig.add_trace(
                    go.Histogram(
                        x=data[cols[column_index]],  # 각 컬럼 데이터
                        name=cols[column_index]
                    ), row=row_num, col=col_num
                )
                    
            column_index += 1

    fig.update_layout(
        width=1200, 
        height=1200,
        margin_l=50
        )
    
    fig.update_annotations(font=dict(size=11))
    
    return fig

def get_abno_value(df: pd.DataFrame) -> int:
    """데이터 프레임의 결측치를 제거하고, 모든 열에 대한 이상치 개수를 return
    
    ## Input:
    - df : 작업할 데이터 프레임을 인자로 받습니다.
    
    ## Output:
    - sum of abno_cnts : 각 열에서 탐지된 이상값의 개수를 합산하여 리턴
    
    """
    df = df.dropna()
    abno_cnts = []
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = (q3 - q1)
        
        minimum = q1 - IQR*1.5
        maximum = q3 + IQR*1.5
        
        abno_cnt = 0
        for val in df[col].values:
            if val < minimum:
                abno_cnt += 1
            elif maximum < val:
                abno_cnt += 1
        abno_cnts.append(abno_cnt)
    
    return sum(abno_cnts)

def get_corr_value(df: pd.DataFrame, target: str):
    """특정 열의 상관계수를 구한 후 해당 상관관계의 평균값을 return
    
    ## Input:
    - df : 작업할 대상 데이터 프레임
    - target : 차후 모델 학습 시 사용할 예측 대상
    
    ## Output:
    - corr mean value or 0 : 상관계수를 구할 수 있다면, 상관계수의 평균값을 리턴하고 아니라면 0값을 리턴합니다.
    """
    
    try:
        pc = pg.pairwise_corr(data=df, method="pearson").round(3)
        new_pc = pc[pc['Y'] == target].reset_index(drop=True)
        
        return new_pc['r'].mean()
    except:
        return 0

