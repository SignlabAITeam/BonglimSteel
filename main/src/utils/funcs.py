"""
데이터 시각화 및 기타 유틸 함수들
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from database.base import SessionLocal

def db_table_to_csv(table_name, including_index=True) -> pd.DataFrame:
    """DB 테이블로부터 데이터를 가져와 데이터프레임으로 리턴
    
    ## Input:
    - table_name : DB에 등재된 테이블 이름
    - including_index : index 컬럼 포함 여부
    ## Output:
    - dataframe : 해당 테이블로 만든 데이터프레임
    """
    db = SessionLocal()
    
    if including_index:
        columns = table_name.__table__.columns
    else:
        columns = table_name.__table__.columns[1:]
    all_data = db.query(table_name).all()
    data = {col.info.get('comment'):[] for col in columns}
    for row in all_data:
        for col in columns:
            data[col.info.get("comment")].append(getattr(row, col.name))
    
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