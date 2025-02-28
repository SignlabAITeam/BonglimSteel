import os
import numpy as np
import pandas as pd
import datapane as dp
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from ...data_prep.src.preprocessing import split_left_right
from ...data_prep.src.functions import plotting_hist_subplots

def make_report(df: pd.DataFrame, idx: int, save_dir: str):
    """간단한 시각화 자료들을 report로 생성하여 공유 폴더에 저장
    
    ## Input:
    - data : 시각화 대상 dataframe
    - idx : 작업 대상 index
    - save_dir : report.html 파일이 저장될 경로
    
    ## Output:
    - return none
    """
    
    left_data, right_data = split_left_right(df)
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
            go.Box(y=df[col].values, name=col)
        )
    boxplot.update_layout(width=600, height=600, title='각 변수 별 박스플롯')

    # 라인플롯 시각화
    line_plot = go.Figure()
    for col in df.columns:
        line_plot.add_trace(go.Scatter(x=df['등록일'], y=df[col], mode='lines', name=col))
    line_plot.update_layout(width=600, height=400, title="각 변수 별 추세선")

    # select로 report의 각 페이지에 실릴 내용들 나열
    view = dp.Blocks(
        dp.Page(
            title="데이터 원본",
            blocks=[
                dp.DataTable(df, label="데이터 원본"),  # 1페이지 : 데이터 테이블
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