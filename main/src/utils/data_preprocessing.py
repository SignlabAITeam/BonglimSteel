"""
functions
"""

import numpy as np
import pandas as pd
import pingouin as pg

def get_abno_value(df: pd.DataFrame) -> int:
    """데이터 프레임의 결측치를 제거하고, 모든 열에 대한 이상치 개수를 return
    
    ## Input:
    - df : 작업할 데이터 프레임을 인자로 받습니다.
    
    ## Output:
    - sum of abno_cnts : 각 열에서 탐지된 이상값의 개수를 합산하여 리턴턴
    
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
    
def remove_outlier_with_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """IQR 방식을 활용해 이상치를 제거합니다.
    
    ## Input:
    - df : 작업할 데이터 프레임
    
    ## Output:
    - df : IQR방식으로 이상치 제거가 완료된 데이터프레임을 리턴합니다.
    """
    
    for col in df.select_dtypes(exclude=["object", "datetime64[ns]"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        
        IQR = q3 - q1

        upper_limit = q3 + IQR*1.5
        lower_limit = q1 - IQR*1.5
        
        drop_upper = df[df[col] > upper_limit].index
        drop_lower = df[df[col] < lower_limit].index
        
        df[col] = df[col].drop(drop_upper)
        df[col] = df[col].drop(drop_lower)

    before_row_cnt = df.shape[0]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    after_row_cnt = df.shape[0]
    
    print(f"이상치를 제거하며 {before_row_cnt - after_row_cnt}개의 결측치가 발생, 제거 완료")
    
    return df

def log_transformation(df: pd.DataFrame):
    """데이터에 로그 변환을 적용합니다.
    
    이 때, 만약 **로그 변환 중에 발생한 결측치의 경우 drop한 후 return**합니다.
    
    ## Input:
    - df : 작업할 데이터프레임
    
    ## Output:
    - df : 로그 변환을 적용한 데이터 프레임을 리턴합니다.
    """
    for col in df.select_dtypes(exclude=['object', 'datetime64[ns]']).columns:
        df[col] = df[col].apply(lambda x: np.log1p(x) if x != 0 else np.log1p(x + 1e-6))
    
    before_row_cnt = df.shape[0]
    df.dropna(inplace=True)
    after_row_cnt = df.shape[0]
    
    print(f"로그 변환 중 {before_row_cnt - after_row_cnt}개의 결측치가 발생, 제거 완료")
    return df