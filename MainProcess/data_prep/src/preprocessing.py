import numpy as np
import pandas as pd
from .functions import db_table_to_csv, is_sorted_ascending

def slicing_data(tablename: str, start_date: str, end_date: str) -> pd.DataFrame:
    """시작 일자와 끝 일자를 기준으로 데이터를 슬라이싱한다.
    
    ## Input:
    - tablename : db에 등록된 table 이름
    - start_date : 슬라이싱 하려는 데이터 시작점
    - end_date : 슬라이싱 하려는 데이터의 끝 점
    
    ## Output:
    - df : 슬라이싱된 데이터프레임을 리턴
    """
    
    flag = False
    df = db_table_to_csv(table_name=tablename, including_index=False)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            flag = True
            break
        
    if flag:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df['등록일'] = pd.to_datetime(df['등록일'])
        df = df[(start_date <= df['등록일']) & (df['등록일'] <= end_date)]
    else:
        print("Dataframe has no any datetime column. It will return original dataframe.")
        return df
    
    return df

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

def split_left_right(df: pd.DataFrame, rename_cols: bool = False, transform_unit: bool = False):
    """데이터 프레임을 왼팔, 오른팔 데이터로 각각 분할하여 리턴합니다.
    
    ## Input:
    - df : 작업할 대상 데이터 프레임입니다.
    - rename_cols : 만약 컬럼 명이 한글로 변환되지 않은 상태라면, 변환을 진행하기 위한 컬럼명 변환 여부입니다. 기본값은 `False`입니다.
    - transform_unit : 만약 단위 변환이 되지 않은 상태라면, 변환을 진행하기 위한 단위 변환 여부입니다. 기본값은 `False`입니다.
    
    ## Output:
    - left_df : 왼팔 데이터셋입니다.
    - right_df : 오른팔 데이터셋입니다.
    """
    
    if rename_cols:
        db_korean_cols = db_table_to_csv("CONT_COMP_FAC_DATA_ONE", including_index=False).columns
        df.rename(columns={origin_col : rename_col for origin_col, rename_col in zip(df.columns, db_korean_cols)}, inplace=True)
        
    if transform_unit:
        factors = [10, 10, 10, 10, 1000, 100, 10, 1000, 100, 10, 10]*2
        keys = [key for key in list(df.keys()) if key not in ["카운트-왼팔", "카운트-오른팔", "row index", "등록일"]]
        for factor, key in zip(factors*2, keys):
            new_values = []
            for val in df[key]:
                new_values.append(int(val/factor))
            df[key] = new_values
    
    left_cols = [col for col in df.columns if "-왼팔" in col] + ['등록일']
    right_cols = [col for col in df.columns if "-오른팔" in col] + ['등록일']

    left_data = df[left_cols]
    right_data = df[right_cols]

    left_data.rename(columns={col : col.replace("-왼팔", "") for col in left_data.columns if "-왼팔" in col}, inplace=True)
    right_data.rename(columns={col : col.replace("-오른팔", "") for col in right_data.columns if "-오른팔" in col}, inplace=True)
    
    return left_data, right_data

def set_label_with_boring_location(df: pd.DataFrame, loc_list: list) -> pd.DataFrame:
    """황삭, 정삭, 정삭1, 정삭2, 종료지점, 대기지점에 따라 데이터에 라벨값을 부여합니다.
    
    - 1구간 : 대기 위치 ~ 황삭 전
    - 2구간 : 황삭 ~ 정삭 전
    - 3구간 : 정삭 ~ 정삭1 전
    - 4구간 : 정삭1 ~ 정삭2 전
    - 5구간 : 정삭2 ~ 종료 전
    
    ## Input:
    - df : 작업할 데이터프레임입니다. 이 때, 해당 데이터프레임의 경우 왼팔, 혹은 오른팔로 분할된 상태여야 합니다.
    - loc_list : 각 구간 별 위치를 담은 리스트입니다. 
        + 이 때, 리스트의 경우 `[대기지점, 황삭, 정삭, 정삭1, 정삭2, 종료지점]`의 6개 포인트로 구성되어야 합니다.
        
        ```python
        # e.g.
        location_list = [90, 222, 259, 288, 294, 301.5]
        
        ```
        
    ## Output:
    - df : 구간(라벨값)이 추가된 데이터프레임을 리턴합니다.
    """
    if not is_sorted_ascending(loc_list):
        loc_list = sorted(loc_list)
    loc_list.insert(0, 0)
    
    labels = [0, 1, 2, 3, 4, 5]
    df['보링, 핀 이동구간'] = pd.cut(df['보링 가공 전,후 위치'], bins=loc_list, labels=labels, right=False)
    df = df.loc[df['보링, 핀 이동구간'] != 0].reset_index(drop=True)
    
    return df

