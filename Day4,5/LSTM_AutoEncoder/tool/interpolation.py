import pandas as pd
import numpy as np
from scipy import interpolate
import os

# RGB 데이터를 CSV 파일로부터 로드
df = pd.read_csv('/home/bg/lstm_autoencoder_trajectory/clustered/expert_20_stand_clus7.csv')

# 원본 RGB 값
original_rgb = df.values

# 원본 데이터 길이
original_length = original_rgb.shape[0]

# 목표 길이
target_length = 2416

# 선형 보간 함수 생성
func = interpolate.interp1d(np.arange(original_length), original_rgb, axis=0)

# 새로운 인덱스 (선형으로 간격을 조절)
new_index = np.linspace(0, original_length-1, target_length)

# 보간을 수행하여 새로운 RGB 값을 얻음
new_rgb = func(new_index)

# 결과를 DataFrame으로 변환
df_new = pd.DataFrame(new_rgb, columns=df.columns)

# 저장할 폴더 이름
folder_name = 'interpolate'

# 해당 폴더가 없다면 생성
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# DataFrame을 새로운 CSV 파일로 저장
df_new.to_csv(f'{folder_name}/new_expert_gps.csv', index=False)
