#읽기 증강 삭제 버전
import numpy as np
import pandas as pd
from pathlib import Path
import os
import shutil
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import random # 데이터 증강용
ny = 5
Z_DIM = 64
EPOCHS_STEP1 = 10000
EPOCHS_STEP2 = 10000
BATCH_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_STEPS = 40
R_clip_min=100.0 # <<< build_features_paper에서 사용하기 위해 전역으로 이동

def build_features_paper(State, V, I, ny=5):
    """
    [근본적 수정 v2]
    '읽기 교란' 로직이 'Pristine 증강 데이터'를 오염시키는 문제를 해결.
    
    [!] '타겟 상태(Y_digital)'는 이전 상태를 따르도록 유지 (Step 1 안정화)
    [!] '타겟 저항(Y_R_log)'은 V<0.6V여도 '현재 상태'를 따르도록 수정 (Step 2 스파이크 해결)
    """
    print(f"[build-FIX-v2] Building features (R_target fix)...")
    dV = np.diff(V, prepend=V[0])
    dir_ = np.sign(dV)
    rate = np.abs(dV)
    eps = 1e-12
    I_safe = np.where(np.abs(I) < eps, np.sign(I) * eps, I)
    
    # [!] R_clip_min을 전역 변수에서 가져옴
    R_all = np.clip(V / I_safe, R_clip_min, 1e8) 
    
    R_digitized = State
    X_rows, Y_R_log_rows, Y_digital_rows = [], [], []
    V_aligned, I_aligned = [], []
    
    V_FILTER_THRESHOLD = 0.01
    V_READ_THRESHOLD = 0.6 

    for t in range(ny, len(V)):
        v_t = V[t]
        
        if np.abs(v_t) < V_FILTER_THRESHOLD:
            continue
            
        p_t = [dir_[t], rate[t]]
        
        # 2. [수정] '읽기' 구간 (0.01V < |V| < 0.6V)
        if np.abs(v_t) < V_READ_THRESHOLD:
            # 타겟 상태(Y_digital)는 '이전 상태'로 강제 (Step 1 안정화)
            Y_digital_rows.append(R_digitized[t - 1])
            
            # [!] [핵심 수정] 타겟 저항(Y_R_log)은 '현재 저항'을 그대로 학습
            # (Pristine 증강 데이터(R=100k)가 덮어쓰기(R=100)되는 것을 방지)
            Y_R_log_rows.append(np.log(R_all[t]))
            
            # 입력 히스토리(X)는 노이즈 없이 (Pristine/Cycled)
            r_hist_digital = [R_digitized[t - k] for k in range(1, ny + 1)]

        # 3. '쓰기' 구간 (|V| >= 0.6V)
        else:
            # "쓰기" 중에는 현재 상태와 저항을 그대로 학습 (기존과 동일)
            Y_digital_rows.append(R_digitized[t])
            Y_R_log_rows.append(np.log(R_all[t]))
            r_hist_digital = [R_digitized[t - k] for k in range(1, ny + 1)]

        # 공통 로직: X벡터 생성
        X_rows.append([v_t] + p_t + r_hist_digital)
        V_aligned.append(v_t)
        I_aligned.append(I[t])

    # ... (for 루프 종료) ...
    
    X = np.asarray(X_rows, dtype=np.float32)
    Y_R_log = np.asarray(Y_R_log_rows, dtype=np.float32).reshape(-1, 1)
    Y_digital = np.asarray(Y_digital_rows, dtype=np.float32).reshape(-1, 1)
    
    # [!] 누락된 V_aligned_np 변환 라인 추가
    V_aligned_np = np.asarray(V_aligned, dtype=np.float32)
    
    I_aligned_np = np.asarray(I_aligned, dtype=np.float32)
    
    print(f"[build-FIX] V=0 Filtered. Original: {len(V)-ny}, Filtered: {len(X)}")
    print(f"[build-FIX] X: {X.shape}, Y_R_log: {Y_R_log.shape}, Y_digital: {Y_digital.shape}")
    return X, Y_R_log, Y_digital, V_aligned_np, I_aligned_np # <<< [정상 동작]
