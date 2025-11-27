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
    [수정] '읽기 교란'을 막기 위해 저전압 구간(Read)의 타겟을
    현재(t)가 아닌 이전(t-1) 상태/저항으로 강제한다.
    """
    print(f"[build-AUG] Building features with Read-Disturb Augmentation...")
    dV = np.diff(V, prepend=V[0])
    dir_ = np.sign(dV)
    rate = np.abs(dV)
    eps = 1e-12
    I_safe = np.where(np.abs(I) < eps, np.sign(I) * eps, I)
    R_all = np.clip(V / I_safe, 1e2, 1e8)
    R_digitized = State
    X_rows, Y_R_log_rows, Y_digital_rows = [], [], []
    V_aligned, I_aligned = [], []
    
    V_FILTER_THRESHOLD = 0.01
    V_READ_THRESHOLD = 0.6 # '읽기'로 간주할 전압 상한 (0.5V 펄스 포함)

    for t in range(ny, len(V)):
        v_t = V[t]
        
        # 1. 0.01V 미만은 노이즈로 간주하고 완전히 스킵
        if np.abs(v_t) < V_FILTER_THRESHOLD:
            continue
            
        p_t = [dir_[t], rate[t]]
        
        # 2. [수정] '읽기' 구간 (0.01V < |V| < 0.6V)
        if np.abs(v_t) < V_READ_THRESHOLD:
            # "읽기" 중에는 상태가 변하면 안 됨
            # 타겟 상태(Y_digital)를 현재(t)가 아닌 이전(t-1) 상태로 강제
            Y_digital_rows.append(R_digitized[t - 1])
            
            # "읽기" 중에는 저항이 변하면 안 됨
            # 타겟 저항(Y_R_log)을 현재(t)가 아닌 이전(t-1) 저항으로 강제
            # (R_all[t-1]이 0이 되는 것을 방지하기 위해 R_all[t]를 최소값으로 사용)
            log_r_safe = np.log(R_all[t - 1] if R_all[t - 1] > R_clip_min else R_all[t])
            Y_R_log_rows.append(log_r_safe)
            
            # 히스토리(X)는 10% 확률로 노이즈를 주입 (기존 상태 고착 문제 해결용)
            r_hist_digital = []
            for k in range(1, ny + 1):
                true_state = R_digitized[t - k]
                if random.random() < 0.1: # 10% 노이즈
                    r_hist_digital.append(1.0 - true_state)
                else:
                    r_hist_digital.append(true_state)

        # 3. '쓰기' 구간 (|V| >= 0.6V)
        else:
            # "쓰기" 중에는 현재 상태와 저항을 그대로 학습
            Y_digital_rows.append(R_digitized[t])
            Y_R_log_rows.append(np.log(R_all[t]))
            r_hist_digital = [R_digitized[t - k] for k in range(1, ny + 1)]

        # 공통 로직: X벡터 생성
        X_rows.append([v_t] + p_t + r_hist_digital)
        V_aligned.append(v_t)
        I_aligned.append(I[t])

    X = np.asarray(X_rows, dtype=np.float32)
    Y_R_log = np.asarray(Y_R_log_rows, dtype=np.float32).reshape(-1, 1)
    Y_digital = np.asarray(Y_digital_rows, dtype=np.float32).reshape(-1, 1)
    V_aligned_np = np.asarray(V_aligned, dtype=np.float32)
    I_aligned_np = np.asarray(I_aligned, dtype=np.float32)
    
    print(f"[build-AUG] V=0 Filtered. Original: {len(V)-ny}, Filtered: {len(X)}")
    print(f"[build-AUG] X: {X.shape}, Y_R_log: {Y_R_log.shape}, Y_digital: {Y_digital.shape}")
    return X, Y_R_log, Y_digital, V_aligned_np, I_aligned_np
