# [train.py에 포함된 build_features_paper 함수]
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
    print(f"[build-FIX] Building features (No Read/Write split)...")
    dV = np.diff(V, prepend=V[0])
    dir_ = np.sign(dV)
    rate = np.abs(dV)
    eps = 1e-12
    I_safe = np.where(np.abs(I) < eps, np.sign(I) * eps, I)
    
    # R_clip_min 하드코딩 (또는 인자로 받음)
    R_clip_min = 100.0
    R_all = np.clip(V / I_safe, R_clip_min, 1e8) 
    
    R_digitized = State
    X_rows, Y_R_log_rows, Y_digital_rows = [], [], []
    V_aligned, I_aligned = [], []
    
    V_FILTER_THRESHOLD = 0.01

    for t in range(ny, len(V)):
        v_t = V[t]
        if np.abs(v_t) < V_FILTER_THRESHOLD:
            continue
            
        p_t = [dir_[t], rate[t]]
        
        # [핵심] 읽기/쓰기 구분 없이 현재 상태/저항을 그대로 학습
        Y_digital_rows.append(R_digitized[t])
        Y_R_log_rows.append(np.log(R_all[t]))
        r_hist_digital = [R_digitized[t - k] for k in range(1, ny + 1)]

        X_rows.append([v_t] + p_t + r_hist_digital)
        V_aligned.append(v_t)
        I_aligned.append(I[t])

    X = np.asarray(X_rows, dtype=np.float32)
    Y_R_log = np.asarray(Y_R_log_rows, dtype=np.float32).reshape(-1, 1)
    Y_digital = np.asarray(Y_digital_rows, dtype=np.float32).reshape(-1, 1)
    V_aligned_np = np.asarray(V_aligned, dtype=np.float32)
    I_aligned_np = np.asarray(I_aligned, dtype=np.float32)
    
    print(f"[build-FIX] Filtered: {len(X)}")
    return X, Y_R_log, Y_digital, V_aligned_np, I_aligned_np