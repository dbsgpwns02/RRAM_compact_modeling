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

# (이 파일 상단에 load_iv_csv, build_features_paper, NARX_Paper가
#  정의되어 있거나 import 되었다고 가정합니다.)

# [필수] train.py와 동일한 build_features_paper 함수가 필요합니다.
# (파일이 분리되어 있다면)
try:
    from build_feature_v2 import build_features_paper
except ImportError:
    print("WARNING: build_feature_v3.py를 찾을 수 없어, 내부 함수를 사용합니다.")
    # build_features_paper 함수가 이 파일 내에 정의되어 있어야 함
    pass
    
try:
    from narx_paper import NARX_Paper
except ImportError:
    print("WARNING: narx_paper.py를 찾을 수 없어, 내부 클래스를 사용합니다.")
    # NARX_Paper 클래스가 이 파일 내에 정의되어 있어야 함
    pass

# ... (경로, 파라미터 정의는 그대로 둠) ...
HERE = Path().resolve()
DATA_DIR = (HERE / "../../data").resolve()
FIGURE_DIR = (HERE / "../FIG").resolve() # 로그/CSV/그래프/VA파일 저장 위치
EXPORT_DIR = (HERE / "../VA").resolve()
TRI_CSV = DATA_DIR / "IV_RRAM_TriD_35_1_real_new2.csv" # 훈련용
CSV_COLS = ["State", "Voltage", "Current"]
ny = 5
Z_DIM = 64
EPOCHS_STEP1 = 10000
EPOCHS_STEP2 = 10000
BATCH_SIZE = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_STEPS = 40
R_clip_min=100.0

# (load_iv_csv 함수가 여기에 정의되어 있다고 가정)
# (build_features_paper 함수가 여기에 정의되어 있다고 가정)
# (NARX_Paper 클래스가 여기에 정의되어 있다고 가정)

def load_iv_csv(path: Path):
    df = pd.read_csv(path, header=None, names=CSV_COLS)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    State = df["State"].to_numpy(dtype=np.float32)
    V = df["Voltage"].to_numpy(dtype=np.float32)
    I = df["Current"].to_numpy(dtype=np.float32)
    print(f"[load] {path.name} -> State:{State.shape} V:{V.shape} I:{I.shape}")
    return State, V, I
def set_seed(seed):
    """모든 무작위성을 통제하는 함수"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# [train.py 또는 main_v2.py의 main() 함수]

def main():
    # [!] 재현성을 위해 시드 고정
    SEED_VALUE = 42
    set_seed(SEED_VALUE) 
    print(f"[main] Random seed fixed to {SEED_VALUE}")
    
    if os.path.exists(FIGURE_DIR):
        pass
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print(f"[main] Device set to: {device}")

    # =========================================================
    # 1. 훈련 (TRI_CSV 사용)
    # =========================================================
    print(f"\n[Training] Loading {TRI_CSV.name} for training...")
    State_tri, V_tri, I_tri = load_iv_csv(TRI_CSV)
    
    # --- [!] 데이터 증강 로직 제거 ---
    # (V_tri, State_tri, I_tri는 원본 CSV 데이터만 사용)
    
    # '증강 안 된' 원본 데이터로 특징 생성
    # [!] narx_paper_v2.py를 사용하므로, build_feature_v3.py를 import해야 함
    X_tri, Y_R_log_tri, Y_digital_tri, V_tri_aligned, I_tri_aligned = build_features_paper(State_tri, V_tri, I_tri, ny=ny)

    print(f"\n[Training] Fitting StandardScaler...")
    scX = StandardScaler().fit(X_tri)
    X_tri_scaled = scX.transform(X_tri)

    ds_all = TensorDataset(
        torch.from_numpy(X_tri_scaled).to(device),
        torch.from_numpy(Y_R_log_tri).to(device),
        torch.from_numpy(Y_digital_tri).to(device)
    )
    dl_all = DataLoader(ds_all, batch_size=BATCH_SIZE, shuffle=True)
    
    # [!] 모델 정의 (NARX_Paper_v2 사용)
    model = NARX_Paper(in_dim=X_tri.shape[1]).to(device)

    # --- Step 1: State Module ---
    print(f"\n[Training Step 1] Training State Module... ({EPOCHS_STEP1} epochs)")
    model.head_hr.requires_grad_(False)
    model.head_lr.requires_grad_(False)
    model.head_ref.requires_grad_(True)
    model.z_backbone.requires_grad_(True)
    
    opt1 = optim.Adam(list(model.z_backbone.parameters()) + list(model.head_ref.parameters()), lr=1e-3)
    loss_fn1 = nn.BCEWithLogitsLoss()
    
    for ep in range(1, EPOCHS_STEP1 + 1):
        for xb, _, yb_digital in dl_all:
            r_ref_logit, _, _ = model(xb)
            loss1 = loss_fn1(r_ref_logit, yb_digital)
            opt1.zero_grad(); loss1.backward(); opt1.step()
        if ep % 1000 == 0 or ep == 1:
            print(f" [Step 1 Epoch {ep}] State Loss (BCE): {loss1.item():.4e}")

            
    # --- Step 2: R Modules (z_backbone 미세 조정) ---
    print(f"\n[Training Step 2] Training R Modules (Finetuning z_backbone)... ({EPOCHS_STEP2} epochs)")
    model.head_ref.requires_grad_(False) # Alpha 예측기는 동결
    model.head_hr.requires_grad_(True)
    model.head_lr.requires_grad_(True)
    
    # [!] z_backbone 동결 해제 (미세 조정을 위해)
    model.z_backbone.requires_grad_(True) 

    # '데이터 분리' 방식 (원래 방식)
    # [!] '증강 안 된' 원본 데이터만 포함된 X_tri_scaled 사용
    hrs_mask = (Y_digital_tri.squeeze() == 1)
    lrs_mask = (Y_digital_tri.squeeze() == 0)
    
    X_hrs_all = X_tri_scaled[hrs_mask]
    Y_R_log_hrs_all = Y_R_log_tri[hrs_mask]
    X_lrs_all = X_tri_scaled[lrs_mask]
    Y_R_log_lrs_all = Y_R_log_tri[lrs_mask]
    
    dl_hrs_train = DataLoader(TensorDataset(torch.from_numpy(X_hrs_all).to(device), torch.from_numpy(Y_R_log_hrs_all).to(device)), batch_size=BATCH_SIZE, shuffle=True)
    dl_lrs_train = DataLoader(TensorDataset(torch.from_numpy(X_lrs_all).to(device), torch.from_numpy(Y_R_log_lrs_all).to(device)), batch_size=BATCH_SIZE, shuffle=True)
    
    # [!] 옵티마이저에 z_backbone 파라미터 다시 추가
    opt2 = optim.Adam(
        list(model.z_backbone.parameters()) + 
        list(model.head_hr.parameters()) +
        list(model.head_lr.parameters()),
        lr=1e-4 # (미세 조정이므로 1e-4 유지)
    )
    
    loss_fn2 = nn.MSELoss() 

    for ep in range(1, EPOCHS_STEP2 + 1):
        loss_hr_val, loss_lr_val = 0.0, 0.0
        
        # head_hr 훈련 (z_backbone도 함께 미세 조정됨)
        for xb_hrs, yb_r_log_hrs in dl_hrs_train:
            _, log_hr_pred, _ = model(xb_hrs) 
            loss_hr = loss_fn2(log_hr_pred, yb_r_log_hrs)
            opt2.zero_grad(); loss_hr.backward(); opt2.step()
            loss_hr_val = loss_hr.item()
            
        # head_lr 훈련 (z_backbone도 함께 미세 조정됨)
        for xb_lrs, yb_r_log_lrs in dl_lrs_train:
            _, _, log_lr_pred = model(xb_lrs) 
            loss_lr = loss_fn2(log_lr_pred, yb_r_log_lrs)
            opt2.zero_grad(); loss_lr.backward(); opt2.step()
            loss_lr_val = loss_lr.item()
            
        if ep % 1000 == 0 or ep == 1:
            print(f" [Step 2 Epoch {ep}] HR Loss: {loss_hr_val:.4e}, LR Loss: {loss_lr_val:.4e}")
            
    print("\n[Training] Training Complete.")
    
    # =========================================================
    # 2. 훈련된 파라미터 계산 (R_ref_minmax)
    # =========================================================
    print("\n[Post-Process] Calculating R_ref min/max (required for eval)...")
    model.eval()
    with torch.no_grad():
        X_all_torch = torch.from_numpy(X_tri_scaled).to(device)
        r_ref_all, _, _ = model(X_all_torch)
        R_ref_min = r_ref_all.min().item()
        R_ref_max = r_ref_all.max().item()
        R_ref_minmax = (R_ref_min, R_ref_max)
        print(f"[info] R_ref Min: {R_ref_min:.4f}, R_ref Max: {R_ref_max:.4f}")

    # =========================================================
    # 3. 훈련 결과물 파일로 저장
    # =========================================================
    CHECKPOINT_PATH = 'checkpoint.pth'
    print(f"\n[Save] Saving checkpoint to '{CHECKPOINT_PATH}'...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_scX': scX,
        'R_ref_minmax': R_ref_minmax,
        'ny': ny,
        'in_dim': X_tri.shape[1]
    }, CHECKPOINT_PATH)
    print(f"[Save] Checkpoint saved successfully.")
    

if __name__ == "__main__":
    main()