# plot_dc.py
# 훈련된 모델(checkpoint.pth)을 불러와 "DC가 잘 맞는 버전"과
# 동일한 스타일 (True vs Pred)로 DC 플롯을 생성합니다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# [필수] narx_paper.py에서 모델 클래스를 가져옵니다.
# (이 스크립트는 'narx_paper.py'가 '원래'의
#  "DC가 잘 맞는 버전"으로 복구되었다고 가정합니다)
try:
    from narx_paper import NARX_Paper
except ImportError:
    print("ERROR: narx_paper.py 파일을 찾을 수 없습니다.")
    sys.exit(1)

# ------------------------------------------------------------
# 경로 & 설정 (train.py와 동일해야 함)
# ------------------------------------------------------------
HERE = Path().resolve()
DATA_DIR = (HERE / "../../data").resolve() 
FIGURE_DIR = (HERE / "../DC_FIG").resolve()
CHECKPOINT_PATH = 'checkpoint.pth'
TRI_CSV = DATA_DIR / "IV_RRAM_TriD_35_1_real_new2.csv" # 훈련용
CSV_COLS = ["State", "Voltage", "Current"]

# --- 모델 및 훈련 파라미터 (train.py와 동일) ---
ny = 5
Z_DIM = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
R_clip_min = 100.0

# ============================================================
# [!] "DC가 잘 맞는 버전"의 함수들 (복사) [!]
# ============================================================

# --- 1. CSV 읽기 ---
def load_iv_csv(path: Path):
    df = pd.read_csv(path, header=None, names=CSV_COLS)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    State = df["State"].to_numpy(dtype=np.float32)
    V = df["Voltage"].to_numpy(dtype=np.float32)
    I = df["Current"].to_numpy(dtype=np.float32)
    print(f"[load] {path.name} -> State:{State.shape} V:{V.shape} I:{I.shape}")
    return State, V, I

# --- 2. Feature 생성 ---
# [!] 이 함수는 반드시 train.py의 build_features_paper와 100% 동일해야 합니다.
#     (narx_paper_2step_final.py의 '읽기 교란 없는' 버전으로 가정)
def build_features_paper(State, V, I, ny=5):
    print(f"[build] Building features using ground-truth 'State' column...")
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

    for t in range(ny, len(V)):
        v_t = V[t]
        if np.abs(v_t) < V_FILTER_THRESHOLD:
            continue
        p_t = [dir_[t], rate[t]]
        r_hist_digital = [R_digitized[t - k] for k in range(1, ny + 1)]
        X_rows.append([v_t] + p_t + r_hist_digital)
        Y_R_log_rows.append(np.log(R_all[t]))
        Y_digital_rows.append(R_digitized[t])
        V_aligned.append(v_t)
        I_aligned.append(I[t])

    X = np.asarray(X_rows, dtype=np.float32)
    Y_R_log = np.asarray(Y_R_log_rows, dtype=np.float32).reshape(-1, 1)
    Y_digital = np.asarray(Y_digital_rows, dtype=np.float32).reshape(-1, 1)
    V_aligned_np = np.asarray(V_aligned, dtype=np.float32)
    I_aligned_np = np.asarray(I_aligned, dtype=np.float32)
    
    print(f"[build] V=0 Filtered. Original: {len(V)-ny}, Filtered: {len(X)}")
    return X, Y_R_log, Y_digital, V_aligned_np, I_aligned_np

# --- 3. 플로팅 헬퍼 ---

def plot_iv_curve(V, I_true, I_pred, data_type, save_path):
    """
    [복구] Total I-V (True: 파란 점/선, Pred: 빨간 선)
    """
    plt.figure(figsize=(6, 4))
    plt.plot(V, I_true, "b.-", ms=3, label="True")
    plt.plot(V, I_pred, "r--", lw=1, label="Pred")
    plt.title(f"I-V Curve - {data_type}")
    plt.xlabel("Voltage (V)"); plt.ylabel("Current (A)")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()
    print(f"Saved: {save_path}")

def plot_iv_log_curve(V, I_true, I_pred, data_type, save_path):
    """
    [복구] Total log(I)-V (True: 빨간 점, Pred: 파란 점)
    """
    eps = 1e-30
    plt.figure(figsize=(6, 4))
    plt.plot(V, np.log10(np.abs(I_true)+eps), 'rs-', ms=3, label='True')
    plt.plot(V, np.log10(np.abs(I_pred)+eps), 'bo-', ms=2, label='Pred')
    plt.title(f"Log|I|-V - {data_type}")
    plt.xlabel("Voltage (V)"); plt.ylabel("log10 |Current| (A)")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()
    print(f"Saved: {save_path}")

def plot_hr_lr_curves(V_hrs, R_hrs_true, R_hrs_pred, V_lrs, R_lrs_true, R_lrs_pred, save_dir):
    """
    [복구] HR/LR 플롯 (True: 파란 점, Pred: 빨간 점)
    """
    plt.figure(figsize=(10, 5))
    
    # --- HR Plot ---
    plt.subplot(1, 2, 1)
    plt.scatter(V_hrs, R_hrs_true, c='blue', s=5, alpha=0.5, label="True HRS Data")
    plt.scatter(V_hrs, R_hrs_pred, c='red', s=5, alpha=0.5, label="Predicted HRS Curve")
    plt.title("Learned HRS Curve")
    plt.xlabel("Voltage (V)"); plt.ylabel("Resistance (Ohm)")
    plt.legend(); plt.yscale('log'); plt.grid(True)

    # --- LR Plot ---
    plt.subplot(1, 2, 2)
    plt.scatter(V_lrs, R_lrs_true, c='blue', s=5, alpha=0.5, label="True LRS Data")
    plt.scatter(V_lrs, R_lrs_pred, c='red', s=5, alpha=0.5, label="Predicted LRS Curve")
    plt.title("Learned LRS Curve")
    plt.xlabel("Voltage (V)"); plt.ylabel("Resistance (Ohm)")
    plt.legend(); plt.yscale('log'); plt.grid(True)

    plt.tight_layout()
    save_path = save_dir / "dc_plot_HR_LR_curves.png" # 이름 변경
    plt.savefig(save_path, dpi=200); plt.close()
    print(f"Saved: {save_path}")

# ============================================================
# Main 실행 함수
# ============================================================

def main_plot_dc():
    # 1. 체크포인트 로드
    print(f"[Load] Loading checkpoint from '{CHECKPOINT_PATH}'...")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[Error] '{CHECKPOINT_PATH}' not found. Please run train.py first.")
        sys.exit(1)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    print("[Load] Checkpoint loaded.")

    # 2. 모델 및 스케일러 복원
    ny = checkpoint['ny']
    in_dim = checkpoint['in_dim']
    model = NARX_Paper(in_dim=in_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    scX = checkpoint['scaler_scX']
    R_ref_minmax = checkpoint['R_ref_minmax']
    model.eval()
    print("[Load] Model and scaler restored successfully.")

    # --- 3. 'True' 훈련 데이터 로드 및 처리 ---
    print(f"\n[Load True Data] Loading {TRI_CSV.name} for comparison...")
    State_tri, V_tri, I_tri = load_iv_csv(TRI_CSV)
    
    # 훈련 때와 동일한 로직으로 'True' 데이터 생성
    X_tri, Y_R_log_tri, Y_digital_tri, V_tri_aligned, I_tri_aligned = \
        build_features_paper(State_tri, V_tri, I_tri, ny=ny)

    X_tri_scaled = scX.transform(X_tri) # 훈련된 스케일러 사용

    hrs_mask = (Y_digital_tri.squeeze() == 1)
    lrs_mask = (Y_digital_tri.squeeze() == 0)

    # 'True' HR/LR 값
    V_hrs_true = V_tri_aligned[hrs_mask]
    R_hrs_true = np.exp(Y_R_log_tri[hrs_mask].squeeze())
    V_lrs_true = V_tri_aligned[lrs_mask]
    R_lrs_true = np.exp(Y_R_log_tri[lrs_mask].squeeze())
    
    print("[Load True Data] True HR/LR data points extracted.")

    # --- 4. 'Pred' 데이터 생성 (Teacher Forcing) ---
    print("\n[Predict] Running Teacher Forcing prediction on TRI data...")
    model.eval()
    with torch.no_grad():
        X_all_torch = torch.from_numpy(X_tri_scaled).to(device)
        V_aligned_torch = torch.from_numpy(V_tri_aligned.reshape(-1, 1)).to(device)
        
        # 모델의 full forward pass 실행
        r_ref_all, log_hr_all, log_lr_all = model(X_all_torch)
        
        # --- Total I-V Pred ---
        R_ref_min, R_ref_max = R_ref_minmax
        if R_ref_max == R_ref_min: R_ref_max += 1e-9
        
        hr_tf = torch.exp(log_hr_all)
        lr_tf = torch.exp(log_lr_all)
        alpha_tf = (r_ref_all - R_ref_min) / (R_ref_max - R_ref_min)
        alpha_tf = torch.clamp(alpha_tf, 0.0, 1.0)
        
        R_tf_pred_torch = alpha_tf * hr_tf + (1.0 - alpha_tf) * lr_tf
        I_tf_pred_torch = V_aligned_torch / R_tf_pred_torch
        
        I_teacher_pred = I_tf_pred_torch.cpu().numpy().reshape(-1)
        
        # --- HR/LR Pred ---
        # 훈련 데이터의 HRS/LRS 입력에 대한 예측값
        X_hrs_all_torch = torch.from_numpy(X_tri_scaled[hrs_mask]).to(device)
        X_lrs_all_torch = torch.from_numpy(X_tri_scaled[lrs_mask]).to(device)
        
        R_hrs_pred = torch.exp(model(X_hrs_all_torch)[1]).cpu().numpy().squeeze()
        R_lrs_pred = torch.exp(model(X_lrs_all_torch)[2]).cpu().numpy().squeeze()
        
    print("[Predict] Prediction complete.")

    # --- 5. DC 플롯 생성 ---
    print("\n[Plotting] Saving DC plots (True vs Pred)...")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # 플롯 1: 총 I-V (True/Pred Hysteresis)
    plot_iv_curve(
        V_tri_aligned, I_tri_aligned, I_teacher_pred,
        "Teacher Forcing (TRI)", 
        FIGURE_DIR / "dc_plot_total_IV.png"
    )

    # 플롯 2: 총 log(I)-V (True/Pred Hysteresis)
    plot_iv_log_curve(
        V_tri_aligned, I_tri_aligned, I_teacher_pred,
        "Teacher Forcing (TRI)", 
        FIGURE_DIR / "dc_plot_total_logIV.png"
    )
    
    # 플롯 3 & 4: HR/LR (True: 파란 점, Pred: 빨간 점)
    plot_hr_lr_curves(
        V_hrs_true, R_hrs_true, R_hrs_pred,
        V_lrs_true, R_lrs_true, R_lrs_pred,
        FIGURE_DIR
    )
    
    print(f"\n[Done] All DC plots saved to {FIGURE_DIR}")

if __name__ == "__main__":
    main_plot_dc()