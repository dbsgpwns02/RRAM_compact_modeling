import numpy as np
import torch
V_READ_TH = 0.6
# [참고] 이 함수는 debug_sample_like.py 파일 내의
# 기존 debug_spice_sampling_va_matched_CONTINUOUS 함수를 대체합니다.

def debug_spice_sampling_va_matched_CONTINUOUS(
    model, V_data, I_data, State_data, scX, ny, R_ref_minmax,
    device, 
    Ts=1e-9, V_FILTER=0.01, 
    warmup_steps=40, INIT_W=1,
    tau_slew=2e-09,
    r_clip_min=100.0,
    save_log=True, log_path="spice_debug_va_matched_CONTINUOUS.txt",
    
    # === 연속적 모델을 위한 스무딩 파라미터 ===
    V_smooth_read=0.1,
    hyst_smooth=0.1
):
    """
    [CONTINUOUS ver. / 최종 방어 로직]
    'V_READ_TH'(0.6V)를 기준으로 READ/WRITE를 구분.
    - READ 구간 (gate_write=0): 
        1. 전류/저항(I/R)은 G_last (이전 값) 사용
        2. 상태(r_ref/alpha)도 r_ref_last_valid (이전 값) 사용
    - WRITE 구간 (gate_write=1):
        1. 전류/저항(I/R)은 신경망(ANN) 값 사용
        2. 상태(r_ref/alpha)도 신경망(ANN) 값 사용
    """
    print(f"[DEBUG-VA-CONT] Continuous Simulation (V_READ_TH={V_READ_TH} 기준, 최종 방어)")
    print(f"[DEBUG-VA-CONT] Smooth Params: V_read={V_smooth_read}, Hyst={hyst_smooth}")
    
    log_lines = [
        f"[DEBUG-VA-CONT] Continuous Simulation (V_READ_TH={V_READ_TH} 기준, 최종 방어)",
        f"Smooth Params: V_read={V_smooth_read}, Hyst={hyst_smooth}"
    ]

    model.eval()

    # ----- 시간축 구성 -----
    t_stop_corrected = (len(V_data) - 1) * Ts
    time_array = np.arange(0, t_stop_corrected + Ts/2, Ts)
    if len(time_array) != len(V_data):
      
        time_array = time_array[:len(V_data)]

    # ----- 초기 조건 -----
    V_prev_raw = float(V_data[0])
    R_last = float(r_clip_min)
    G_last = 1.0 / R_last
    k_sample = 0
    k_eff = 0
    W_hist = [float(INIT_W)] * ny

    R_ref_min, R_ref_max = R_ref_minmax
    if R_ref_max == R_ref_min:
        R_ref_max += 1e-9
        
    # [!] r_ref_last_valid의 초기값을 INIT_W (초기 상태)에 맞게 설정
    # INIT_W=1 (HRS)이면 R_ref_max, 0 (LRS)이면 R_ref_min
    if INIT_W == 1:
        r_ref_last_valid = R_ref_max
    else:
        r_ref_last_valid = R_ref_min

    # LPF 계수
    I_hold = 0.0
    if tau_slew <= 0.0:
        alpha_slew = 1.0
    else:
        alpha_slew = Ts / (tau_slew + Ts)
    alpha_slew = min(1.0, max(0.0, alpha_slew))

    # --- 반환할 리스트 ---
    all_V, all_I, all_R, all_t, all_r_ref = [], [], [], [], []
    all_I_raw = [] 
    all_alpha = []
    all_log_hr = []
    all_log_lr = []
    
    # ========================================================
    # 메인 타임 스텝 루프
    # ========================================================
    for t_idx, t_now in enumerate(time_array):
        Vt = float(V_data[t_idx])
        dV = Vt - V_prev_raw
        dir_t = np.sign(dV)
        rate_t = abs(dV)
        all_t.append(t_now)

        # -----------------------------
        # 1. 'READ' 구간의 계산 (G_last 기준)
        # -----------------------------
        I_pred_read = Vt * G_last
        R_pred_read = R_last
        # [!] '읽기' 중에는 r_ref/alpha/logHr/logLr도 이전 값을 유지
        r_ref_logit_read = r_ref_last_valid 
        alpha_read = (r_ref_last_valid - R_ref_min) / (R_ref_max - R_ref_min) # (이전 값 기준 alpha)
        log_hr_read = 0.0 # (읽기 중엔 의미 없음)
        log_lr_read = 0.0 # (읽기 중엔 의미 없음)

        # -----------------------------
        # 2. 'WRITE' 구간의 계산 (신경망 ON)
        # -----------------------------
        x_raw = np.array([Vt, dir_t, rate_t] + W_hist, dtype=np.float32).reshape(1, -1)
        x_scaled = scX.transform(x_raw)
        x_t = torch.from_numpy(x_scaled).to(device).float()

        with torch.no_grad():
            (I_t_pred, R_t_pred, _,
             r_ref_logit_t, log_hr_pred_t, log_lr_pred_t, alpha_t) = \
                model.predict_with_internals(
                    x_t,
                    torch.tensor(Vt).to(device).float(),
                    R_ref_minmax
                )

        # 신경망이 예측한 '쓰기' 후보 값
        I_pred_write = I_t_pred.item()
        R_pred_write = R_t_pred.item()
        r_ref_logit_write = r_ref_logit_t.item()
        alpha_write = alpha_t.item()
        log_hr_write = log_hr_pred_t.item()
        log_lr_write = log_lr_pred_t.item()

        # -----------------------------
        # 3. [핵심 수정] 'Soft Switch' (V_READ_TH 기준으로 모든 값 혼합)
        # -----------------------------
        gate_write = 0.5 * (1.0 + np.tanh((abs(Vt) - V_READ_TH) / V_smooth_read))

        # '읽기'(gate_write=0)면 'read'값, '쓰기'(gate_write=1)면 'write'값
        I_pred_raw = (1.0 - gate_write) * I_pred_read + gate_write * I_pred_write
        R_pred = (1.0 - gate_write) * R_pred_read + gate_write * R_pred_write
        r_ref_logit = (1.0 - gate_write) * r_ref_logit_read + gate_write * r_ref_logit_write
        alpha = (1.0 - gate_write) * alpha_read + gate_write * alpha_write
        log_hr = (1.0 - gate_write) * log_hr_read + gate_write * log_hr_write
        log_lr = (1.0 - gate_write) * log_lr_read + gate_write * log_lr_write
        
        all_I_raw.append(I_pred_raw) 
        I_hold = I_hold + alpha_slew * (I_pred_raw - I_hold)
        R_last = R_pred
        G_last = 1.0 / R_last
        
        # [!] 중요: 다음 스텝을 위해 '방어된' r_ref_logit 값을 저장
        r_ref_last_valid = r_ref_logit 

        # -----------------------------
        # 4. 상태(W_hist) 업데이트
        # -----------------------------
        W_current = W_hist[0]
        W_target = 0.5 * (1.0 + np.tanh(r_ref_logit / hyst_smooth)) # 방어된 r_ref 사용
        W_next = (1.0 - gate_write) * W_current + gate_write * W_target
        
        # 5. 상태 히스토리(W_hist) 업데이트
        if k_eff >= warmup_steps:
            W_hist = [W_next] + W_hist[:-1]

        # 6. 결과 저장 및 루프 마무리
        valid_log = (
            f"LOG t={t_now:.6e} V={Vt:+.6f} dV={dV:+.6f} | "
            f"gate_W={gate_write:.4f} | "
            f"r_ref={r_ref_logit:+.4f} W_targ={W_target:.4f} W_next={W_next:.4f} | "
            f"R={R_pred:.6e} I_raw={I_pred_raw:.6e} I_lpf={I_hold:.6e} | "
            f"alpha={alpha:+.4f} logHr={log_hr:+.4f} logLr={log_lr:+.4f}"
        )
        log_lines.append(valid_log)

        all_V.append(Vt)
        all_I.append(I_hold)
        all_R.append(R_pred)
        all_r_ref.append(r_ref_logit)
        all_alpha.append(alpha)
        all_log_hr.append(log_hr)
        all_log_lr.append(log_lr)

        k_eff += 1
        V_prev_raw = Vt
        k_sample += 1
    
    # === 루프 종료 ===

    if save_log:
        try:
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines) + "\n")
            print(f"[DEBUG-VA-CONT] Continuous log saved to {log_path}")
        except Exception as e:
            print(f"[ERROR] Could not save log file: {e}")

    return (
        np.array(all_t),
        np.array(all_V),
        np.array(all_I),
        np.array(all_R),
        np.array(all_r_ref),
        np.array(all_I_raw),
        np.array(all_alpha),
        np.array(all_log_hr),
        np.array(all_log_lr)
    )