#
# generate_pwl.py (PWL 생성 함수 2개 포함)
#
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# [함수 1] SPICE .sp 파일 1:1 복제 (첫 번째 버전)
# ------------------------------------------------------------
def generate_v_data_from_spice_pwl(
    Ts=1e-9, 
    t_stop=1728e-9, 
    save_csv_path=None
):
    """
    SPICE .sp 파일의 하드코딩된 PWL 정의를 Ts 간격으로 샘플링
    [수정] (time_data, voltage_data)를 반환.
    """
    print(f"[PWL Gen] Generating V_data from Static SPICE PWL (Ts={Ts}, t_stop={t_stop})")

    param_set = 1.2
    param_reset = -1.5

    pwl_points = [
        (0e-9, 0.0), (25e-9, 0.0), (35e-9, 0.5), (60e-9, 0.5),
        (70e-9, 0.0), (95e-9, 0.0), (125e-9, param_reset), (150e-9, param_reset),
        (180e-9, 0.0), (205e-9, 0.0), (215e-9, 0.5), (240e-9, 0.5),
        (250e-9, 0.0), (275e-9, 0.0), (299e-9, param_set), (324e-9, param_set),
        (348e-9, 0.0), (373e-9, 0.0), (383e-9, 0.5), (408e-9, 0.5),
        (418e-9, 0.0), (443e-9, 0.0), (473e-9, param_reset), (498e-9, param_reset),
        (528e-9, 0.0), (553e-9, 0.0), (563e-9, 0.5), (588e-9, 0.5),
        (598e-9, 0.0), (623e-9, 0.0), (647e-9, param_set), (672e-9, param_set),
        (696e-9, 0.0), (721e-9, 0.0), (731e-9, 0.5), (756e-9, 0.5),
        (766e-9, 0.0), (791e-9, 0.0), (821e-9, param_reset), (846e-9, param_reset),
        (876e-9, 0.0), (901e-9, 0.0), (911e-9, 0.5), (936e-9, 0.5),
        (946e-9, 0.0), (971e-9, 0.0), (995e-9, param_set), (1020e-9, param_set),
        (1044e-9, 0.0), (1069e-9, 0.0), (1079e-9, 0.5), (1104e-9, 0.5),
        (1114e-9, 0.0), (1139e-9, 0.0), (1163e-9, param_set), (1188e-9, param_set),
        (1212e-9, 0.0), (1237e-9, 0.0), (1247e-9, 0.5), (1272e-9, 0.5),
        (1282e-9, 0.0), (1307e-9, 0.0), (1331e-9, param_set), (1356e-9, param_set),
        (1380e-9, 0.0), (1405e-9, 0.0), (1415e-9, 0.5), (1440e-9, 0.5),
        (1450e-9, 0.0), (1475e-9, 0.0), (1505e-9, param_reset), (1530e-9, param_reset),
        (1560e-9, 0.0), (1585e-9, 0.0), (1595e-9, 0.5), (1620e-9, 0.5),
        (1630e-9, 0.0), (1655e-9, 0.0), (1679e-9, param_set), (1704e-9, param_set),
        (1728e-9, 0.0)
    ]

    pwl_times = np.array([t for t, v in pwl_points])
    pwl_voltages = np.array([v for t, v in pwl_points])

    t_new = np.arange(0, t_stop + Ts/2, Ts)
    V_new = np.interp(t_new, pwl_times, pwl_voltages)
    V_new = V_new.astype(np.float32)

    print(f"[PWL Gen] V_data 생성 완료. 총 {len(V_new)} 스텝.")
    
    if save_csv_path:
        p_path = Path(save_csv_path)
        p_path.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_state = np.zeros_like(V_new)
        dummy_current = np.zeros_like(V_new)
        df = pd.DataFrame({
            'State': dummy_state,
            'Voltage': V_new,
            'Current': dummy_current
        })
        df.to_csv(save_csv_path, header=False, index=False)
        print(f"[PWL Gen] 새 V_data를 {save_csv_path}에 저장했습니다.")

    # [중요] 반드시 2개만 반환해야 함
    return t_new, V_new

# ------------------------------------------------------------
# [함수 2] Sweeping Rate 조절 (두 번째 버전)
# ------------------------------------------------------------
def generate_sweep_v_data(
    V_set=1.2, 
    V_reset=-1.5,
    V_read=0.5,
    sweep_rate_V_per_s=1.0e8,
    hold_time_s=25e-9,
    zero_time_s=25e-9,
    Ts=1e-9
):
    """
    지정된 sweep_rate (V/s)로 SET/RESET 펄스를 생성하고,
    Ts 간격으로 샘플링된 V_data 배열을 반환합니다.
    """
    print(f"[PWL Gen] sweep_rate={sweep_rate_V_per_s:.1e} V/s 로 파형 생성...")
    pwl_points = []
    t_now = 0.0

    # 1. 초기 0V 구간
    t_now += zero_time_s
    pwl_points.append((t_now, 0.0))
    
    # 2. SET 펄스 (Ramp-up)
    t_ramp_set = abs(V_set - 0.0) / sweep_rate_V_per_s
    t_now += t_ramp_set
    pwl_points.append((t_now, V_set))
    
    # ... (이하 함수 내용은 동일) ...
    t_now += hold_time_s
    pwl_points.append((t_now, V_set))
    t_ramp_down = abs(V_set - 0.0) / sweep_rate_V_per_s
    t_now += t_ramp_down
    pwl_points.append((t_now, 0.0))
    t_now += zero_time_s
    pwl_points.append((t_now, 0.0))
    t_ramp_reset = abs(V_reset - 0.0) / sweep_rate_V_per_s
    t_now += t_ramp_reset
    pwl_points.append((t_now, V_reset))
    t_now += hold_time_s
    pwl_points.append((t_now, V_reset))
    t_ramp_up = abs(V_reset - 0.0) / sweep_rate_V_per_s
    t_now += t_ramp_up
    pwl_points.append((t_now, 0.0))
    
    t_stop = t_now

    pwl_times = np.array([0.0] + [t for t, v in pwl_points])
    pwl_voltages = np.array([0.0] + [v for t, v in pwl_points])
    
    t_new = np.arange(0, t_stop + Ts/2, Ts)
    V_new = np.interp(t_new, pwl_times, pwl_voltages)
    V_new = V_new.astype(np.float32)

    print(f"[PWL Gen] V_data 생성 완료. (총 {len(V_new)} 스텝, {t_stop*1e9:.0f} ns)")
    
    return V_new