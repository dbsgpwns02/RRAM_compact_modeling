import torch
import numpy as np

def format_real(val):
    return f"{val:.16e}"

def get_params(model):
    params = {}
    params['bz_']   = model.z_backbone[0].bias.data.cpu().numpy()
    params['Wz_']   = model.z_backbone[0].weight.data.cpu().numpy().flatten()
    params['bref_'] = model.head_ref.bias.data.cpu().numpy()[0]
    params['Wref_'] = model.head_ref.weight.data.cpu().numpy().flatten()
    params['bhr1_'] = model.head_hr[0].bias.data.cpu().numpy()
    params['Whr1_'] = model.head_hr[0].weight.data.cpu().numpy().flatten()
    params['bhr2_'] = model.head_hr[2].bias.data.cpu().numpy()[0]
    params['Whr2_'] = model.head_hr[2].weight.data.cpu().numpy().flatten()
    params['blr1_'] = model.head_lr[0].bias.data.cpu().numpy()
    params['Wlr1_'] = model.head_lr[0].weight.data.cpu().numpy().flatten()
    params['blr2_'] = model.head_lr[2].bias.data.cpu().numpy()[0]
    params['Wlr2_'] = model.head_lr[2].weight.data.cpu().numpy().flatten()
    return params


def export_veriloga_hspice_split_CONTINUOUS(
    model, scX, R_ref_minmax, ny,
    out_base="narx_rram_cont",
    module_name="narx_rram_cont",
    Ts=1e-9, time_step=1e-9,
    init_w=1, warmup_steps=40,
    r_clip_min=100.0, r_clip_max=1.0e8,
    v_filter=0.01,          # 실제 continuous 버전에서는 사용하지 않지만 파라미터로만 유지
    V_READ_TH=0.6,
    V_smooth_read=0.1,
    hyst_smooth=0.1,
):
    """
    debug_spice_sampling_va_matched_CONTINUOUS 와 1:1로 로직을 맞춘
    HSPICE용 3-file Verilog-A 모델을 export 한다.

    - LPF (tau_slew, I_hold) 사용
    - V_READ_TH, V_smooth_read 로 soft gate (READ/WRITE blending)
    - r_ref_last_valid 유지
    - W_hist 업데이트: W_target = 0.5*(1 + tanh(r_ref / hyst_smooth))
    """
    print(f"[Export] Generating CONTINUOUS Verilog-A model ({out_base}*)...")

    model.eval()
    params = get_params(model)

    in_dim = len(scX.mean_)
    z_dim = len(params['bz_'])
    hr_lr_in_dim  = model.head_hr[0].in_features
    hr_lr_hid_dim = model.head_hr[0].out_features

    shell_path  = f"{out_base}.vams"
    assign_path = f"{out_base}_assign.vams"
    math_path   = f"{out_base}_unrolled_math.inc"

    # =========================================================
    # 1) Shell (module) 파일 : CONTINUOUS 로직
    # =========================================================
    try:
        with open(shell_path, "w") as f:
            f.write(f"""// ----------------------------------------------------
//  NARX-based RRAM compact model (Continuous, HSPICE-safe)
//  - Logic matched to debug_spice_sampling_va_matched_CONTINUOUS
// ----------------------------------------------------
`include "constants.vams"
`include "disciplines.vams"

module {module_name}(g, s);
  inout g, s;
  electrical g, s;

  // ================== Parameters ==================
  parameter real Ts          = {format_real(Ts)};
  parameter integer N_warmup = {warmup_steps};
  parameter integer INIT_W   = {init_w};

  // READ/WRITE soft gate 기준
  parameter real V_READ_TH     = {format_real(V_READ_TH)};
  parameter real V_smooth_read = {format_real(V_smooth_read)};
  parameter real hyst_smooth   = {format_real(hyst_smooth)};

  parameter real V_FILTER    = {format_real(v_filter)}; // (실제 연산에는 사용하지 않음)
  parameter real Rref_min    = {format_real(R_ref_minmax[0])};
  parameter real Rref_max    = {format_real(R_ref_minmax[1])};

  // LPF (초기 스파이크 제어)
  parameter real tau_slew    = 2e-09 from [0:inf);

  parameter real R_clip_min  = {format_real(r_clip_min)};
  parameter real R_clip_max  = {format_real(r_clip_max)};
  parameter integer DBG      = 0;
  parameter real time_step   = {format_real(time_step)};

  // ================== Sizes ==================
  integer ny, z_dim, in_dim;
  integer hr_lr_in_dim, hr_lr_hid_dim;
  integer i, m;

  // ================== States / temps ==================
  real V_prev_raw;
  real dV, dir, rate;
  real I_hold;          // LPF 출력
  real I_pred_raw;      // gate + NN 결과 (LPF 입력)
  real R_last, G_last;  // 마지막 유효 저항 / 컨덕턴스
  real r_ref_last_valid; // READ 중에 유지할 r_ref
  integer k_sample_total;
  integer k_eff;

  // ================== Input / feature buffers ==================
  real x_raw[0:{in_dim-1}];
  real x_scl[0:{in_dim-1}];

  // ================== Scaler buffers ==================
  real mean_[0:{in_dim-1}];
  real scale_[0:{in_dim-1}];

  // ================== Weights / Biases ==================
  real bz_[0:{z_dim-1}];
  real Wz_[0:{(z_dim*in_dim)-1}];

  real Wref_[0:{z_dim-1}];
  real bref_;

  real Whr1_[0:{(hr_lr_hid_dim*hr_lr_in_dim)-1}];
  real bhr1_[0:{hr_lr_hid_dim-1}];
  real Whr2_[0:{hr_lr_hid_dim-1}];
  real bhr2_;

  real Wlr1_[0:{(hr_lr_hid_dim*hr_lr_in_dim)-1}];
  real blr1_[0:{hr_lr_hid_dim-1}];
  real Wlr2_[0:{hr_lr_hid_dim-1}];
  real blr2_;

  // ================== Network intermediates ==================
  real z[0:{z_dim-1}];
  real hr_h[0:{hr_lr_hid_dim-1}];
  real lr_h[0:{hr_lr_hid_dim-1}];

  // ================== Digital history ==================
  real W_hist[0:{ny-1}];

  // ================== Outputs / temps ==================
  real r_ref_logit, v_scl, log_hr, log_lr;
  real hr_val, lr_val, alpha_write, R_pred_write, I_pred_write;
  real alpha_read;
  real R_pred_read, I_pred_read;
  real R_pred;
  real gate_write;
  real W_current, W_target, W_next;
  real Vgs;
  real alpha_slew;

  // --- 가중치/스케일러 로드 ---
  `include "{module_name}_assign.vams"

  analog begin
    $bound_step(Ts);

    @(initial_step("tran")) begin
      ny            = {ny};
      z_dim         = {z_dim};
      in_dim        = {in_dim};
      hr_lr_in_dim  = {hr_lr_in_dim};
      hr_lr_hid_dim = {hr_lr_hid_dim};

      V_prev_raw = V(g, s);
      I_hold     = 0.0;
      I_pred_raw = 0.0;
      R_last     = R_clip_min;
      G_last     = 1.0 / R_last;
      k_sample_total = 0;
      k_eff      = 0;

      // INIT_W 에 따라 r_ref_last_valid 초기화
      if (INIT_W == 1)
        r_ref_last_valid = Rref_max;
      else
        r_ref_last_valid = Rref_min;

      for (i = 0; i <= ny-1; i = i+1)
        W_hist[i] = INIT_W;

      if (DBG) $strobe("{module_name}_cont: init (ny=%0d, z_dim=%0d, in_dim=%0d)", 
                       ny, z_dim, in_dim);
    end

    // 출력: LPF 결과
    I(g, s) <+ I_hold;

    @(timer(0, Ts)) begin
      Vgs = V(g, s);

      dV  = Vgs - V_prev_raw;
      dir = (dV > 0) ?  1 : ((dV < 0) ? -1 : 0);
      rate = abs(dV);

      // LPF 계수
      if (tau_slew <= 0.0)
        alpha_slew = 1.0;
      else
        alpha_slew = Ts / (tau_slew + Ts);
      if (alpha_slew > 1.0) alpha_slew = 1.0;
      else if (alpha_slew < 0.0) alpha_slew = 0.0;

      // ----------------------------------------------------
      // 1) READ 경로 (이전 상태 기반)
      // ----------------------------------------------------
      I_pred_read  = Vgs * G_last;
      R_pred_read  = R_last;
      alpha_read   = (r_ref_last_valid - Rref_min) / (Rref_max - Rref_min);
      if (alpha_read < 0.0) alpha_read = 0.0;
      else if (alpha_read > 1.0) alpha_read = 1.0;
      // log_hr/log_lr 는 READ 중 의미 없으니 0 취급

      // ----------------------------------------------------
      // 2) WRITE 경로 (NN 한 번 실행)
      // ----------------------------------------------------
      x_raw[0] = Vgs;
      x_raw[1] = dir;
      x_raw[2] = rate;
      for (i = 0; i <= ny-1; i = i+1)
        x_raw[i+3] = W_hist[i];

      for (i = 0; i <= in_dim-1; i = i+1)
        x_scl[i] = (x_raw[i] - mean_[i]) / scale_[i];

      v_scl = x_scl[0];

      `include "{module_name}_unrolled_math.inc"

      hr_val = exp(log_hr);
      lr_val = exp(log_lr);

      alpha_write = (r_ref_logit - Rref_min) / (Rref_max - Rref_min);
      if (alpha_write < 0.0) alpha_write = 0.0;
      else if (alpha_write > 1.0) alpha_write = 1.0;

      R_pred_write = alpha_write * hr_val + (1.0 - alpha_write) * lr_val;
      if (R_pred_write < R_clip_min) R_pred_write = R_clip_min;
      if (R_pred_write > R_clip_max) R_pred_write = R_clip_max;

      if (R_pred_write != 0.0)
        I_pred_write = Vgs / R_pred_write;
      else
        I_pred_write = 0.0;

      // ----------------------------------------------------
      // 3) Soft gate (READ/WRITE blending)
      // gate_write = 0.5 * (1 + tanh((|V|-V_READ_TH)/V_smooth_read))
      // ----------------------------------------------------
      gate_write = 0.5 * (1.0 + tanh((abs(Vgs) - V_READ_TH) / V_smooth_read));

      I_pred_raw = (1.0 - gate_write) * I_pred_read  + gate_write * I_pred_write;
      R_pred     = (1.0 - gate_write) * R_pred_read  + gate_write * R_pred_write;

      // r_ref/log_hr/log_lr 도 동일하게 블렌딩
      r_ref_logit = (1.0 - gate_write) * r_ref_last_valid + gate_write * r_ref_logit;
      // READ 쪽 log_hr/log_lr=0 이라서 WRITE 값 * gate_write 와 동일
      // alpha 또한 read/write 섞어서 쓸 수 있지만, 실제로는 R_pred만 사용

      // LPF 적용
      I_hold = I_hold + alpha_slew * (I_pred_raw - I_hold);

      // 다음 step을 위해 마지막 유효값 업데이트
      R_last          = R_pred;
      if (R_last < R_clip_min) R_last = R_clip_min;
      if (R_last > R_clip_max) R_last = R_clip_max;
      G_last          = 1.0 / R_last;
      r_ref_last_valid = r_ref_logit;

      // ----------------------------------------------------
      // 4) 상태 W_hist 업데이트 (continuous)
      // W_target = 0.5*(1 + tanh(r_ref / hyst_smooth))
      // W_next   = (1-gate)*W_current + gate*W_target
      // ----------------------------------------------------
      W_current = W_hist[0];
      W_target  = 0.5 * (1.0 + tanh(r_ref_logit / hyst_smooth));
      W_next    = (1.0 - gate_write) * W_current + gate_write * W_target;

      if (k_eff >= N_warmup) begin
        for (m = ny-1; m >= 1; m = m-1)
          W_hist[m] = W_hist[m-1];
        W_hist[0] = W_next;
      end

      if (DBG) $strobe("LOG t=%g V=%g dV=%g | gate_W=%g | r_ref=%g W_targ=%g W_next=%g | R=%g I_raw=%g I_lpf=%g",
                       $abstime, Vgs, dV, gate_write, r_ref_logit, W_target, W_next,
                       R_pred, I_pred_raw, I_hold);

      V_prev_raw = Vgs;
      k_eff = k_eff + 1;
      k_sample_total = k_sample_total + 1;
    end
  end
endmodule
""")
        print(f"[Export] CONT shell written: {shell_path}")
    except Exception as e:
        print(f"[Export] ERROR writing CONT shell: {e}")
        return

    # =========================================================
    # 2) assign.vams (스케일러 + 가중치)
    # =========================================================
    try:
        with open(assign_path, "w") as f:
            f.write("// Verilog-A numeric assignments (Continuous ver, auto-generated)\n")
            f.write(f"// Included by: {module_name}.vams\n")
            f.write("`include \"constants.vams\"\n\n")

            f.write("// --- Scaler ---\n")
            for i in range(in_dim):
                f.write(f"mean_[{i}]  = {format_real(scX.mean_[i])};\n")
                f.write(f"scale_[{i}] = {format_real(scX.scale_[i])};\n")

            f.write("\n// --- Z_backbone ---\n")
            for i in range(z_dim):
                f.write(f"bz_[{i}] = {format_real(params['bz_'][i])};\n")
            for i in range(z_dim * in_dim):
                f.write(f"Wz_[{i}] = {format_real(params['Wz_'][i])};\n")

            f.write("\n// --- Head_ref ---\n")
            f.write(f"bref_ = {format_real(params['bref_'])};\n")
            for i in range(z_dim):
                f.write(f"Wref_[{i}] = {format_real(params['Wref_'][i])};\n")

            f.write("\n// --- Head_hr ---\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"bhr1_[{i}] = {format_real(params['bhr1_'][i])};\n")
            for i in range(hr_lr_hid_dim * hr_lr_in_dim):
                f.write(f"Whr1_[{i}] = {format_real(params['Whr1_'][i])};\n")
            f.write(f"bhr2_ = {format_real(params['bhr2_'])};\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"Whr2_[{i}] = {format_real(params['Whr2_'][i])};\n")

            f.write("\n// --- Head_lr ---\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"blr1_[{i}] = {format_real(params['blr1_'][i])};\n")
            for i in range(hr_lr_hid_dim * hr_lr_in_dim):
                f.write(f"Wlr1_[{i}] = {format_real(params['Wlr1_'][i])};\n")
            f.write(f"blr2_ = {format_real(params['blr2_'])};\n")
            for i in range(hr_lr_hid_dim):
                f.write(f"Wlr2_[{i}] = {format_real(params['Wlr2_'][i])};\n")

            f.write("\n// --- End of assignments ---\n")
        print(f"[Export] CONT assign written: {assign_path}")
    except Exception as e:
        print(f"[Export] ERROR writing CONT assign: {e}")
        return

    # =========================================================
    # 3) _unrolled_math.inc (기존 discrete 버전이랑 동일)
    # =========================================================
    try:
        with open(math_path, "w") as f:
            f.write("// Verilog-A unrolled math (Continuous ver, auto-generated)\n")
            f.write(f"// Included by: {module_name}.vams\n")
            f.write("// [!!] DO NOT add 'real' declarations here [!!]\n\n")

            f.write("// --- A. Z_backbone (x_scl -> z) ---\n")
            for i in range(z_dim):
                line = f"z[{i}] = bz_[{i}]"
                for j in range(in_dim):
                    line += f" + Wz_[{i * in_dim + j}]*x_scl[{j}]"
                line += ";\n"
                f.write(line)
                f.write(f"z[{i}] = tanh(z[{i}]);\n")

            f.write("\n// --- B. Head_ref (z -> r_ref_logit) ---\n")
            line = "r_ref_logit = bref_"
            for i in range(z_dim):
                line += f" + Wref_[{i}]*z[{i}]"
            line += ";\n"
            f.write(line)

            f.write("\n// --- C. Head_hr (z, v_scl -> log_hr) ---\n")
            for i in range(hr_lr_hid_dim):
                line = f"hr_h[{i}] = bhr1_[{i}]"
                for j in range(z_dim):
                    line += f" + Whr1_[{i * hr_lr_in_dim + j}]*z[{j}]"
                line += f" + Whr1_[{i * hr_lr_in_dim + z_dim}]*v_scl"
                line += ";\n"
                f.write(line)
                f.write(f"hr_h[{i}] = tanh(hr_h[{i}]);\n")
            line = "log_hr = bhr2_"
            for i in range(hr_lr_hid_dim):
                line += f" + Whr2_[{i}]*hr_h[{i}]"
            line += ";\n"
            f.write(line)

            f.write("\n// --- D. Head_lr (z, v_scl -> log_lr) ---\n")
            for i in range(hr_lr_hid_dim):
                line = f"lr_h[{i}] = blr1_[{i}]"
                for j in range(z_dim):
                    line += f" + Wlr1_[{i * hr_lr_in_dim + j}]*z[{j}]"
                line += f" + Wlr1_[{i * hr_lr_in_dim + z_dim}]*v_scl"
                line += ";\n"
                f.write(line)
                f.write(f"lr_h[{i}] = tanh(lr_h[{i}]);\n")
            line = "log_lr = blr2_"
            for i in range(hr_lr_hid_dim):
                line += f" + Wlr2_[{i}]*lr_h[{i}]"
            line += ";\n"
            f.write(line)

            f.write("\n// --- End of math ---\n")
        print(f"[Export] CONT math written: {math_path}")
    except Exception as e:
        print(f"[Export] ERROR writing CONT math: {e}")
        return

    print(f"[Export] CONTINUOUS Verilog-A files generated: {shell_path}, {assign_path}, {math_path}")
