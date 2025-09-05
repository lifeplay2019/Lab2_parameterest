import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt
from scipy import stats, signal
import warnings

warnings.filterwarnings('ignore')


class RLS_ThermalBattery:
    def __init__(
            self,
            Cs_fixed=3.5,
            lambda_factor=0.999,  # 仅兜底；开启VFF后以lambda(k)为准
            P0=1e2,
            param_bounds=None,
            adaptive_lambda=False,  # 已不建议使用；保留接口
            use_vff=True,
            lambda_min=0.96,
            lambda_max=0.9995,
            vff_rho=0.6,
            vff_window=80
    ):
        """
        VFFRLS用于热模型参数辨识（Cs 固定）
        参数: theta = [Cc, Rc, Rs]
        模型: Ts[k] = b1*Ts[k-1] + b2*Ta[k] + b3*Ta[k-1] + b4*H[k-1]
        """
        # 固定参数
        self.Cs = Cs_fixed
        self.lambda_factor = lambda_factor  # 兜底
        self.n_params = 3  # [Cc, Rc, Rs]

        # 初始参数（可按经验值设置）
        self.theta = np.array([[100.0], [1.5], [3.5]])  # [Cc, Rc, Rs]
        self.theta_init = self.theta.copy()
        self.P = P0 * np.eye(self.n_params)

        # 数值稳定相关参数
        self.min_eigenvalue = 1e-12
        self.max_eigenvalue = 1e8
        self.regularization = 1e-8

        # 参数边界
        self.param_bounds = param_bounds or {
            'Cc': (20.0, 200.0),  # J/K
            'Rc': (0.1, 50.0),  # K/W
            'Rs': (0.1, 50.0)  # K/W
        }

        # 参数平滑：指数加权移动平均
        self.window_size = 10
        self.param_buffer = []

        # 历史存储
        self.theta_history = []
        self.theta_smooth_history = []
        self.error_history = []
        self.residuals_history = []
        self.physical_params_history = []
        self.b_coefficients_history = []
        self.condition_number_history = []
        self.covariance_history = []
        self.information_history = []
        self.Ts_pred_history = []
        self.Tc_pred_history = []  # 添加Tc预测历史
        self.innovation_history = []

        # 自适应遗忘（旧接口）
        self.adaptive_lambda = adaptive_lambda

        # 异常值检测
        self.residual_buffer = []
        self.residual_threshold = 3.0  # 3-sigma

        # VFFRLS 参数
        self.use_vff = use_vff
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.vff_rho = float(vff_rho)
        self.vff_window = int(vff_window)
        self.e_window = []  # 最近M个残差窗口
        self.lambda_history = []  # 每步lambda(k)

    @staticmethod
    def _safe(x, eps=1e-12):
        if abs(x) >= eps:
            return x
        return eps if x == 0 else np.sign(x) * eps

    def detect_outlier(self, residual):
        """基于滑动窗口的异常值检测（MAD）"""
        self.residual_buffer.append(abs(residual))
        if len(self.residual_buffer) > 50:
            self.residual_buffer.pop(0)

        if len(self.residual_buffer) < 10:
            return False

        median_res = np.median(self.residual_buffer)
        mad = np.median(np.abs(np.array(self.residual_buffer) - median_res))
        threshold = median_res + self.residual_threshold * mad * 1.4826
        return abs(residual) > max(threshold, 0.5)

    def regularize_covariance(self):
        """协方差矩阵正则化"""
        self.P = 0.5 * (self.P + self.P.T)
        eigenvals, eigenvecs = np.linalg.eigh(self.P)
        eigenvals = np.clip(eigenvals, self.min_eigenvalue, self.max_eigenvalue)
        self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        self.P += self.regularization * np.eye(self.n_params)

    def calculate_b_coefficients(self, dt=1.0):
        """从物理参数计算 b1, b2, b3, b4（经验稳定公式）"""
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])
        Cs = self.Cs
        eps = 1e-12
        try:
            tau_c = Cc * Rc
            tau_s = Cs * Rs
            tau_cs = Cc * Rs
            denom = self._safe(Cc * (Rc + Rs), eps)

            b1 = np.exp(-dt / max(tau_c + tau_s, eps))
            b2 = (1 - np.exp(-dt / max(tau_cs, eps))) * Rc / self._safe(Rc + Rs, eps)
            b3 = np.exp(-dt / max(tau_cs, eps)) * Rc / self._safe(Rc + Rs, eps)
            b4 = Rs * dt / denom

            b1 = np.clip(b1, 0.0, 1.0)
            b2 = np.clip(b2, 0.0, 1.0)
            b3 = np.clip(b3, 0.0, 1.0)
            b4 = np.clip(b4, 0.0, 10.0)
        except Exception:
            b1, b2, b3, b4 = 0.9, 0.05, 0.0, 0.01
        return np.array([b1, b2, b3, b4])

    def calculate_Tc(self, Ts_pred, Ta, H, dt=1.0):
        """计算核心温度Tc"""
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])

        # 根据热模型计算核心温度
        # Tc = Ts + (H * Rc)  # 简化的核心温度计算
        # 更准确的计算方式
        Tc = Ts_pred + (Ts_pred - Ta) * (Rc / (Rc + Rs)) + H * Rc
        return float(Tc)

    def construct_jacobian_robust(self, Ts_prev, Ta_curr, Ta_prev, H_prev, dt=1.0):
        """数值差分雅可比 J = ∂Ts_pred/∂theta (3x1)"""
        h_rel = 1e-6
        J = np.zeros((3, 1))
        b_current = self.calculate_b_coefficients(dt)
        phi = np.array([Ts_prev, Ta_curr, Ta_prev, H_prev])
        y_current = float(np.dot(phi, b_current))

        for i in range(3):
            theta_plus = self.theta.copy()
            step = max(abs(theta_plus[i, 0]) * h_rel, 1e-6)
            theta_plus[i, 0] += step

            old_theta = self.theta.copy()
            self.theta = theta_plus
            b_plus = self.calculate_b_coefficients(dt)
            y_plus = float(np.dot(phi, b_plus))
            self.theta = old_theta

            J[i, 0] = (y_plus - y_current) / step

        J = np.clip(J, -1e6, 1e6)
        J = np.nan_to_num(J, nan=0.0, posinf=1e6, neginf=-1e6)
        return J

    def apply_parameter_constraints(self):
        self.theta[0, 0] = np.clip(self.theta[0, 0], *self.param_bounds['Cc'])
        self.theta[1, 0] = np.clip(self.theta[1, 0], *self.param_bounds['Rc'])
        self.theta[2, 0] = np.clip(self.theta[2, 0], *self.param_bounds['Rs'])

    def smooth_parameters(self):
        """指数加权平均平滑参数输出"""
        current_params = self.theta.flatten().copy()
        self.param_buffer.append(current_params)
        if len(self.param_buffer) > self.window_size:
            self.param_buffer.pop(0)

        if len(self.param_buffer) >= 5:
            param_array = np.array(self.param_buffer)
            weights = np.exp(np.linspace(-1, 0, len(param_array)))
            weights /= weights.sum()
            smoothed_params = np.average(param_array, axis=0, weights=weights)
            return smoothed_params
        return current_params

    @staticmethod
    def _adaptive_huber(e, residual_history, delta_factor=2.0):
        """自适应Huber损失，返回(e_huber, w)"""
        if len(residual_history) < 10:
            delta = 2.0
        else:
            mad = np.median(np.abs(np.array(residual_history[-50:])))
            delta = delta_factor * max(mad, 0.1)

        a = abs(e)
        if a <= delta:
            return e, 1.0
        return delta * np.sign(e), delta / a

    def compute_lambda_vff(self, e_raw):
        """
        可变遗忘因子 VFF:
        L = rho * MSE(k), lambda(k) = lam_min + (lam_max - lam_min) * 2^{-L}
        残差MSE用MAD限幅后的窗口估计
        """
        self.e_window.append(float(e_raw))
        if len(self.e_window) > self.vff_window:
            self.e_window.pop(0)

        if len(self.e_window) < max(10, self.vff_window // 5):
            lam = self.lambda_max
            self.lambda_history.append(lam)
            return lam

        e_arr = np.array(self.e_window, dtype=float)
        med = np.median(e_arr)
        mad = np.median(np.abs(e_arr - med))
        scale = max(mad * 1.4826, 1e-6)
        e_clipped = np.clip(e_arr - med, -3.0 * scale, 3.0 * scale)
        mse = float(np.mean(e_clipped ** 2))

        L = self.vff_rho * mse
        s = 2.0 ** (-L)
        lam = self.lambda_min + (self.lambda_max - self.lambda_min) * s
        lam = float(np.clip(lam, self.lambda_min, self.lambda_max))

        if self.lambda_history:
            lam = 0.7 * self.lambda_history[-1] + 0.3 * lam

        self.lambda_history.append(lam)
        return lam

    def update(self, Ts_prev, Ta_curr, Ta_prev, H_prev, Ts_curr, dt=1.0):
        """VFFRLS更新"""
        # 1) 计算当前b与预测
        b = self.calculate_b_coefficients(dt)
        phi = np.array([Ts_prev, Ta_curr, Ta_prev, H_prev], dtype=float)
        Ts_pred = float(np.dot(phi, b))
        self.Ts_pred_history.append(Ts_pred)

        # 计算并存储Tc预测
        Tc_pred = self.calculate_Tc(Ts_pred, Ta_curr, H_prev, dt)
        self.Tc_pred_history.append(Tc_pred)

        # 2) 原始残差 & 预测外点检测（仅用于统计）
        e_raw = float(Ts_curr - Ts_pred)
        _is_outlier = self.detect_outlier(e_raw)

        # 3) Huber鲁棒残差（用于参数更新）
        e_huber, _w = self._adaptive_huber(
            e_raw,
            self.residuals_history[-50:] if len(self.residuals_history) > 50 else self.residuals_history
        )

        # 4) 可变遗忘因子
        lam_k = self.compute_lambda_vff(e_raw) if self.use_vff else self.lambda_factor

        # 5) 雅可比（回归向量）
        J = self.construct_jacobian_robust(Ts_prev, Ta_curr, Ta_prev, H_prev, dt)  # 3x1

        # 6) 协方差正则
        self.regularize_covariance()

        # 7) VFFRLS更新
        JT_P = (J.T @ self.P)  # 1x3
        J_T_P_J = float(JT_P @ J)  # 标量
        denom = lam_k + J_T_P_J
        if abs(denom) < 1e-12:
            denom = 1e-12
        K = (self.P @ J) / denom  # 3x1

        # 限制增益幅度
        K_norm = float(np.linalg.norm(K))
        if K_norm > 5.0:
            K = K * (5.0 / K_norm)

        # 参数更新
        self.theta = self.theta + K * e_huber
        self.apply_parameter_constraints()

        # 协方差更新
        self.P = (self.P - K @ (J.T @ self.P)) / lam_k

        # 数值修正
        self.regularize_covariance()

        # 记录
        self.condition_number_history.append(float(np.linalg.cond(self.P)))
        self.covariance_history.append(self.P.copy())
        self.information_history.append(J_T_P_J)
        self.innovation_history.append(abs(e_huber))
        self.theta_history.append(self.theta.flatten().copy())
        self.error_history.append(float(e_huber))
        self.residuals_history.append(float(e_raw))
        self.b_coefficients_history.append(b.copy())

        # 平滑参数
        smoothed_params = self.smooth_parameters()
        self.theta_smooth_history.append(smoothed_params.copy())

        # 存储物理参数
        physical_params = {
            'Cc': float(smoothed_params[0]),
            'Cs': self.Cs,
            'Rc': float(smoothed_params[1]),
            'Rs': float(smoothed_params[2])
        }
        self.physical_params_history.append(physical_params)

        return self.theta.flatten()

    def calculate_confidence_intervals(self, confidence_level=0.95):
        """基于最终P与鲁棒残差方差估计参数置信区间（近似）"""
        if len(self.covariance_history) == 0 or len(self.residuals_history) <= self.n_params:
            return None
        P_final = self.covariance_history[-1]
        final_params = self.theta_smooth_history[-1]
        residuals = np.array(self.residuals_history, dtype=float)

        dof = len(residuals) - self.n_params
        if dof <= 0:
            return None

        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma2 = (mad * 1.4826) ** 2
        P_param = sigma2 * P_final  # 若你认为P已是绝对协方差，可去掉sigma2这一步

        std_errors = np.sqrt(np.diag(P_param))
        alpha = 1 - confidence_level
        t_value = float(stats.t.ppf(1 - alpha / 2, dof))

        confidence_intervals = {}
        for name, i in zip(['Cc', 'Rc', 'Rs'], [0, 1, 2]):
            margin = t_value * std_errors[i]
            confidence_intervals[name] = (final_params[i] - margin, final_params[i] + margin)
        confidence_intervals['Cs'] = (self.Cs, self.Cs)
        return confidence_intervals

    def assess_convergence(self, window_size=100):
        """评估收敛性"""
        if len(self.theta_smooth_history) < window_size:
            return False, {}
        recent_params = np.array(self.theta_smooth_history[-window_size:])
        param_std = np.std(recent_params, axis=0)
        param_mean = np.mean(recent_params, axis=0)

        cv_threshold = 0.005  # 0.5%
        convergence_status = {}
        all_converged = True
        for i, name in enumerate(['Cc', 'Rc', 'Rs']):
            mean_abs = abs(param_mean[i])
            cv = param_std[i] / mean_abs if mean_abs > 1e-12 else float('inf')
            converged = bool(cv < cv_threshold)
            convergence_status[name] = {
                'converged': converged,
                'cv': float(cv),
                'std': float(param_std[i]),
                'mean': float(param_mean[i])
            }
            if not converged:
                all_converged = False
        return all_converged, convergence_status


def load_soc_ocv_data(filepath):
    """读取SOC-OCV数据并拟合多项式（6阶）"""
    print(f"Loading SOC-OCV data from: {filepath}")
    try:
        df_soc_ocv = pd.read_excel(filepath)
        soc_data = pd.to_numeric(df_soc_ocv.iloc[:, 0], errors='coerce').values
        ocv_data = pd.to_numeric(df_soc_ocv.iloc[:, 1], errors='coerce').values
        valid_mask = ~(np.isnan(soc_data) | np.isnan(ocv_data))
        soc_data = soc_data[valid_mask]
        ocv_data = ocv_data[valid_mask]
        poly_coeffs = np.polyfit(soc_data, ocv_data, 6)
        print(f"Loaded {len(soc_data)} SOC-OCV points. Poly deg=6.")
        return poly_coeffs
    except Exception as e:
        print(f"Error loading SOC-OCV data: {e}")
        return np.array([0, 0, 0, 0, 0, 1.7, 2.5], dtype=float)


def calculate_ocv_from_soc(soc, poly_coeffs):
    soc_clipped = np.clip(soc, 0, 1)
    ocv = np.polyval(poly_coeffs, soc_clipped)
    return ocv


def calculate_soc_from_voltage(voltage, poly_coeffs, soc_range=None):
    if soc_range is None:
        soc_range = np.linspace(0, 1, 2001)
    ocv_range = calculate_ocv_from_soc(soc_range, poly_coeffs)
    order = np.argsort(ocv_range)
    ocv_sorted = ocv_range[order]
    soc_sorted = soc_range[order]
    ocv_unique, idx = np.unique(ocv_sorted, return_index=True)
    soc_unique = soc_sorted[idx]
    soc_estimated = np.interp(voltage, ocv_unique, soc_unique)
    soc_estimated = np.clip(soc_estimated, 0, 1)
    return soc_estimated


def preprocess_data_with_filtering(t, Ts, Ta, H):
    """异常值处理、插值、滤波与变化率限制"""
    n = len(Ts)

    def remove_outliers(data, threshold=3.0):
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        return np.where(z_scores > threshold, np.nan, data)

    Ts_clean = remove_outliers(Ts)
    Ta_clean = remove_outliers(Ta)
    H_clean = remove_outliers(H)

    def interpolate_nans(data):
        mask = ~np.isnan(data)
        if mask.sum() < len(data):
            indices = np.arange(len(data))
            data = np.interp(indices, indices[mask], data[mask])
        return data

    Ts_clean = interpolate_nans(Ts_clean)
    Ta_clean = interpolate_nans(Ta_clean)
    H_clean = interpolate_nans(H_clean)

    if n > 5:
        win_len_ts = min(9, max(5, (n // 30) * 2 + 1))  # odd
        polyorder = 2

        Ts_filtered = savgol_filter(Ts_clean, window_length=win_len_ts, polyorder=polyorder, mode='interp')
        Ta_filtered = savgol_filter(Ta_clean, window_length=win_len_ts, polyorder=polyorder, mode='interp')

        # medfilt kernel必须奇数
        k = max(3, n // 50)
        if k % 2 == 0:
            k += 1
        k = min(k, 7)
        H_med = medfilt(H_clean, kernel_size=k)

        win_len_h = min(7, max(5, (n // 50) * 2 + 1))
        H_filtered = savgol_filter(H_med, window_length=win_len_h, polyorder=2, mode='interp')

        def limit_derivative(data, max_change_rate=5.0):
            diff = np.diff(data)
            mask = np.abs(diff) > max_change_rate
            if mask.any():
                for i in np.where(mask)[0]:
                    if i + 1 < len(data):
                        data[i + 1] = data[i] + np.sign(diff[i]) * max_change_rate
            return data

        Ts_filtered = limit_derivative(Ts_filtered, max_change_rate=2.0)
        H_filtered = limit_derivative(H_filtered, max_change_rate=10.0)
    else:
        Ts_filtered = Ts_clean
        Ta_filtered = Ta_clean
        H_filtered = H_clean

    return Ts_filtered, Ta_filtered, H_filtered


def load_and_preprocess_data(filepath, soc_ocv_filepath):
    """加载与预处理数据"""
    poly_coeffs = load_soc_ocv_data(soc_ocv_filepath)
    print(f"Loading main data from: {filepath}")
    try:
        df = pd.read_excel(filepath)
        if df.shape[1] >= 8:
            t_0 = df.iloc[:, 0].values
            t = df.iloc[:, 1].values
            v = df.iloc[:, 2].values
            i = df.iloc[:, 3].values
            Ta = df.iloc[:, 4].values
            Ts = df.iloc[:, 5].values
            Q_rm = df.iloc[:, 6].values
            SOC = df.iloc[:, 7].values
        else:
            t_0 = df.iloc[:, 0].values
            t = df.iloc[:, 1].values
            v = df.iloc[:, 2].values
            i = df.iloc[:, 3].values
            Ta = df.iloc[:, 4].values
            Ts = df.iloc[:, 5].values
            v = pd.to_numeric(v, errors='coerce')
            SOC = calculate_soc_from_voltage(v, poly_coeffs)

        # 转换类型与清洗
        t = np.asarray(pd.to_numeric(t, errors='coerce'))
        v = np.asarray(pd.to_numeric(v, errors='coerce'))
        i = np.asarray(pd.to_numeric(i, errors='coerce'))
        Ta = np.asarray(pd.to_numeric(Ta, errors='coerce'))
        Ts = np.asarray(pd.to_numeric(Ts, errors='coerce'))
        SOC = np.asarray(pd.to_numeric(SOC, errors='coerce'))

        valid_idx = ~(np.isnan(t) | np.isnan(v) | np.isnan(i) | np.isnan(Ta) | np.isnan(Ts) | np.isnan(SOC))
        t = t[valid_idx]
        v = v[valid_idx]
        i = i[valid_idx]
        Ta = Ta[valid_idx]
        Ts = Ts[valid_idx]
        SOC = SOC[valid_idx]

        if len(t) > 1:
            dt = float(np.median(np.diff(t)))
        else:
            dt = 10.0

        # OCV与产热估计（只考虑过电位项）
        Uocv = calculate_ocv_from_soc(SOC, poly_coeffs)
        overpotential = Uocv - v
        H = np.where(np.abs(overpotential) > 0.01, np.abs(overpotential * i), 0.0001)
        H = np.clip(H, 0.0001, np.percentile(H, 95))

        Ts_filtered, Ta_filtered, H_filtered = preprocess_data_with_filtering(t, Ts, Ta, H)

        print("Data preprocessing completed")
        print(f"After cleaning: {len(t)} valid data points")
        print(f"Time step dt: {dt:.3f} s")
        print(f"Ts range: {float(Ts_filtered.min()):.1f}~{float(Ts_filtered.max()):.1f} °C")
        print(f"H range:  {float(H_filtered.min()):.4f}~{float(H_filtered.max()):.3f} W")

        return t, Ts_filtered, Ta_filtered, H_filtered, dt, SOC, Uocv
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_results_tables(rls, final_params, confidence_intervals, convergence_status):
    print("\n" + "=" * 70)
    print("THERMAL SYSTEM MODEL (TSM) VFFRLS IDENTIFICATION RESULTS")
    print("=" * 70)

    # Table 2: 初始条件（与真实初始化一致）
    print("\nTable 2")
    print("TSM VFFRLS initial conditions.")
    print("-" * 50)
    print(f"{'Parameters':<15} {'Cc (J/K)':<15} {'Rc (K/W)':<15} {'Rs (K/W)':<15} {'Cs (J/K)':<15}")
    print(
        f"{'Values':<15} {rls.theta_init[0, 0]:<15.2f} {rls.theta_init[1, 0]:<15.3f} {rls.theta_init[2, 0]:<15.3f} {rls.Cs:<15.2f}")

    # Table 3: 辨识参数
    print("\nTable 3")
    print("Identified thermal parameters.")
    print("-" * 70)
    print(f"{'Parameters':<15} {'Values':<20} {'95% Confidence interval':<30}")
    print("-" * 70)

    cc_val = final_params['Cc']
    cs_val = final_params['Cs']
    rc_val = final_params['Rc']
    rs_val = final_params['Rs']

    if confidence_intervals:
        cc_ci = confidence_intervals['Cc']
        cs_ci = confidence_intervals['Cs']
        rc_ci = confidence_intervals['Rc']
        rs_ci = confidence_intervals['Rs']
        print(f"{'Cc (J/K)':<15} {cc_val:<20.4f} {cc_ci[0]:.4f}~{cc_ci[1]:.4f}")
        print(f"{'Cs (J/K)':<15} {cs_val:<20.2f} {cs_ci[0]:.4f}~{cs_ci[1]:.4f}")
        print(f"{'Rc (K/W)':<15} {rc_val:<20.4f} {rc_ci[0]:.4f}~{rc_ci[1]:.4f}")
        print(f"{'Rs (K/W)':<15} {rs_val:<20.4f} {rs_ci[0]:.4f}~{rs_ci[1]:.4f}")
    else:
        print(f"{'Cc (J/K)':<15} {cc_val:<20.4f} {'N/A':<30}")
        print(f"{'Cs (J/K)':<15} {cs_val:<20.2f} {'Fixed':<30}")
        print(f"{'Rc (K/W)':<15} {rc_val:<20.4f} {'N/A':<30}")
        print(f"{'Rs (K/W)':<15} {rs_val:<20.4f} {'N/A':<30}")

    print("=" * 70)

    # 收敛分析
    print("\nParameter Convergence Analysis:")
    print("-" * 50)
    for param_name, status in convergence_status.items():
        converged_str = "✓" if status['converged'] else "✗"
        print(f"{param_name:<5}: {converged_str} (CV: {status['cv']:.4f}, Std: {status['std']:.4f})")

    # 模型性能
    if rls.residuals_history:
        residuals = np.array(rls.residuals_history, dtype=float)
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        mae = float(np.mean(np.abs(residuals)))
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        print(f"\nModel Performance:")
        print("-" * 30)
        print(f"RMSE: {rmse:.4f} °C")
        print(f"MAE:  {mae:.4f} °C")
        print(f"MAD:  {mad:.4f} °C")
        print(f"Final condition number: {rls.condition_number_history[-1]:.2e}")

        outlier_count = sum(1 for r in rls.residuals_history if abs(r) > 3 * mad * 1.4826)
        print(f"Outliers detected (3σ via MAD): {outlier_count}/{len(rls.residuals_history)} "
              f"({outlier_count / len(rls.residuals_history) * 100:.1f}%)")


def run_rls_identification(
        filepath,
        soc_ocv_filepath,
        Cs_value=3.5,
        lambda_factor=0.999,  # 兜底；use_vff=True时不会使用
        P0=1e2,
        use_vff=True,
        lambda_min=0.96,
        lambda_max=0.9995,
        vff_rho=0.6,
        vff_window=80
):
    """运行VFFRLS辨识"""
    print("Loading data...")
    t, Ts, Ta, H, dt, SOC, Uocv = load_and_preprocess_data(filepath, soc_ocv_filepath)

    print(f"Using fixed Cs = {Cs_value} J/K")
    rls = RLS_ThermalBattery(
        Cs_fixed=Cs_value,
        lambda_factor=lambda_factor,
        P0=P0,
        adaptive_lambda=False,  # 已弃用
        use_vff=use_vff,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        vff_rho=vff_rho,
        vff_window=vff_window
    )

    n_samples = len(Ts)
    print(f"Running VFFRLS identification on {n_samples} samples...")
    print(f"Time step dt = {dt:.3f} seconds")

    # 可选预热期：前少量样本用lambda_max（长记忆），让窗口稳定
    warmup = min(50, n_samples // 20)

    for k in range(1, n_samples):
        # 计算一步更新
        theta = rls.update(
            Ts_prev=float(Ts[k - 1]),
            Ta_curr=float(Ta[k]),
            Ta_prev=float(Ta[k - 1]),
            H_prev=float(H[k - 1]),
            Ts_curr=float(Ts[k]),
            dt=dt
        )

        # 预热阶段时，把lambda强制为最大值，稳定窗口（仅影响lambda_history）
        if use_vff and k < warmup:
            if rls.lambda_history:
                rls.lambda_history[-1] = rls.lambda_max

        if k % 500 == 0 or k == n_samples - 1:
            smoothed = rls.theta_smooth_history[-1]
            cond_num = rls.condition_number_history[-1]
            recent_error = np.mean(np.abs(rls.residuals_history[-min(10, len(rls.residuals_history)):]))
            lam_show = rls.lambda_history[-1] if rls.lambda_history else rls.lambda_factor
            print(f"Step {k}/{n_samples}: "
                  f"Cc={smoothed[0]:.2f}, Rc={smoothed[1]:.4f}, Rs={smoothed[2]:.4f}, "
                  f"Err={recent_error:.3f}, λ={lam_show:.6f}, Cond={cond_num:.2e}")

    # 结果分析
    final_params = rls.physical_params_history[-1]
    confidence_intervals = rls.calculate_confidence_intervals()
    all_converged, convergence_status = rls.assess_convergence()

    print_results_tables(rls, final_params, confidence_intervals, convergence_status)
    plot_results_separate(rls, t[1:], SOC[1:], Uocv[1:], Ts[1:], Ta[1:])

    return rls, final_params


def plot_results_separate(rls, t, SOC, Uocv, Ts_measured, Ta_measured):
    """结果可视化 - 分开展示每个图片"""
    theta_history = np.array(rls.theta_history)
    theta_smooth_history = np.array(rls.theta_smooth_history)
    Ts_pred = np.array(rls.Ts_pred_history, dtype=float)
    Tc_pred = np.array(rls.Tc_pred_history, dtype=float)
    residuals = np.array(rls.residuals_history, dtype=float)

    is_minutes = bool(np.max(t) > 1000)
    t_plot = t / 60.0 if is_minutes else t
    t_label = 'Time (min)' if is_minutes else 'Time (s)'

    # 图1: Cc 参数辨识
    plt.figure(figsize=(10, 6))
    # plt.plot(t_plot, theta_history[:, 0], 'b-', linewidth=1, alpha=0.5, label='Cc (raw)')
    plt.plot(t_plot, theta_smooth_history[:, 0], 'b-', linewidth=2.5, label='Cc (J/K)')
    plt.xlabel(t_label, fontsize=12)
    plt.ylabel('Cc (J/K)', fontsize=12)
    plt.title('Core Heat Capacity Identification', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 图2: Rc & Rs 参数辨识
    plt.figure(figsize=(10, 6))
    # plt.plot(t_plot, theta_history[:, 1], 'r-', linewidth=1, alpha=0.5, label='Rc (raw)')
    # plt.plot(t_plot, theta_history[:, 2], 'g-', linewidth=1, alpha=0.5, label='Rs (raw)')
    plt.plot(t_plot, theta_smooth_history[:, 1], 'r-', linewidth=2.5, label='Rc (J/k)')
    plt.plot(t_plot, theta_smooth_history[:, 2], 'g-', linewidth=2.5, label='Rs (J/k)')
    plt.xlabel(t_label, fontsize=12)
    plt.ylabel('Thermal Resistance (K/W)', fontsize=12)
    plt.title('Thermal Resistance Identification', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 图3: Ts & Tc 预测对比
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_plot, Ts_measured, 'k-', lw=2, label='Measured Ts', alpha=0.8)
    plt.plot(t_plot, Ts_pred, 'r--', lw=2, label='Predicted Ts', alpha=0.8)
    plt.plot(t_plot, Ta_measured, 'b:', lw=1.5, label='Ambient Ta', alpha=0.7)
    plt.fill_between(t_plot, Ts_measured, Ts_pred, alpha=0.2, color='orange', label='Prediction Error')
    plt.xlabel(t_label, fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Surface Temperature Prediction Comparison', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)

    plt.subplot(2, 1, 2)
    plt.plot(t_plot, Tc_pred, 'purple', lw=2, label='Predicted Tc', alpha=0.8)
    plt.plot(t_plot, Ts_measured, 'k-', lw=1.5, label='Measured Ts', alpha=0.6)
    plt.plot(t_plot, Ta_measured, 'b:', lw=1.5, label='Ambient Ta', alpha=0.7)
    plt.xlabel(t_label, fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Core Temperature Prediction', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 图4: 残差分析
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t_plot, residuals, 'k-', linewidth=1, alpha=0.8)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mad = np.median(np.abs(residuals - np.median(residuals)))
    std_err = np.std(residuals)
    plt.axhline(0, color='blue', linestyle='-', alpha=0.5)
    plt.axhline(std_err, color='r', linestyle='--', alpha=0.7, label=f'±1σ: {std_err:.3f}°C')
    plt.axhline(-std_err, color='r', linestyle='--', alpha=0.7)
    plt.fill_between(t_plot, -std_err, std_err, alpha=0.2, color='red')
    plt.xlabel(t_label, fontsize=12)
    plt.ylabel('Temperature Error (°C)', fontsize=12)
    plt.title(f'Prediction Residuals (RMSE: {rmse:.3f}°C, MAD: {mad:.3f}°C)', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label=f'Normal fit (μ={mu:.3f}, σ={sigma:.3f})')
    plt.xlabel('Residual (°C)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Residual Distribution', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    # 图5: 创新序列 + λ(k)
    plt.figure(figsize=(10, 6))
    innovation = np.array(rls.innovation_history)
    ax1 = plt.gca()
    ax1.semilogy(t_plot, innovation, 'purple', linewidth=1.5, alpha=0.8, label='Innovation (|e_huber|)')
    ax1.set_xlabel(t_label, fontsize=12)
    ax1.set_ylabel('Innovation (log)', fontsize=12)
    ax1.set_title('Innovation and VFF Lambda', fontsize=14)
    ax1.grid(True, linestyle=':', alpha=0.7)

    # twin y for lambda
    ax2 = ax1.twinx()
    if rls.lambda_history:
        lam = np.array(rls.lambda_history)
        ax2.plot(t_plot, lam, color='tab:orange', lw=1.5, alpha=0.9, label='Lambda(k)')
        ax2.set_ylabel('Lambda', fontsize=12)
        # 合并图例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()

    #图6: 条件数
    plt.figure(figsize=(10, 6))
    plt.semilogy(t_plot, rls.condition_number_history, 'orange', linewidth=2)
    plt.xlabel(t_label, fontsize=12)
    plt.ylabel('Condition Number', fontsize=12)
    plt.title('Covariance Matrix Condition Number', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 图7: 性能指标分析
    plot_performance_metrics_separate(rls, t_plot)


def plot_performance_metrics_separate(rls, t_plot):
    """性能指标分析 - 分开展示"""
    window = 100
    residuals = np.array(rls.residuals_history)

    # 图7.1: 滑动窗口RMSE & MAD & MAE
    plt.figure(figsize=(10, 6))
    if len(residuals) > window:
        sliding_rmse = []
        sliding_mad = []
        sliding_mae = []
        for i in range(window, len(residuals)):
            window_res = residuals[i - window:i]
            rmse = np.sqrt(np.mean(window_res ** 2))
            mad = np.median(np.abs(window_res - np.median(window_res)))
            mae = np.mean(np.abs(window_res))
            sliding_rmse.append(rmse)
            sliding_mad.append(mad)
            sliding_mae.append(mae)

        plt.plot(t_plot[window:], sliding_rmse, 'b-', lw=2, label='Sliding RMSE')
        plt.plot(t_plot[window:], sliding_mad, 'r--', lw=2, label='Sliding MAD')
        plt.plot(t_plot[window:], sliding_mae, 'g-.', lw=2, label='Sliding MAE')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Error Metric (°C)', fontsize=12)
        plt.title('Sliding Window Error Metrics', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.7)
    else:
        plt.text(0.5, 0.5, f'Not enough samples for window={window}', ha='center', va='center',
                 transform=plt.gca().transAxes)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 图7.2: 参数收敛（Cc的CV）
    # plt.figure(figsize=(10, 6))
    # theta_smooth = np.array(rls.theta_smooth_history)
    # if len(theta_smooth) > window:
    #     cv_cc = []
    #     for i in range(window, len(theta_smooth)):
    #         window_params = theta_smooth[i - window:i, 0]
    #         cv = np.std(window_params) / np.mean(window_params)
    #         cv_cc.append(cv)
    #     plt.semilogy(t_plot[window:], cv_cc, 'g-', lw=2)
    #     plt.axhline(0.005, color='r', linestyle='--', label='Convergence threshold')
    #     plt.xlabel('Time', fontsize=12)
    #     plt.ylabel('CV of Cc', fontsize=12)
    #     plt.title('Parameter Convergence (Cc)', fontsize=14)
    #     plt.legend(fontsize=10)
    #     plt.grid(True, alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    # 图7.3: 自相关
    # plt.figure(figsize=(10, 6))
    # if len(residuals) > 50:
    #     autocorr = signal.correlate(residuals, residuals, mode='full')
    #     autocorr = autocorr[autocorr.size // 2:]
    #     autocorr = autocorr / autocorr[0]
    #     lags = np.arange(min(50, len(autocorr)))
    #     plt.plot(lags, autocorr[:len(lags)], 'b-', lw=2)
    #     plt.axhline(0, color='k', linestyle='-', alpha=0.5)
    #     plt.axhline(0.2, color='r', linestyle='--', alpha=0.7, label='Significance level')
    #     plt.axhline(-0.2, color='r', linestyle='--', alpha=0.7)
    #     plt.xlabel('Lag', fontsize=12)
    #     plt.ylabel('Autocorrelation', fontsize=12)
    #     plt.title('Residual Autocorrelation', fontsize=14)
    #     plt.legend(fontsize=10)
    #     plt.grid(True, alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    # 图7.4: QQ图
    # plt.figure(figsize=(10, 6))
    # from scipy.stats import probplot
    # probplot(residuals, dist="norm", plot=plt)
    # plt.title('Q-Q Plot (Normality Check)', fontsize=14)
    # plt.grid(True, alpha=0.7)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    # 文件路径（请按实际环境修改）
    filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_n15_env.xlsx"
    soc_ocv_filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_n15_sococv.xlsx"

    try:
        rls_model, params = run_rls_identification(
            filepath=filepath,
            soc_ocv_filepath=soc_ocv_filepath,
            Cs_value=3.4,
            lambda_factor=0.999,  # 兜底
            P0=1e2,
            use_vff=True,  # 启用VFFRLS
            lambda_min=0.96,
            lambda_max=0.9995,
            vff_rho=0.6,
            vff_window=80
        )

        print("\n" + "=" * 50)
        print("VFFRLS IDENTIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()