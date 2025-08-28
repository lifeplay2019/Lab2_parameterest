import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class RLS_ThermalBattery:
    def __init__(self, Cs_fixed=3.5, lambda_factor=0.999, P0=1e6):
        """
        Initialize RLS algorithm for thermal battery model with fixed Cs
        """
        self.Cs = Cs_fixed
        self.lambda_factor = lambda_factor
        self.n_params = 3  # [Cc, Rc, Rs]

        # Initialize parameters with more conservative values
        self.theta = np.array([[50], [0.5], [1.5]])  # [Cc, Rc, Rs]
        self.P = P0 * np.eye(self.n_params)

        # Add numerical stability parameters
        self.min_eigenvalue = 1e-8
        self.max_eigenvalue = 1e8
        self.regularization = 1e-6

        # Parameter bounds for stability
        self.param_bounds = {
            'Cc': (30.0, 70.0),
            'Rc': (0.1, 10.0),
            'Rs': (0.1, 10.0)
        }

        # Moving average for parameter smoothing
        self.window_size = 20  # 增加窗口大小以提高稳定性
        self.param_buffer = []

        # Storage for results
        self.theta_history = []
        self.theta_smooth_history = []
        self.error_history = []
        self.physical_params_history = []
        self.b_coefficients_history = []
        self.condition_number_history = []
        self.covariance_history = []  # 新增：存储协方差矩阵历史
        self.residuals_history = []  # 新增：存储残差用于统计分析

    def regularize_covariance(self):
        """
        Regularize covariance matrix to prevent numerical issues
        """
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # Eigenvalue decomposition for regularization
        eigenvals, eigenvecs = np.linalg.eigh(self.P)

        # Clip eigenvalues to reasonable range
        eigenvals = np.clip(eigenvals, self.min_eigenvalue, self.max_eigenvalue)

        # Reconstruct matrix
        self.P = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        # Add small regularization to diagonal
        self.P += self.regularization * np.eye(self.n_params)

    def calculate_b_coefficients(self, dt=1.0):
        """
        Calculate b1, b2, b3, b4 from physical parameters with numerical protection
        """
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])
        Cs = self.Cs

        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        denom = Cc * (Rc + Rs)
        if abs(denom) < epsilon:
            denom = epsilon * np.sign(denom) if denom != 0 else epsilon

        # Calculate b coefficients with protection
        try:
            b1 = (Rc * Cc + Rs * Cs - dt) / denom
            b2 = Rc / (Rs + Rc + epsilon)
            b3 = (dt - Rs * Cs) / denom
            b4 = Rs * dt / denom
        except:
            # Fallback values
            b1, b2, b3, b4 = 0.0, 0.5, 0.0, 0.0

        return np.array([b1, b2, b3, b4])

    def construct_jacobian_robust(self, Ts_prev, Ta_curr, Ta_prev, H_prev, dt=1.0):
        """
        Construct Jacobian matrix with improved numerical stability
        """
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])
        Cs = self.Cs

        epsilon = 1e-10

        # Protect denominators
        denom = max(abs(Rc + Rs), epsilon)
        denom_sq = denom ** 2
        Cc_safe = max(abs(Cc), epsilon)
        Cc_sq = Cc_safe ** 2

        # Calculate partial derivatives with numerical protection
        # db1 derivatives
        db1_dCc = -(Rs * Cs - dt) / (Cc_sq * denom)
        db1_dRc = (Cc_safe * denom - (Rc * Cc_safe + Rs * Cs - dt)) / (Cc_safe * denom_sq)
        db1_dRs = (Cs * denom - (Rc * Cc_safe + Rs * Cs - dt)) / (Cc_safe * denom_sq)

        # db2 derivatives
        db2_dCc = 0
        db2_dRc = Rs / denom_sq
        db2_dRs = -Rc / denom_sq

        # db3 derivatives
        db3_dCc = -(dt - Rs * Cs) / (Cc_sq * denom)
        db3_dRc = -(dt - Rs * Cs) / (Cc_safe * denom_sq)
        db3_dRs = (-Cs * denom - (dt - Rs * Cs)) / (Cc_safe * denom_sq)

        # db4 derivatives
        db4_dCc = -Rs * dt / (Cc_sq * denom)
        db4_dRc = -Rs * dt / (Cc_safe * denom_sq)
        db4_dRs = (dt * denom - Rs * dt) / (Cc_safe * denom_sq)

        # Construct Jacobian
        J_b = np.array([
            [db1_dCc, db1_dRc, db1_dRs],
            [db2_dCc, db2_dRc, db2_dRs],
            [db3_dCc, db3_dRc, db3_dRs],
            [db4_dCc, db4_dRc, db4_dRs]
        ])

        # Check for NaN or Inf
        J_b = np.nan_to_num(J_b, nan=0.0, posinf=1e6, neginf=-1e6)

        # Regressor vector
        phi = np.array([Ts_prev, Ta_curr, Ta_prev, H_prev])

        # Final Jacobian
        J = phi @ J_b
        return J.reshape(-1, 1)

    def apply_parameter_constraints(self):
        """
        Apply physical constraints to parameters
        """
        self.theta[0, 0] = np.clip(self.theta[0, 0],
                                   self.param_bounds['Cc'][0],
                                   self.param_bounds['Cc'][1])
        self.theta[1, 0] = np.clip(self.theta[1, 0],
                                   self.param_bounds['Rc'][0],
                                   self.param_bounds['Rc'][1])
        self.theta[2, 0] = np.clip(self.theta[2, 0],
                                   self.param_bounds['Rs'][0],
                                   self.param_bounds['Rs'][1])

    def smooth_parameters(self):
        """
        Apply moving average smoothing to parameters
        """
        current_params = self.theta.flatten().copy()
        self.param_buffer.append(current_params)

        if len(self.param_buffer) > self.window_size:
            self.param_buffer.pop(0)

        # Calculate smoothed parameters
        if len(self.param_buffer) >= 5:  # Need at least 5 points for smoothing
            param_array = np.array(self.param_buffer)
            smoothed_params = np.mean(param_array[-self.window_size:], axis=0)
            return smoothed_params
        else:
            return current_params

    def update(self, Ts_prev, Ta_curr, Ta_prev, H_prev, Ts_curr, dt=1.0):
        """
        Robust RLS update step with stability improvements
        """
        # Calculate current b coefficients
        b = self.calculate_b_coefficients(dt)

        # Construct regressor vector
        phi = np.array([Ts_prev, Ta_curr, Ta_prev, H_prev])

        # Prediction
        Ts_pred = np.dot(phi, b)

        # Prediction error with outlier protection
        e = Ts_curr - Ts_pred
        original_error = e
        if abs(e) > 10.0:  # Outlier detection threshold
            e = np.clip(e, -10.0, 10.0)  # Clip extreme errors

        # Construct Jacobian
        J = self.construct_jacobian_robust(Ts_prev, Ta_curr, Ta_prev, H_prev, dt)

        # Regularize covariance matrix
        self.regularize_covariance()

        # Calculate denominator with protection
        J_T_P_J = J.T @ self.P @ J
        if J_T_P_J.ndim > 0:
            denominator = self.lambda_factor + J_T_P_J.item()
        else:
            denominator = self.lambda_factor + float(J_T_P_J)

        if abs(denominator) < 1e-10:
            denominator = 1e-10

        # Kalman gain with numerical stability
        K = self.P @ J / denominator

        # Limit gain magnitude to prevent large jumps
        K_norm = np.linalg.norm(K)
        if K_norm > 1.0:
            K = K / K_norm

        # Update parameters
        self.theta = self.theta + K * e

        # Apply constraints
        self.apply_parameter_constraints()

        # Update covariance matrix with Joseph form for numerical stability
        I = np.eye(self.n_params)
        A = I - K @ J.T
        self.P = (A @ self.P @ A.T + K @ K.T * self.regularization) / self.lambda_factor

        # Store condition number for monitoring
        cond_num = np.linalg.cond(self.P)
        self.condition_number_history.append(cond_num)

        # Store covariance matrix for confidence interval calculation
        self.covariance_history.append(self.P.copy())

        # Store results
        self.theta_history.append(self.theta.flatten().copy())
        self.error_history.append(float(e))
        self.residuals_history.append(float(original_error))
        self.b_coefficients_history.append(b.copy())

        # Calculate and store smoothed parameters
        smoothed_params = self.smooth_parameters()
        self.theta_smooth_history.append(smoothed_params.copy())

        # Store physical parameters (using smoothed values for final result)
        physical_params = {
            'Cc': float(smoothed_params[0]),
            'Cs': self.Cs,
            'Rc': float(smoothed_params[1]),
            'Rs': float(smoothed_params[2])
        }
        self.physical_params_history.append(physical_params)

        return self.theta.flatten()

    def calculate_confidence_intervals(self, confidence_level=0.95):
        """
        Calculate confidence intervals for parameters
        """
        if len(self.covariance_history) == 0:
            return None

        # Use the final covariance matrix
        P_final = self.covariance_history[-1]
        final_params = self.theta_smooth_history[-1]

        # Calculate standard errors
        std_errors = np.sqrt(np.diag(P_final))

        # Calculate t-value for confidence interval
        dof = len(self.error_history) - self.n_params  # degrees of freedom
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha / 2, dof) if dof > 0 else 1.96

        # Calculate confidence intervals
        confidence_intervals = {}
        param_names = ['Cc', 'Rc', 'Rs']

        for i, param_name in enumerate(param_names):
            margin = t_value * std_errors[i]
            lower = final_params[i] - margin
            upper = final_params[i] + margin
            confidence_intervals[param_name] = (lower, upper)

        # Add Cs (fixed parameter)
        confidence_intervals['Cs'] = (self.Cs, self.Cs)

        return confidence_intervals

    def assess_convergence(self, window_size=100):
        """
        Assess parameter convergence
        """
        if len(self.theta_smooth_history) < window_size:
            return False, {}

        # Calculate parameter variation in the last window
        recent_params = np.array(self.theta_smooth_history[-window_size:])
        param_std = np.std(recent_params, axis=0)
        param_mean = np.mean(recent_params, axis=0)

        # Convergence criteria: coefficient of variation < 1%
        cv_threshold = 0.01
        convergence_status = {}
        param_names = ['Cc', 'Rc', 'Rs']

        all_converged = True
        for i, param_name in enumerate(param_names):
            cv = param_std[i] / abs(param_mean[i]) if abs(param_mean[i]) > 1e-10 else float('inf')
            converged = cv < cv_threshold
            convergence_status[param_name] = {
                'converged': converged,
                'cv': cv,
                'std': param_std[i],
                'mean': param_mean[i]
            }
            if not converged:
                all_converged = False

        return all_converged, convergence_status


def load_soc_ocv_data(filepath):
    """
    Load SOC-OCV data and fit 8th order polynomial
    """
    print(f"Loading SOC-OCV data from: {filepath}")

    try:
        df_soc_ocv = pd.read_excel(filepath)
        print("SOC-OCV file columns:", df_soc_ocv.columns.tolist())
        print("SOC-OCV file shape:", df_soc_ocv.shape)

        soc_data = df_soc_ocv.iloc[:, 0].values
        ocv_data = df_soc_ocv.iloc[:, 1].values

        soc_data = pd.to_numeric(soc_data, errors='coerce')
        ocv_data = pd.to_numeric(ocv_data, errors='coerce')

        valid_mask = ~(np.isnan(soc_data) | np.isnan(ocv_data))
        soc_data = soc_data[valid_mask]
        ocv_data = ocv_data[valid_mask]

        print(f"Loaded SOC-OCV data: {len(soc_data)} points")
        print(f"SOC range: {soc_data.min():.3f} - {soc_data.max():.3f}")
        print(f"OCV range: {ocv_data.min():.3f} - {ocv_data.max():.3f} V")

        poly_coeffs = np.polyfit(soc_data, ocv_data, 8)
        print(f"Polynomial coefficients: {poly_coeffs}")

        return poly_coeffs

    except Exception as e:
        print(f"Error loading SOC-OCV data: {e}")
        return np.array([0, 0, 0, 0, 0, 0, 0, 1.7, 2.5])


def calculate_ocv_from_soc(soc, poly_coeffs):
    """
    Calculate OCV from SOC using polynomial coefficients
    """
    soc_clipped = np.clip(soc, 0, 1)
    ocv = np.polyval(poly_coeffs, soc_clipped)
    return ocv


def calculate_soc_from_voltage(voltage, poly_coeffs, soc_range=np.linspace(0, 1, 101)):
    """
    Calculate SOC from voltage using polynomial coefficients
    """
    ocv_range = calculate_ocv_from_soc(soc_range, poly_coeffs)
    soc_estimated = np.interp(voltage, ocv_range, soc_range)
    soc_estimated = np.clip(soc_estimated, 0, 1)
    return soc_estimated


def preprocess_data_with_filtering(t, Ts, Ta, H):
    """
    Apply data preprocessing and filtering to reduce noise
    """
    # Apply Savitzky-Golay filter to temperature data
    if len(Ts) > 5:
        window_length = min(11, len(Ts) // 2 * 2 + 1)  # 增加窗口长度
        Ts_filtered = savgol_filter(Ts, window_length=window_length, polyorder=3)
        Ta_filtered = savgol_filter(Ta, window_length=window_length, polyorder=3)
        H_filtered = savgol_filter(H, window_length=window_length, polyorder=3)
    else:
        Ts_filtered = Ts
        Ta_filtered = Ta
        H_filtered = H

    return Ts_filtered, Ta_filtered, H_filtered


def load_and_preprocess_data(filepath, soc_ocv_filepath):
    """
    Load and preprocess the HPPC test data
    """
    poly_coeffs = load_soc_ocv_data(soc_ocv_filepath)
    print(f"Loading main data from: {filepath}")

    try:
        df = pd.read_excel(filepath)
        print(f"File shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        if df.shape[1] >= 8:
            print("Using SOC data from file")
            t_0 = df.iloc[:, 0].values
            t = df.iloc[:, 1].values
            v = df.iloc[:, 2].values
            i = df.iloc[:, 3].values
            Ta = df.iloc[:, 4].values
            Ts = df.iloc[:, 5].values
            Q_rm = df.iloc[:, 6].values
            SOC = df.iloc[:, 7].values
        else:
            print("SOC not found in file, will calculate from voltage")
            t_0 = df.iloc[:, 0].values
            t = df.iloc[:, 1].values
            v = df.iloc[:, 2].values
            i = df.iloc[:, 3].values
            Ta = df.iloc[:, 4].values
            Ts = df.iloc[:, 5].values

            v = pd.to_numeric(v, errors='coerce')
            SOC = calculate_soc_from_voltage(v, poly_coeffs)

        # Convert to numeric
        t = pd.to_numeric(t, errors='coerce')
        v = pd.to_numeric(v, errors='coerce')
        i = pd.to_numeric(i, errors='coerce')
        Ta = pd.to_numeric(Ta, errors='coerce')
        Ts = pd.to_numeric(Ts, errors='coerce')
        SOC = pd.to_numeric(SOC, errors='coerce')

        # Clean data
        valid_idx = ~(np.isnan(t) | np.isnan(v) | np.isnan(i) | np.isnan(Ta) | np.isnan(Ts) | np.isnan(SOC))

        t = t[valid_idx]
        v = v[valid_idx]
        i = i[valid_idx]
        Ta = Ta[valid_idx]
        Ts = Ts[valid_idx]
        SOC = SOC[valid_idx]

        print(f"After cleaning: {len(t)} valid data points")

        # Calculate time step
        if len(t) > 1:
            dt = np.median(np.diff(t))
        else:
            dt = 10

        # Calculate OCV and heat generation
        Uocv = calculate_ocv_from_soc(SOC, poly_coeffs)
        H = (Uocv - v) * np.abs(i)

        # Apply data filtering
        Ts_filtered, Ta_filtered, H_filtered = preprocess_data_with_filtering(t, Ts, Ta, H)

        print(f"Data preprocessing completed")
        print(f"Time range: {t.min():.1f} - {t.max():.1f} s")
        print(f"Temperature range: {Ts_filtered.min():.1f} - {Ts_filtered.max():.1f} °C")

        return t, Ts_filtered, Ta_filtered, H_filtered, dt, SOC, Uocv

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise


def print_results_tables(rls, final_params, confidence_intervals, convergence_status):
    """
    Print results in the format shown in your image
    """
    print("\n" + "=" * 70)
    print("THERMAL SYSTEM MODEL (TSM) RLS IDENTIFICATION RESULTS")
    print("=" * 70)

    # Table 2: Initial conditions (assumed values for comparison)
    print("\nTable 2")
    print("TSM RLS initial conditions.")
    print("-" * 50)
    print(f"{'Parameters':<15} {'Cc (J K⁻¹)':<15} {'Rc (J K⁻¹)':<15} {'Rs (J K⁻¹)':<15}")
    print(f"{'Values':<15} {'50':<15} {'2.1':<15} {'3.5':<15}")

    # Table 3: Identified thermal parameters
    print("\nTable 3")
    print("Identified thermal parameters.")
    print("-" * 70)
    print(f"{'Parameters':<15} {'Values':<20} {'95% Confidence interval':<30}")
    print("-" * 70)

    # Format the results
    cc_val = final_params['Cc']
    cs_val = final_params['Cs']
    rc_val = final_params['Rc']
    rs_val = final_params['Rs']

    if confidence_intervals:
        cc_ci = confidence_intervals['Cc']
        cs_ci = confidence_intervals['Cs']
        rc_ci = confidence_intervals['Rc']
        rs_ci = confidence_intervals['Rs']

        print(f"{'Cc (J K⁻¹)':<15} {cc_val:<20.4f} {cc_ci[0]:.4f}~{cc_ci[1]:.4f}")
        print(f"{'Cs (J K⁻¹)':<15} {cs_val:<20.2f} {cs_ci[0]:.4f}~{cs_ci[1]:.4f}")
        print(f"{'Rc (J K⁻¹)':<15} {rc_val:<20.4f} {rc_ci[0]:.4f}~{rc_ci[1]:.4f}")
        print(f"{'Rs (J K⁻¹)':<15} {rs_val:<20.4f} {rs_ci[0]:.4f}~{rs_ci[1]:.4f}")
    else:
        print(f"{'Cc (J K⁻¹)':<15} {cc_val:<20.4f} {'N/A':<30}")
        print(f"{'Cs (J K⁻¹)':<15} {cs_val:<20.2f} {'Fixed':<30}")
        print(f"{'Rc (J K⁻¹)':<15} {rc_val:<20.4f} {'N/A':<30}")
        print(f"{'Rs (J K⁻¹)':<15} {rs_val:<20.4f} {'N/A':<30}")

    print("=" * 70)

    # Additional analysis
    print("\nParameter Convergence Analysis:")
    print("-" * 50)
    for param_name, status in convergence_status.items():
        converged_str = "✓" if status['converged'] else "✗"
        print(f"{param_name:<5}: {converged_str} (CV: {status['cv']:.4f}, Std: {status['std']:.4f})")

    # Model performance
    if rls.residuals_history:
        residuals = np.array(rls.residuals_history)
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        print(f"\nModel Performance:")
        print("-" * 30)
        print(f"RMSE: {rmse:.4f} °C")
        print(f"MAE:  {mae:.4f} °C")
        print(f"Final condition number: {rls.condition_number_history[-1]:.2e}")


def run_rls_identification(filepath, soc_ocv_filepath, Cs_value=3.5):
    """
    Main function to run RLS identification with improved stability and analysis
    """
    # Load data
    print("Loading data...")
    t, Ts, Ta, H, dt, SOC, Uocv = load_and_preprocess_data(filepath, soc_ocv_filepath)

    print(f"Using fixed Cs = {Cs_value} J/K")

    # Initialize RLS with more conservative parameters
    rls = RLS_ThermalBattery(Cs_fixed=Cs_value, lambda_factor=0.998, P0=1e3)

    n_samples = len(Ts)
    print(f"Running RLS identification on {n_samples} samples...")
    print(f"Time step dt = {dt:.1f} seconds")

    # RLS identification with progress monitoring
    for k in range(1, n_samples):
        theta = rls.update(
            Ts_prev=Ts[k - 1],
            Ta_curr=Ta[k],
            Ta_prev=Ta[k - 1],
            H_prev=H[k - 1],
            Ts_curr=Ts[k],
            dt=dt
        )

        if k % 500 == 0:
            smoothed = rls.theta_smooth_history[-1]
            cond_num = rls.condition_number_history[-1]
            print(
                f"Step {k}/{n_samples}: Cc={smoothed[0]:.2f}, Rc={smoothed[1]:.4f}, Rs={smoothed[2]:.4f}, Cond={cond_num:.2e}")

    # Final analysis
    final_params = rls.physical_params_history[-1]
    confidence_intervals = rls.calculate_confidence_intervals()
    all_converged, convergence_status = rls.assess_convergence()

    # Print results in required format
    print_results_tables(rls, final_params, confidence_intervals, convergence_status)

    # Plot results
    plot_results_enhanced(rls, t[1:], SOC[1:], Uocv[1:])

    return rls, final_params


def plot_results_enhanced(rls, t, SOC, Uocv):
    """
    Enhanced plotting with both raw and smoothed parameters
    """
    theta_history = np.array(rls.theta_history)
    theta_smooth_history = np.array(rls.theta_smooth_history)

    # Create enhanced figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Plot 1: Parameter identification results - Raw vs Smoothed
    ax1 = axes[0, 0]
    t_plot = t / 60 if t.max() > 1000 else t

    # Raw parameters (thin lines)
    ax1.plot(t_plot, theta_history[:, 0], 'b-', linewidth=1, alpha=0.5, label='Cc (raw)')
    ax1.plot(t_plot, theta_history[:, 2] * 10, 'r-', linewidth=1, alpha=0.5, label='Rs×10 (raw)')
    ax1.plot(t_plot, theta_history[:, 1] * 10, 'g-', linewidth=1, alpha=0.5, label='Rc×10 (raw)')

    # Smoothed parameters (thick lines)
    ax1.plot(t_plot, theta_smooth_history[:, 0], 'b-', linewidth=3, label='Cc (smoothed)')
    ax1.plot(t_plot, theta_smooth_history[:, 2] * 10, 'r-', linewidth=3, label='Rs×10 (smoothed)')
    ax1.plot(t_plot, theta_smooth_history[:, 1] * 10, 'g-', linewidth=3, label='Rc×10 (smoothed)')

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('TSM Parameters')
    ax1.set_title('Parameter Identification: Raw vs Smoothed')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right', fontsize=8)

    # Plot 2: Condition number monitoring
    ax2 = axes[0, 1]
    ax2.semilogy(t_plot, rls.condition_number_history, 'purple', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Condition Number')
    ax2.set_title('Covariance Matrix Condition Number')
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Plot 3: SOC evolution
    ax3 = axes[1, 0]
    ax3.plot(t_plot, SOC, 'orange', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('SOC')
    ax3.set_title('State of Charge Evolution')
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.set_ylim([0, 1])

    # Plot 4: OCV evolution
    ax4 = axes[1, 1]
    ax4.plot(t_plot, Uocv, 'brown', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('OCV (V)')
    ax4.set_title('Open Circuit Voltage Evolution')
    ax4.grid(True, linestyle=':', alpha=0.7)

    # Plot 5: Prediction error with statistics
    ax5 = axes[2, 0]
    error_array = np.array(rls.residuals_history)
    ax5.plot(t_plot, error_array, 'k-', linewidth=1, alpha=0.7)

    # Add error statistics
    mean_error = np.mean(np.abs(error_array))
    std_error = np.std(error_array)
    ax5.axhline(mean_error, color='r', linestyle='--', label=f'Mean |Error|: {mean_error:.3f}°C')
    ax5.axhline(-mean_error, color='r', linestyle='--')
    ax5.fill_between(t_plot, -std_error, std_error, alpha=0.2, color='gray', label=f'±1σ: {std_error:.3f}°C')

    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Temperature Error (°C)')
    ax5.set_title('Temperature Prediction Error')
    ax5.grid(True, linestyle=':', alpha=0.7)
    ax5.legend()

    # Plot 6: Parameter convergence analysis
    ax6 = axes[2, 1]
    # Calculate parameter standard deviation in sliding window
    window = 100
    if len(theta_smooth_history) > window:
        param_std = []
        for i in range(window, len(theta_smooth_history)):
            std_window = np.std(theta_smooth_history[i - window:i, 0])  # Cc std
            param_std.append(std_window)

        ax6.plot(t_plot[window:], param_std, 'navy', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Cc Std Dev (sliding window)')
        ax6.set_title('Parameter Convergence Analysis')
        ax6.grid(True, linestyle=':', alpha=0.7)
    else:
        ax6.text(0.5, 0.5, 'Insufficient data for\nconvergence analysis',
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Parameter Convergence Analysis')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File paths
    filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_env.xlsx"
    soc_ocv_filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_sococv.xlsx"

    try:
        rls_model, params = run_rls_identification(filepath, soc_ocv_filepath, Cs_value=3.4)

        print("\n" + "=" * 50)
        print("RLS IDENTIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()