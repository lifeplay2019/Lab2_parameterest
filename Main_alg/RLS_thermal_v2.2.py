import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RLS_ThermalBattery:
    def __init__(self, Cs_fixed=3.5, lambda_factor=0.99, P0=1e6):
        """
        Initialize RLS algorithm for thermal battery model with fixed Cs

        Parameters:
        Cs_fixed: fixed surface heat capacity (J/K)
        lambda_factor: forgetting factor (0 < λ ≤ 1)
        P0: initial covariance matrix value
        """
        self.Cs = Cs_fixed  # Fixed value from the paper
        self.lambda_factor = lambda_factor
        self.n_params = 3  # [Cc, Rc, Rs] - only 3 parameters to estimate

        # Initialize parameters with values from Table 2
        self.theta = np.array([[50.0], [2.1], [3.5]])  # [Cc, Rc, Rs]
        self.P = P0 * np.eye(self.n_params)  # Covariance matrix

        # Storage for results
        self.theta_history = []
        self.error_history = []
        self.physical_params_history = []
        self.b_coefficients_history = []

    def calculate_b_coefficients(self, dt=1.0):
        """
        Calculate b1, b2, b3, b4 from physical parameters
        Based on equations (17) from the paper
        """
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])
        Cs = self.Cs

        # Avoid division by zero
        if abs(Cc * (Rc + Rs)) < 1e-10:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Calculate b coefficients from physical parameters (equation 17)
        b1 = (Rc * Cc + Rs * Cs - dt) / (Cc * (Rc + Rs))
        b2 = Rc / (Rs + Rc)
        b3 = (dt - Rs * Cs) / (Cc * (Rc + Rs))
        b4 = Rs * dt / (Cc * (Rc + Rs))

        return np.array([b1, b2, b3, b4])

    def construct_jacobian(self, Ts_prev, Ta_curr, Ta_prev, H_prev, dt=1.0):
        """
        Construct Jacobian matrix for the relationship between physical and b parameters
        """
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])
        Cs = self.Cs

        # Avoid division by zero
        denom = (Rc + Rs)
        if abs(denom) < 1e-10:
            denom = 1e-10
        denom_sq = denom ** 2

        # Partial derivatives of b coefficients w.r.t physical parameters
        # db1/dCc, db1/dRc, db1/dRs
        db1_dCc = -(Rs * Cs - dt) / (Cc ** 2 * denom)
        db1_dRc = (Cc * denom - (Rc * Cc + Rs * Cs - dt)) / (Cc * denom_sq)
        db1_dRs = (Cs * denom - (Rc * Cc + Rs * Cs - dt)) / (Cc * denom_sq)

        # db2/dCc, db2/dRc, db2/dRs
        db2_dCc = 0
        db2_dRc = Rs / denom_sq
        db2_dRs = -Rc / denom_sq

        # db3/dCc, db3/dRc, db3/dRs
        db3_dCc = -(dt - Rs * Cs) / (Cc ** 2 * denom)
        db3_dRc = -(dt - Rs * Cs) / (Cc * denom_sq)
        db3_dRs = (-Cs * denom - (dt - Rs * Cs)) / (Cc * denom_sq)

        # db4/dCc, db4/dRc, db4/dRs
        db4_dCc = -Rs * dt / (Cc ** 2 * denom)
        db4_dRc = -Rs * dt / (Cc * denom_sq)
        db4_dRs = (dt * denom - Rs * dt) / (Cc * denom_sq)

        # Construct Jacobian matrix
        J_b = np.array([
            [db1_dCc, db1_dRc, db1_dRs],
            [db2_dCc, db2_dRc, db2_dRs],
            [db3_dCc, db3_dRc, db3_dRs],
            [db4_dCc, db4_dRc, db4_dRs]
        ])

        # Regressor vector
        phi = np.array([Ts_prev, Ta_curr, Ta_prev, H_prev])

        # Final Jacobian for output
        J = phi @ J_b

        return J.reshape(-1, 1)

    def update(self, Ts_prev, Ta_curr, Ta_prev, H_prev, Ts_curr, dt=1.0):
        """
        RLS update step
        """
        # Calculate current b coefficients
        b = self.calculate_b_coefficients(dt)

        # Construct regressor vector (equation 16)
        phi = np.array([Ts_prev, Ta_curr, Ta_prev, H_prev])

        # Prediction
        Ts_pred = np.dot(phi, b)

        # Prediction error
        e = Ts_curr - Ts_pred

        # Construct Jacobian
        J = self.construct_jacobian(Ts_prev, Ta_curr, Ta_prev, H_prev, dt)

        # Kalman gain
        denominator = self.lambda_factor + float(J.T @ self.P @ J)
        if abs(denominator) < 1e-10:
            denominator = 1e-10
        K = self.P @ J / denominator

        # Update parameters
        self.theta = self.theta + K * e

        # Ensure parameters stay positive and within reasonable bounds
        self.theta[0, 0] = np.clip(self.theta[0, 0], 10.0, 100.0)  # Cc
        self.theta[1, 0] = np.clip(self.theta[1, 0], 0.1, 10.0)  # Rc
        self.theta[2, 0] = np.clip(self.theta[2, 0], 0.1, 10.0)  # Rs

        # Update covariance matrix
        self.P = (self.P - K @ J.T @ self.P) / self.lambda_factor

        # Store results
        self.theta_history.append(self.theta.flatten().copy())
        self.error_history.append(float(e))
        self.b_coefficients_history.append(b.copy())

        # Store physical parameters
        physical_params = {
            'Cc': float(self.theta[0, 0]),
            'Cs': self.Cs,
            'Rc': float(self.theta[1, 0]),
            'Rs': float(self.theta[2, 0])
        }
        self.physical_params_history.append(physical_params)

        return self.theta.flatten()


def load_soc_ocv_data(filepath):
    """
    Load SOC-OCV data and fit 8th order polynomial
    """
    print(f"Loading SOC-OCV data from: {filepath}")

    try:
        # Load SOC-OCV data
        df_soc_ocv = pd.read_excel(filepath)

        print("SOC-OCV file columns:", df_soc_ocv.columns.tolist())
        print("SOC-OCV file shape:", df_soc_ocv.shape)

        # Extract SOC and OCV columns
        soc_data = df_soc_ocv.iloc[:, 0].values  # SOC column
        ocv_data = df_soc_ocv.iloc[:, 1].values  # OCV column

        # Convert to numpy arrays and ensure they are numeric
        soc_data = pd.to_numeric(soc_data, errors='coerce')
        ocv_data = pd.to_numeric(ocv_data, errors='coerce')

        # Remove any NaN values
        valid_mask = ~(np.isnan(soc_data) | np.isnan(ocv_data))
        soc_data = soc_data[valid_mask]
        ocv_data = ocv_data[valid_mask]

        print(f"Loaded SOC-OCV data: {len(soc_data)} points")
        print(f"SOC range: {soc_data.min():.3f} - {soc_data.max():.3f}")
        print(f"OCV range: {ocv_data.min():.3f} - {ocv_data.max():.3f} V")

        # Fit 8th order polynomial: OCV = f(SOC)
        poly_coeffs = np.polyfit(soc_data, ocv_data, 8)
        print(f"Polynomial coefficients: {poly_coeffs}")

        return poly_coeffs

    except Exception as e:
        print(f"Error loading SOC-OCV data: {e}")
        # Fallback coefficients for typical Li-ion battery
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
    Calculate SOC from voltage using polynomial coefficients (inverse function)
    This uses interpolation to find SOC given voltage
    """
    # Generate OCV values for the SOC range
    ocv_range = calculate_ocv_from_soc(soc_range, poly_coeffs)

    # Use interpolation to find SOC from voltage
    soc_estimated = np.interp(voltage, ocv_range, soc_range)

    # Clip to valid SOC range
    soc_estimated = np.clip(soc_estimated, 0, 1)

    return soc_estimated


def load_and_preprocess_data(filepath, soc_ocv_filepath):
    """
    Load and preprocess the HPPC test data - FIXED VERSION
    """
    # Load SOC-OCV polynomial coefficients
    poly_coeffs = load_soc_ocv_data(soc_ocv_filepath)

    print(f"Loading main data from: {filepath}")

    try:
        # Load the main data file
        df = pd.read_excel(filepath)
        print(f"File shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("First few rows:")
        print(df.head())

        # Based on the columns shown: ['t_0', 't', 'v', 'i', 'Ta', 'Ts', 'Unnamed: 6']
        # But looking at your data description, we need to check if there's SOC data

        # Check if we have 8 columns (including SOC)
        if df.shape[1] >= 8:
            # We have SOC data in the file
            print("Using SOC data from file")
            t_0 = df.iloc[:, 0].values  # Original time
            t = df.iloc[:, 1].values  # time
            v = df.iloc[:, 2].values  # voltage
            i = df.iloc[:, 3].values  # current
            Ta = df.iloc[:, 4].values  # ambient temperature
            Ts = df.iloc[:, 5].values  # surface temperature
            Q_rm = df.iloc[:, 6].values  # remaining capacity
            SOC = df.iloc[:, 7].values  # SOC from file
        else:
            # We don't have SOC, need to calculate it using polynomial
            print("SOC not found in file, will calculate from voltage using polynomial")
            t_0 = df.iloc[:, 0].values  # Original time
            t = df.iloc[:, 1].values  # time
            v = df.iloc[:, 2].values  # voltage
            i = df.iloc[:, 3].values  # current
            Ta = df.iloc[:, 4].values  # ambient temperature
            Ts = df.iloc[:, 5].values  # surface temperature

            # Convert voltage to numeric first
            v = pd.to_numeric(v, errors='coerce')

            # Calculate SOC from voltage using the polynomial function
            SOC = calculate_soc_from_voltage(v, poly_coeffs)
            print(f"Calculated SOC from voltage using polynomial")

        print(f"Data shapes after extraction:")
        print(f"  Time: {len(t)}")
        print(f"  Voltage: {len(v)}")
        print(f"  Current: {len(i)}")
        print(f"  Ambient temp: {len(Ta)}")
        print(f"  Surface temp: {len(Ts)}")
        print(f"  SOC: {len(SOC)}")

        # Convert to numeric
        t = pd.to_numeric(t, errors='coerce')
        v = pd.to_numeric(v, errors='coerce')
        i = pd.to_numeric(i, errors='coerce')
        Ta = pd.to_numeric(Ta, errors='coerce')
        Ts = pd.to_numeric(Ts, errors='coerce')
        SOC = pd.to_numeric(SOC, errors='coerce')

        # Clean data - remove NaN values
        valid_idx = ~(np.isnan(t) | np.isnan(v) | np.isnan(i) | np.isnan(Ta) | np.isnan(Ts) | np.isnan(SOC))

        print(f"Valid data points: {np.sum(valid_idx)} out of {len(valid_idx)}")

        if np.sum(valid_idx) == 0:
            raise ValueError("No valid data points after cleaning")

        # Apply cleaning
        t = t[valid_idx]
        v = v[valid_idx]
        i = i[valid_idx]
        Ta = Ta[valid_idx]
        Ts = Ts[valid_idx]
        SOC = SOC[valid_idx]

        print(f"After cleaning: {len(t)} valid data points")
        print(f"Time range: {t.min():.1f} - {t.max():.1f} s")
        print(f"Voltage range: {v.min():.3f} - {v.max():.3f} V")
        print(f"Current range: {i.min():.3f} - {i.max():.3f} A")
        print(f"Ta range: {Ta.min():.1f} - {Ta.max():.1f} °C")
        print(f"Ts range: {Ts.min():.1f} - {Ts.max():.1f} °C")
        print(f"SOC range: {SOC.min():.3f} - {SOC.max():.3f}")

        # Calculate time step
        if len(t) > 1:
            dt = np.median(np.diff(t))
            print(f"Median time step: {dt:.1f} s")
        else:
            dt = 10

        # Calculate OCV from SOC using the polynomial
        Uocv = calculate_ocv_from_soc(SOC, poly_coeffs)
        print(f"OCV range: {Uocv.min():.3f} - {Uocv.max():.3f} V")

        # Calculate heat generation H = (Uocv - Ut) * |I|
        H = (Uocv - v) * np.abs(i)
        print(f"Heat generation range: {H.min():.6f} - {H.max():.6f} W")

        return t, Ts, Ta, H, dt, SOC, Uocv

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_rls_identification(filepath, soc_ocv_filepath, Cs_value=3.5):
    """
    Main function to run RLS identification
    """
    # Load data
    print("Loading data...")
    t, Ts, Ta, H, dt, SOC, Uocv = load_and_preprocess_data(filepath, soc_ocv_filepath)

    print(f"Using fixed Cs = {Cs_value} J/K")

    # Initialize RLS
    rls = RLS_ThermalBattery(Cs_fixed=Cs_value, lambda_factor=0.995, P0=1e4)

    n_samples = len(Ts)
    print(f"Running RLS identification on {n_samples} samples...")
    print(f"Time step dt = {dt:.1f} seconds")

    # RLS identification starting from k=1
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
            print(f"Step {k}/{n_samples}: Cc={theta[0]:.2f}, Rc={theta[1]:.4f}, Rs={theta[2]:.4f}")

    # Final results
    final_params = rls.physical_params_history[-1]
    final_b = rls.calculate_b_coefficients(dt)

    print("\n" + "=" * 60)
    print("FINAL IDENTIFICATION RESULTS")
    print("=" * 60)
    print(f"{'Parameter':<15} {'Value':<15} {'Unit'}")
    print("-" * 60)
    print(f"{'Cc':<15} {final_params['Cc']:<15.3f} {'J/K'}")
    print(f"{'Cs':<15} {final_params['Cs']:<15.3f} {'J/K (fixed)'}")
    print(f"{'Rc':<15} {final_params['Rc']:<15.4f} {'K/W'}")
    print(f"{'Rs':<15} {final_params['Rs']:<15.4f} {'K/W'}")
    print("=" * 60)

    print(f"\nCorresponding b coefficients:")
    print(f"b1 = {final_b[0]:.6f}")
    print(f"b2 = {final_b[1]:.6f}")
    print(f"b3 = {final_b[2]:.6f}")
    print(f"b4 = {final_b[3]:.6f}")

    # Plot results
    plot_results_paper_style(rls, t[1:], SOC[1:], Uocv[1:])

    return rls, final_params


def plot_results_paper_style(rls, t, SOC, Uocv):
    """
    Plot results matching the paper style
    """
    theta_history = np.array(rls.theta_history)

    # Create figure matching paper Fig. 10
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1: Parameter identification results (matching paper)
    ax1 = axes[0, 0]

    # Convert time to match paper scale (assuming seconds, convert to minutes or match scale)
    t_plot = t / 60 if t.max() > 1000 else t  # Convert to minutes if needed

    # Plot parameters with proper scaling to match paper
    ax1.plot(t_plot, theta_history[:, 0], 'b-', linewidth=2, label='Ccore (J/K⁻¹)')
    ax1.plot(t_plot, theta_history[:, 2] * 10, 'r-', linewidth=2, label='Rsurf (10 J K⁻¹)')
    ax1.plot(t_plot, theta_history[:, 1] * 10, 'g-', linewidth=2, label='Rcore (10 J K⁻¹)')

    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('TSM Parameters')
    ax1.set_title('Fig. 10. Battery thermal parameters identification results.')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xlim([0, t_plot.max()])
    ax1.set_ylim([0, 200])
    ax1.legend(loc='upper right')

    # Plot 2: SOC evolution
    ax2 = axes[0, 1]
    ax2.plot(t_plot, SOC, 'purple', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('SOC')
    ax2.set_title('State of Charge Evolution')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_ylim([0, 1])

    # Plot 3: OCV evolution
    ax3 = axes[1, 0]
    ax3.plot(t_plot, Uocv, 'orange', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('OCV (V)')
    ax3.set_title('Open Circuit Voltage Evolution')
    ax3.grid(True, linestyle=':', alpha=0.7)

    # Plot 4: Prediction error
    ax4 = axes[1, 1]
    ax4.plot(t_plot, rls.error_history, 'k-', linewidth=1)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Error (°C)')
    ax4.set_title('Temperature Prediction Error')
    ax4.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # File paths
    filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_env.xlsx"
    soc_ocv_filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_sococv.xlsx"

    try:
        rls_model, params = run_rls_identification(filepath, soc_ocv_filepath, Cs_value=3.4)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()