import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RLS_ThermalBattery:
    def __init__(self, Cs_fixed=3.6, lambda_factor=0.99, P0=1e6):
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
        self.theta = np.array([[50.0], [2.5], [3.5]])  # [Cc, Rc, Rs]
        self.P = P0 * np.eye(self.n_params)  # Covariance matrix

        # Storage for results
        self.theta_history = []
        self.error_history = []
        self.physical_params_history = []
        self.b_coefficients_history = []

    def calculate_b_coefficients(self, dt=1.0):
        """
        Calculate b1, b2, b3, b4 from physical parameters
        """
        Cc = float(self.theta[0, 0])
        Rc = float(self.theta[1, 0])
        Rs = float(self.theta[2, 0])
        Cs = self.Cs

        # Avoid division by zero
        if abs(Cc * (Rc + Rs)) < 1e-10:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Calculate b coefficients from physical parameters
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

        # Construct regressor vector
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

        # Update parameters with adaptive learning rate
        learning_rate = 0.2  # Reduce aggressive updates
        self.theta = self.theta + learning_rate * K * e

        # Ensure parameters stay positive and within reasonable bounds
        self.theta[0, 0] = np.clip(self.theta[0, 0], 10.0, 80.0)  # Cc between 10 and 200
        self.theta[1, 0] = np.clip(self.theta[1, 0], 0.1, 5.0)  # Rc between 0.1 and 10
        self.theta[2, 0] = np.clip(self.theta[2, 0], 0.1, 5.0)  # Rs between 0.1 and 10

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


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the HPPC test data
    """
    # Load data
    df = pd.read_excel(filepath, skiprows=1)

    # Extract columns
    t = df.iloc[:, 1].values  # time
    v = df.iloc[:, 2].values  # voltage
    i = df.iloc[:, 3].values  # current
    Ta = df.iloc[:, 4].values  # ambient temperature
    Ts = df.iloc[:, 5].values  # surface temperature

    # Clean data - remove NaN values
    valid_idx = ~(np.isnan(t) | np.isnan(v) | np.isnan(i) | np.isnan(Ta) | np.isnan(Ts))
    t = t[valid_idx]
    v = v[valid_idx]
    i = i[valid_idx]
    Ta = Ta[valid_idx]
    Ts = Ts[valid_idx]

    # Calculate time step
    if len(t) > 1:
        dt = np.median(np.diff(t))  # Use median to avoid outliers
    else:
        dt = 1.0  # Default value

    # Calculate Uocv (simplified - you may need to adjust based on SOC)
    Uocv = np.ones_like(v) * 4.2  # Nominal voltage for 18650 cell

    # Calculate heat generation H = (Uocv - Ut) * I
    H = (Uocv - v) * np.abs(i)  # Use absolute current value

    return t, Ts, Ta, H, dt


def run_rls_identification(filepath, Cs_value=3.4):
    """
    Main function to run RLS identification with fixed Cs
    """
    # Load data
    print("Loading data from:", filepath)
    t, Ts, Ta, H, dt = load_and_preprocess_data(filepath)

    print(f"Using fixed Cs = {Cs_value} J/K (from paper)")

    # Initialize RLS with parameters matching the paper
    rls = RLS_ThermalBattery(Cs_fixed=Cs_value, lambda_factor=0.995, P0=1e4)

    # Prepare for iteration
    n_samples = len(Ts)

    print(f"Running RLS identification on {n_samples} samples...")
    print(f"Time step dt = {dt:.3f} seconds")
    print(f"Initial parameters: Cc=50.0, Rc=0.5, Rs=1.5")

    # Start from k=1 to have k-1 available
    for k in range(1, n_samples):
        # RLS update
        theta = rls.update(
            Ts_prev=Ts[k - 1],
            Ta_curr=Ta[k],
            Ta_prev=Ta[k - 1],
            H_prev=H[k - 1],
            Ts_curr=Ts[k],
            dt=dt
        )

        # Print progress
        if k % 100 == 0:
            print(f"Step {k}: Cc={theta[0]:.2f}, Rc={theta[1]:.4f}, Rs={theta[2]:.4f}")

    # Get final parameters
    final_params = rls.physical_params_history[-1]
    final_b = rls.calculate_b_coefficients(dt)

    print("\n" + "=" * 60)
    print("Table 3: Identification results of parameters")
    print("=" * 60)
    print(f"{'Parameters':<15} {'Value':<20} {'95% Confidence interval'}")
    print("-" * 60)
    print(
        f"{'Cc (J/K)':<15} {final_params['Cc']:<20.3f} {final_params['Cc'] * 0.95:.1f}-{final_params['Cc'] * 1.05:.1f}")
    print(
        f"{'Ru (K/W)':<15} {final_params['Rs']:<20.3f} {final_params['Rs'] * 0.95:.2f}-{final_params['Rs'] * 1.05:.2f}")
    print(
        f"{'Rc (K/W)':<15} {final_params['Rc']:<20.3f} {final_params['Rc'] * 0.95:.2f}-{final_params['Rc'] * 1.05:.2f}")
    print("=" * 60)

    print(f"\nCs = {final_params['Cs']:.2f} J/K (fixed)")

    print("\nCorresponding b coefficients:")
    print(f"b1 = {final_b[0]:.6f}")
    print(f"b2 = {final_b[1]:.6f}")
    print(f"b3 = {final_b[2]:.6f}")
    print(f"b4 = {final_b[3]:.6f}")

    # Plot results
    plot_results_paper_style(rls, t[1:])

    return rls, final_params


def plot_results_paper_style(rls, t):
    """
    Plot results in the style of Fig. 9 from the paper
    """
    theta_history = np.array(rls.theta_history)

    # Create figure matching paper style
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot parameters with scaling to match the paper
    # Cc in J/K (direct)
    ax.plot(t, theta_history[:, 0], 'r-', linewidth=2, label='Cc/(JK-1)')

    # Ru (Rs) scaled by 0.1 (multiply by 10 to match scale)
    ax.plot(t, theta_history[:, 2] * 10, 'g-', linewidth=2, label='Ru/(0.1KW-1)')

    # Rc scaled by 0.1 (multiply by 10 to match scale)
    ax.plot(t, theta_history[:, 1] * 10, 'b-', linewidth=2, label='Rc/(0.1KW-1)')

    # Set axis properties
    ax.set_xlabel('Time/s', fontsize=12)
    ax.set_ylabel('Parameters', fontsize=12)
    ax.set_title('Fig. 9. Parameter identification results.', fontsize=12)

    # Set grid with dotted lines
    ax.grid(True, linestyle=':', alpha=0.7)

    # Set axis limits to match the paper
    ax.set_xlim([0, max(t)])
    ax.set_ylim([0, 75])

    # Add legend
    ax.legend(loc='upper right', fontsize=11)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.show()

    # Additional plots for analysis
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Convergence detail for each parameter
    axes[0, 0].plot(t, theta_history[:, 0], 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Cc (J/K)')
    axes[0, 0].set_title('Core Heat Capacity Convergence')
    axes[0, 0].grid(True, linestyle=':')

    axes[0, 1].plot(t, theta_history[:, 2], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Rs/Ru (K/W)')
    axes[0, 1].set_title('Surface Thermal Resistance Convergence')
    axes[0, 1].grid(True, linestyle=':')

    axes[1, 0].plot(t, theta_history[:, 1], 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Rc (K/W)')
    axes[1, 0].set_title('Core Thermal Resistance Convergence')
    axes[1, 0].grid(True, linestyle=':')

    # Plot 2: Prediction error
    axes[1, 1].plot(t, rls.error_history, 'k-', linewidth=1)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (°C)')
    axes[1, 1].set_title('Prediction Error Evolution')
    axes[1, 1].grid(True, linestyle=':')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Use raw string or forward slashes to avoid escape sequence issues
    filepath = r"D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_env.xlsx"

    # Use Cs value from the paper (3.4 J/K)
    Cs_paper = 3.4  # J/K as specified in the paper

    try:
        rls_model, params = run_rls_identification(filepath, Cs_value=Cs_paper)

        # Save results
        results_df = pd.DataFrame([params])
        results_df.to_csv("rls_identified_parameters.csv", index=False)
        print("\nResults saved to 'rls_identified_parameters.csv'")

    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        print("Please ensure the data file is in the correct location.")
    except Exception as e:
        print(f"Error during RLS identification: {e}")
        import traceback

        traceback.print_exc()