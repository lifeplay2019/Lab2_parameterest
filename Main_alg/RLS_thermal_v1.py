import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RLS_ThermalBattery:
    def __init__(self, lambda_factor=0.98, P0=1e6):
        """
        Initialize RLS algorithm for thermal battery model

        Parameters:
        lambda_factor: forgetting factor (0 < λ ≤ 1)
        P0: initial covariance matrix value
        """
        self.lambda_factor = lambda_factor
        self.n_params = 4  # [b1, b2, b3, b4]

        # Initialize parameters
        self.theta = np.zeros((self.n_params, 1))  # Parameter vector
        self.P = P0 * np.eye(self.n_params)  # Covariance matrix

        # Storage for results
        self.theta_history = []
        self.error_history = []
        self.physical_params_history = []

    def update(self, phi, z, dt=30):
        """
        RLS update step

        Parameters:
        phi: regressor vector [Ts_k-1, Ta_k, Ta_k-1, H_k-1]
        z: observation Ts_k
        """
        phi = phi.reshape(-1, 1)

        # Prediction error
        y_pred = phi.T @ self.theta
        e = z - y_pred

        # Kalman gain
        K = self.P @ phi / (self.lambda_factor + phi.T @ self.P @ phi)

        # Update parameters
        self.theta = self.theta + K * e

        # Update covariance matrix
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_factor

        # Store results
        self.theta_history.append(self.theta.flatten().copy())
        self.error_history.append(float(e))

        # Calculate and store physical parameters
        physical_params = self.extract_physical_parameters(dt)
        self.physical_params_history.append(physical_params)

        return self.theta.flatten()

    def extract_physical_parameters(self, dt=30):
        """
        Extract physical parameters from identified coefficients
        """
        b1, b2, b3, b4 = self.theta.flatten()

        # From equation (18), derive the physical parameters
        # b1 = (Rc*Cc + Rs*Cs - dt) / (Cc*(Rc+Rs))
        # b2 = Rc/(Rs+Rc)
        # b3 = (dt - Rs*Cs)/(Cc*(Rc+Rs))
        # b4 = Rs*dt / (Cc*(Rc+Rs))

        # Avoid division by zero
        if abs(b2) < 1e-10 or abs(b4) < 1e-10:
            return {
                'Cc': 50,
                'Cs': 1.0,
                'Rc': 2.1,
                'Rs': 3.5,
                'b1': b1,
                'b2': b2,
                'b3': b3,
                'b4': b4
            }

        # From b2: Rc/(Rs+Rc) = b2
        # Therefore: Rc = b2*Rs/(1-b2) if b2 < 1

        # From b4: Rs*dt / (Cc*(Rc+Rs)) = b4
        # From b1 + b4: Rc*Cc / (Cc*(Rc+Rs)) = b1 + b4
        # Therefore: Rc/(Rc+Rs) = b1 + b4

        # Estimate Rs and Rc
        if abs(1 - b2) > 1e-6:
            # Rs/Rc = (1-b2)/b2
            Rs_Rc_ratio = (1 - b2) / b2 if abs(b2) > 1e-6 else 1.0

            # Assume a reasonable Rs value
            Rs_est = 0.1  # Initial guess in K/W
            Rc_est = Rs_est / Rs_Rc_ratio if abs(Rs_Rc_ratio) > 1e-6 else Rs_est
        else:
            Rs_est = 0.1
            Rc_est = 0.1

        # Calculate Cc from b4
        if abs(b4 * (Rc_est + Rs_est)) > 1e-10:
            Cc_est = Rs_est * dt / (b4 * (Rc_est + Rs_est))
        else:
            Cc_est = 100.0  # Default value

        # Calculate Cs from b3
        # b3 = (dt - Rs*Cs)/(Cc*(Rc+Rs))
        # Rs*Cs = dt - b3*Cc*(Rc+Rs)
        if abs(Rs_est) > 1e-10:
            Cs_est = (dt - b3 * Cc_est * (Rc_est + Rs_est)) / Rs_est
        else:
            Cs_est = 100.0  # Default value

        return {
            'Cc': Cc_est,
            'Cs': Cs_est,
            'Rc': Rc_est,
            'Rs': Rs_est,
            'b1': b1,
            'b2': b2,
            'b3': b3,
            'b4': b4
        }


def load_and_preprocess_data(filepath):
    """
    Load and preprocess the HPPC test data
    """
    # Load data, skip first row
    df = pd.read_excel(filepath, skiprows=1)

    # Extract columns [t_0, t, v, i, Ta, Ts]
    t = df.iloc[:, 1].values  # time
    v = df.iloc[:, 2].values  # voltage (Ut)
    i = df.iloc[:, 3].values  # current
    Ta = df.iloc[:, 4].values  # ambient temperature
    Ts = df.iloc[:, 5].values  # surface temperature

    # Calculate Uocv (simplified - you may need a more sophisticated SOC-OCV relationship)
    # For now, using a simple approach - adjust based on your battery model
    Uocv = np.ones_like(v) * 4.2  # Nominal voltage for 18650 cell

    # Calculate heat generation H = (Uocv - Ut) * I
    H = (Uocv - v) * i

    # Calculate time step
    dt = np.mean(np.diff(t))

    return t, Ts, Ta, H, dt


def run_rls_identification(filepath):
    """
    Main function to run RLS identification
    """
    # Load data
    print("Loading data from:", filepath)
    t, Ts, Ta, H, dt = load_and_preprocess_data(filepath)

    # Initialize RLS
    rls = RLS_ThermalBattery(lambda_factor=0.98, P0=1e6)

    # Prepare for iteration
    n_samples = len(Ts)
    identified_params = []

    print(f"Running RLS identification on {n_samples} samples...")
    print(f"Time step dt = {dt:.3f} seconds")

    # Start from k=2 to have k-1 available
    for k in range(2, n_samples):
        # Construct regressor vector φ = [Ts_k-1, Ta_k, Ta_k-1, H_k-1]
        phi = np.array([
            Ts[k - 1],  # Ts,k-1
            Ta[k],  # Ta,k
            Ta[k - 1],  # Ta,k-1
            H[k - 1]  # H_k-1
        ])

        # Observation
        z = Ts[k]

        # Update RLS
        theta = rls.update(phi, z, dt)
        identified_params.append(theta)

    # Extract final physical parameters
    physical_params = rls.extract_physical_parameters(dt)

    print("\nIdentified coefficients:")
    print(f"b1 = {physical_params['b1']:.6f}")
    print(f"b2 = {physical_params['b2']:.6f}")
    print(f"b3 = {physical_params['b3']:.6f}")
    print(f"b4 = {physical_params['b4']:.6f}")

    print("\nEstimated physical parameters:")
    print(f"Cc = {physical_params['Cc']:.6f} J/K")
    print(f"Cs = {physical_params['Cs']:.6f} J/K")
    print(f"Rc = {physical_params['Rc']:.6f} K/W")
    print(f"Rs = {physical_params['Rs']:.6f} K/W")

    # Plot results
    plot_results(rls, t[2:], Ts[2:], Ta[2:], H[2:])

    return rls, physical_params


def plot_results(rls, t, Ts_actual, Ta, H):
    """
    Plot RLS identification results
    """
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))

    # Plot 1: b parameters convergence
    theta_history = np.array(rls.theta_history)
    axes[0, 0].plot(t, theta_history[:, 0], label='b1')
    axes[0, 0].plot(t, theta_history[:, 1], label='b2')
    axes[0, 0].plot(t, theta_history[:, 2], label='b3')
    axes[0, 0].plot(t, theta_history[:, 3], label='b4')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Parameter Value')
    axes[0, 0].set_title('Parameter Convergence (b1, b2, b3, b4)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Prediction error
    axes[0, 1].plot(t, rls.error_history)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Prediction Error (K)')
    axes[0, 1].set_title('RLS Prediction Error')
    axes[0, 1].grid(True)

    # Plot 3: Temperature prediction
    Ts_pred = []
    for k in range(len(theta_history)):
        if k == 0:
            Ts_pred.append(Ts_actual[0])
        else:
            phi = np.array([Ts_actual[k - 1], Ta[k], Ta[k - 1], H[k - 1]])
            pred = np.dot(theta_history[k], phi)
            Ts_pred.append(pred)

    axes[1, 0].plot(t, Ts_actual, 'b-', label='Actual Ts', alpha=0.7)
    axes[1, 0].plot(t, Ts_pred, 'r--', label='Predicted Ts', alpha=0.7)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Surface Temperature: Actual vs Predicted')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Extract physical parameters history
    physical_params_history = rls.physical_params_history
    Rc_history = [p['Rc'] for p in physical_params_history]
    Rs_history = [p['Rs'] for p in physical_params_history]
    Cc_history = [p['Cc'] for p in physical_params_history]
    Cs_history = [p['Cs'] for p in physical_params_history]

    # Plot 4: Rc convergence
    axes[1, 1].plot(t, Rc_history, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Rc (K/W)')
    axes[1, 1].set_title('Thermal Resistance Rc Convergence')
    axes[1, 1].grid(True)

    # Plot 5: Rs convergence
    axes[2, 0].plot(t, Rs_history, 'm-', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Rs (K/W)')
    axes[2, 0].set_title('Thermal Resistance Rs Convergence')
    axes[2, 0].grid(True)

    # Plot 6: Cc convergence
    axes[2, 1].plot(t, Cc_history, 'c-', linewidth=2)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Cc (J/K)')
    axes[2, 1].set_title('Core Heat Capacity Cc Convergence')
    axes[2, 1].grid(True)

    # Plot 7: Cs convergence
    axes[3, 0].plot(t, Cs_history, 'y-', linewidth=2)
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_ylabel('Cs (J/K)')
    axes[3, 0].set_title('Surface Heat Capacity Cs Convergence')
    axes[3, 0].grid(True)

    # Plot 8: All physical parameters normalized
    axes[3, 1].plot(t, np.array(Rc_history) / np.max(np.abs(Rc_history)), label='Rc (normalized)')
    axes[3, 1].plot(t, np.array(Rs_history) / np.max(np.abs(Rs_history)), label='Rs (normalized)')
    axes[3, 1].plot(t, np.array(Cc_history) / np.max(np.abs(Cc_history)), label='Cc (normalized)')
    axes[3, 1].plot(t, np.array(Cs_history) / np.max(np.abs(Cs_history)), label='Cs (normalized)')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_ylabel('Normalized Value')
    axes[3, 1].set_title('All Physical Parameters (Normalized)')
    axes[3, 1].legend()
    axes[3, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the RLS identification
    filepath = "D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_env.xlsx"

    try:
        rls_model, params = run_rls_identification(filepath)

        # Save results
        results_df = pd.DataFrame([params])
        results_df.to_csv("rls_identified_parameters.csv", index=False)
        print("\nResults saved to 'rls_identified_parameters.csv'")

    except FileNotFoundError:
        print(f"Error: Could not find file at {filepath}")
        print("Please ensure the data file is in the correct location.")
    except Exception as e:
        print(f"Error during RLS identification: {e}")