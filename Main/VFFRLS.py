import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Correct path to the CSV file
csv_file_path = '../data/Lab2_data/original_data/hppc_18650_+25.csv'

# Check if the CSV file exists before reading
if not os.path.exists(csv_file_path):
    raise FileNotFoundError(f"No such file: '{csv_file_path}'")

# Load the experimental data from the CSV file
csv_data = pd.read_csv(csv_file_path)
print(csv_data.columns)

V = csv_data['V'].astype(np.float64).values
I = csv_data['I'].astype(np.float64).values


L_data = len(I)  # Length of experimental data
T = 1  # Sampling time step

# Initialization of the model parameters
Uoc = np.zeros(L_data, dtype=np.float64)  # Open-circuit voltage
Ro = np.zeros(L_data, dtype=np.float64)   # Ohmic internal resistance
Rp = np.zeros(L_data, dtype=np.float64)   # Polarization internal resistance
Cp = np.zeros(L_data, dtype=np.float64)   # Polarization capacitance

# Initialization of the identification algorithm
u = 0.1          # Forgetting factor
Phi = np.zeros(4, dtype=np.float64)       # Data vector
theta = np.zeros(4, dtype=np.float64)     # Parameter vector
P = 1e6 * np.eye(4, dtype=np.float64)     # Covariance matrix
K = np.zeros(4, dtype=np.float64)         # Gain matrix

# Online model parameters identification
for k in range(1, L_data):
    Phi = np.array([1, V[k - 1], I[k], I[k - 1]], dtype=np.float64)
    temp = Phi.T @ P @ Phi + u
    K = (P @ Phi) / temp
    theta = theta + K * (V[k] - Phi @ theta)
    P = (np.eye(4, dtype=np.float64) - np.outer(K, Phi)) @ P / u

    # Parameter resolution
    Uoc[k] = theta[0] / (1 - theta[1])
    Ro[k] = (theta[2] - theta[3]) / (1 + theta[1])
    Rp[k] = (theta[2] + theta[3]) / (1 - theta[1]) - Ro[k]
    Cp[k] = (1 + theta[1]) / (2 - 2 * theta[1]) / Rp[k]

# Assuming initial parameters are the same as the ones at the next moment
Uoc[0] = Uoc[1]
Ro[0] = Ro[1]
Rp[0] = Rp[1]
Cp[0] = Cp[1]

# Model accuracy verification
count = 1
if L_data > 1:  # Ensure there's enough data to compare
    for step in range(L_data - 1):
        if I[step + 1] == I[step]:
            count += 1
        else:
            break

# Initialize voltage storage
Vp = np.zeros(L_data, dtype=np.float64)         # Polarization voltage
V_model = np.zeros(L_data, dtype=np.float64)    # Model terminal voltage
V_model[:count] = V[:count]
V_err = np.zeros(L_data, dtype=np.float64)      # Model terminal voltage error

for k1 in range(count, L_data):
    exp_val = np.exp(-T / (Rp[k1] * Cp[k1]))
    Vp[k1] = Vp[k1 - 1] * exp_val + I[k1] * Rp[k1] * (1 - exp_val)
    V_model[k1] = Uoc[k1] + Vp[k1] + I[k1] * Ro[k1]
    V_err[k1] = (V_model[k1] - V[k1]) * 1000

# # Plotting
# plt.figure()
# plt.plot(V_model, label='Model Terminal Voltage', linewidth=2)
# plt.plot(V, label='Actual Terminal Voltage', linewidth=2)
# plt.legend()
# plt.xlabel('Time/s')
# plt.ylabel('Voltage/V')
#
# plt.figure()
# plt.plot(V_err, label='Voltage Error', linewidth=2)
# plt.legend()
# plt.xlabel('Time/s')
# plt.ylabel('Voltage Error/mV')
# plt.show()

plt.figure()
plt.plot(Rp, label='R0', linewidth=2)
plt.legend()
plt.xlabel('Time/s')
plt.ylabel('R0')
plt.show()