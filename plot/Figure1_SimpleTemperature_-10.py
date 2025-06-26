import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import numpy as np

import scienceplots
plt.style.use(['science', 'nature'])

# Define the root path to your data
root = '../data/Lab2_data/Temperature/TempCSV/'

files = os.listdir(root)
fig, ax = plt.subplots(figsize=(4, 3), dpi=200)
color_original = '#80A6E2'
color_filtered = '#FFA07A'
markers = ['o']
batch = 'N10'
line_width = 0.5

ICA_min, ICA_max = float('inf'), float('-inf')
sigma = 50  # Standard deviation for Gaussian filter

for f in files:
    if batch in f:
        try:
            path = os.path.join(root, f)
            data = pd.read_csv(path)

            # Ensure data is not empty and is properly formatted
            if data.empty or 'time' not in data.columns or 'temperature' not in data.columns:
                print(f"Warning: {f} is missing required data.")
                continue

            data.dropna(subset=['time', 'temperature'], inplace=True)

            time = data['t'].values / 1e4  # Convert time to time * 10^4
            temperature = data['temperature'].values

            temperature_filtered = gaussian_filter1d(temperature, sigma=sigma)

            ax.plot(time, temperature, color=color_original, alpha=0.5, linewidth=line_width,
                    marker=markers[0], markersize=2, markevery=50, linestyle='--')
            ax.plot(time, temperature_filtered, color=color_filtered, alpha=1, linewidth=line_width * 1)

            ICA_min = min(ICA_min, np.min(temperature_filtered))
            ICA_max = max(ICA_max, np.max(temperature_filtered))
        except Exception as e:
            print(f"Error processing {f}: {e}")

if np.isfinite(ICA_min) and np.isfinite(ICA_max):
    ax.set_ylim([ICA_min - 1, ICA_max + 1])

ax.set_title('Efest Temperature vs. Time under -10°C Environment')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')

# Setting up the scale notation for the x-axis
ax.text(0.99, 0, r'$\times10^{4}$', fontsize=4, ha='right', va='top', transform=ax.transAxes)

legend_elements = [
    Line2D([0], [0], color=color_original, lw=1, linestyle='--', label='Original Data'),
    Line2D([0], [0], color=color_filtered, lw=1, label='Gaussian Filter')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=6)

plt.tight_layout()
plt.show()