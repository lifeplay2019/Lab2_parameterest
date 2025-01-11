import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scienceplots  # Make sure the scienceplots style is installed.
plt.style.use(['science','nature'])
from matplotlib.backends.backend_pdf import PdfPages

# Correct directory path
directory_path = '../data/Lab2_data/'

# List files in the directory
files = os.listdir(directory_path)
batches = ['N10']
# ...

fig, ax = plt.subplots(figsize=(5, 3), dpi=200)  # 增加了图表尺寸

# Colors and markers
colors = ['#80A6E2']
markers = ['o']
legends = ['18650F']
line_width = 0.7

# Loop to plot
for i in range(len(batches)):
    for f in files:
        if batches[i] in f:
            try:
                path = os.path.join(directory_path, f)  # Path join correction here
                data = pd.read_csv(path)
                SOC = data['SOC'].values
                OCV = data['OCV'].values
                ax.plot(SOC[0:],
                        OCV[0:],
                        color=colors[i], alpha=1, linewidth=line_width,
                        marker=markers[i], markersize=2, markevery=50)
                # ...
            except Exception as e:
                print(f"Error processing {f}: {e}")

# Customize legend
custom_lines = [
    Line2D([0], [0], color=colors[0], marker=markers[0], markersize=1.5, linewidth=line_width)
]

# ax.set_title('Efset 18650 Voltage vs Time')
ax.set_title('Fitorch 21700 Voltage vs Time')
# ax.set_title('Fitorch 18650 Voltage vs Time')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.legend(custom_lines, legends, loc='upper right', bbox_to_anchor=(1.0, 1), frameon=False, ncol=3, fontsize=6)

# Adjust the voltage range to the appropriate min-max range
# ax.set_ylim([voltage_min - 0.1, voltage_max + 0.4])
plt.tight_layout()
plt.show()
