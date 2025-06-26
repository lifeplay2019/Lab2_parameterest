import pandas as pd
import os

from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import scienceplots  # 确保导入了 scienceplots

plt.style.use(['science', 'nature'])

plt.style.use(['science','nature'])
from matplotlib.backends.backend_pdf import PdfPages

root = '../data/18650_procress/'
files = os.listdir(root)
fig, ax = plt.subplots(figsize=(4, 2), dpi=400)
colors = ['#80A6E2', '#7BDFF2', '#FBDD85', '#F46F43', '#403990', '#CF3D3E']
markers = ['o', 'v', 'D', 'p', 's', '^']
legends = ['25 °C', '-10 °C', '-15 °C', '-20 °C', '-25 °C', '-30 °C']
batches = ['P25', 'N10', 'N15', 'N20', 'N25', 'N30']
line_width = 1.0

capacity_min, capacity_max = float('inf'), float('-inf')

# Fixed loop for better error handling and usage of min/max
for i in range(6):
    for f in files:
        if batches[i] in f:
            path = os.path.join(root, f)
            try:
                data = pd.read_csv(path)
                time = data['time'].values
                capacity = data['capacity'].values
                ax.plot(time[1:], capacity[1:], color=colors[i], alpha=1, linewidth=line_width,
                        marker=markers[i], markersize=2, markevery=50)

                # Properly using numpy's min and max functions
                capacity_min = min(capacity_min, min(capacity[1:]))
                capacity_max = max(capacity_max, max(capacity[1:]))
            except Exception as e:
                print(f"Error processing {f}: {e}")

ax.set_xlabel('Time')
ax.set_ylabel('Capacity (mAh)')

custom_lines = [Line2D([0], [0], color=c, marker=m, markersize=2.5, linewidth=1.0) for c, m in zip(colors, markers)]
ax.legend(custom_lines, legends, loc='upper right', bbox_to_anchor=(1.0, 1), frameon=False, ncol=3, fontsize=6)

# Correctly setting the y-axis limits
ax.set_ylim([capacity_min - 0.1, capacity_max + 0.1])

plt.tight_layout()
plt.show()

# Save the figure appropriately
# with PdfPages("output.pdf") as pdf:
#     pdf.savefig(fig)
# plt.savefig('xxxxx.svg', format='svg')