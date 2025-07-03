import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import numpy as np
import scienceplots

# 设置论文风格
plt.style.use(['science', 'nature'])

root = '../data/Lab2_data/Temperature/TempCSV/'
files = os.listdir(root)
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
line_width = 0.5
sigma = 50  # Gaussian filter的标准差

# 需要绘制的所有批次
batches = ['P25', 'N10', 'N15', 'N20', 'N25', 'N30']

# 颜色和线型准备
colors = ['#80A6E2', '#FFA07A', '#7FC97F', '#FB8072', '#FFD700', '#A65628']
filtered_colors = ['#003399', '#D2691E', '#005C00', '#A80000', '#827717', '#654321']
markers = ['o', '^', 's', 'v', 'd', 'p']

temp_min, temp_max = float('inf'), float('-inf')

# legend映射，不加(Filter)
batch_labels = {
    'P25': '25°C',
    'N10': '-10°C',
    'N15': '-15°C',
    'N20': '-20°C',
    'N25': '-25°C',
    'N30': '-30°C',
}

for idx, batch in enumerate(batches):
    for f in files:
        if batch in f:
            try:
                path = os.path.join(root, f)
                data = pd.read_csv(path)
                if data.empty or 't' not in data.columns or 'temperature' not in data.columns:
                    print(f"Warning: {f} 缺少必要数据.")
                    continue
                data.dropna(subset=['t', 'temperature'], inplace=True)
                time = data['t'].values / 1e4  # 单位调整
                temperature = data['temperature'].values
                temperature_filtered = gaussian_filter1d(temperature, sigma=sigma)
                # 原始曲线，去除label不显示legend
                ax.plot(time, temperature,
                        color=colors[idx], alpha=0.3, linewidth=line_width,
                        marker=markers[idx], markersize=2, markevery=50, linestyle='--')
                # 平滑后的曲线，带label
                label = batch_labels.get(batch, batch)
                ax.plot(
                    time, temperature_filtered,
                    color=filtered_colors[idx], alpha=1,
                    linewidth=line_width, label=label
                )

                temp_min = min(temp_min, np.min(temperature_filtered))
                temp_max = max(temp_max, np.max(temperature_filtered))
            except Exception as e:
                print(f"Error processing {f}: {e}")

if np.isfinite(temp_min) and np.isfinite(temp_max):
    ax.set_ylim([temp_min - 1, temp_max + 1])

ax.set_title('Temperature vs. Time for All Batches')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Temperature (°C)')

# x轴科学计数法说明
ax.text(0.98, 0, r'$\times10^{4}$', fontsize=8, ha='right', va='top', transform=ax.transAxes)

# 显示唯一的label
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
ax.legend(unique.values(), unique.keys(), fontsize=6, frameon=False)
plt.legend(bbox_to_anchor=(0.95, 0.85))
plt.tight_layout()
plt.show()