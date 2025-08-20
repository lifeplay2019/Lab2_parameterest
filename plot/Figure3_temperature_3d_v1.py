import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scienceplots

# 设置论文风格，但禁用LaTeX渲染来避免Unicode问题
plt.style.use(['science', 'nature'])
plt.rcParams['text.usetex'] = False  # 禁用LaTeX渲染
plt.rcParams['font.family'] = 'DejaVu Sans'  # 使用支持Unicode的字体

root = '../data/Lab2_data/Temperature/TempCSV/'
files = os.listdir(root)

# 创建3D图形
fig = plt.figure(figsize=(6, 4), dpi=200)
ax = fig.add_subplot(111, projection='3d')

line_width = 1.5
sigma = 20  # Gaussian filter的标准差

# 需要绘制的所有批次
batches = ['P25', 'N10', 'N15', 'N20', 'N25', 'N30']

# 颜色准备
colors = ['#80A6E2', '#FFA07A', '#7FC97F', '#FB8072', '#FFD700', '#A65628']
filtered_colors = ['#003399', '#D2691E', '#005C00', '#A80000', '#827717', '#654321']

temp_min, temp_max = float('inf'), float('-inf')

# legend映射
batch_labels = {
    'P25': '20°C',
    'N10': '-10°C',
    'N15': '-15°C',
    'N20': '-20°C',
    'N25': '-25°C',
    'N30': '-30°C',
}

# Y轴位置（用于3D效果中的分离）
y_positions = [0, 1, 2, 3, 4, 5]

for idx, batch in enumerate(batches):
    batch_data = []
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

                # 计算相对于平均值的波动
                temp_mean = np.mean(temperature_filtered)
                temp_fluctuation = temperature_filtered - temp_mean

                # 将波动数据存储
                batch_data.append((time, temp_fluctuation, temperature_filtered))

                temp_min = min(temp_min, np.min(temperature_filtered))
                temp_max = max(temp_max, np.max(temperature_filtered))
            except Exception as e:
                print(f"Error processing {f}: {e}")

    # 绘制该批次的数据
    for time, temp_fluctuation, temp_original in batch_data:
        y_pos = np.full_like(time, y_positions[idx])

        # 绘制原始温度曲线（半透明）
        ax.plot(time, y_pos, temp_original,
                color=colors[idx], alpha=0.3, linewidth=line_width * 0.5,
                linestyle='--')

        # 绘制相对波动曲线（实线，更明显）
        ax.plot(time, y_pos, temp_fluctuation + y_positions[idx] * 5,
                color=filtered_colors[idx], alpha=0.8, linewidth=line_width,
                label=batch_labels.get(batch, batch) if time is batch_data[0][0] else "")

# 设置轴标签（避免使用上标符号）
ax.set_xlabel('Time (x10^4 s)')  # 使用^4代替⁴
ax.set_ylabel('Temperature Conditions')
ax.set_zlabel('Temperature (°C)')

# 设置Y轴刻度标签
ax.set_yticks(y_positions)
ax.set_yticklabels([batch_labels[batch] for batch in batches])

# 设置标题
ax.set_title('3D Temperature Variations for Different Conditions')

# 设置视角
ax.view_init(elev=20, azim=45)

# 显示图例
handles, labels = ax.get_legend_handles_labels()
if handles:  # 确保有图例项
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right', fontsize=10)

# 移除tight_layout()调用，直接调整子图参数
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# 保存为JPG格式
plt.savefig('temperature_3d_plot.jpg', format='jpeg', dpi=300, bbox_inches='tight')
plt.show()