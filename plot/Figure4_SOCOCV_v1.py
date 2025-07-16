import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

# 设置风格
plt.style.use(['science', 'nature'])

# 设置目录和批次
directory_path = '../data/Lab2_data/SOCOCV/'  # 按需修改路径
batches = ['N10']                             # 可以添加多个，例如 ['N10','P25']
files = os.listdir(directory_path)

# 颜色和图例，可扩展
colors = ['#1c9adb', '#D2691E', '#A2C948', '#D43F3A']
fit_colors = ['#D2691E', '#669966', '#FF8912', '#7C33A2']
legends = ['N10']

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

for i, batch in enumerate(batches):
    for f in files:
        if batch in f and f.endswith('.csv'):
            path = os.path.join(directory_path, f)
            try:
                data = pd.read_csv(path)
                x = data['SOC'].values
                y = data['OCV'].values

                # 原始点
                ax.scatter(x, y, s=38, facecolors='none', edgecolors=colors[i % len(colors)],
                           label=f'Measured Data {batch}', linewidth=1)

                # 8阶多项式拟合，保证x尺寸>8
                if len(x) > 8:
                    p = np.polyfit(x, y, 8)
                    x_fit = np.linspace(x.min(), x.max(), 500)
                    y_fit = np.polyval(p, x_fit)
                    ax.plot(x_fit, y_fit, color=fit_colors[i % len(fit_colors)],
                            linewidth=2.2, label=f'Poly Fit {batch}')
            except Exception as e:
                print(f"Error processing {f}: {e}")

# 坐标轴/标题
ax.set_xlabel('SOC')
ax.set_ylabel('OCV (V)')
ax.set_title('OCV-SOC Relationship by Batch')

ax.set_xlim([0, 1])
ax.set_ylim([2.4, 4.2])
ax.legend(fontsize=9, frameon=False, loc='lower right')
plt.tight_layout()
plt.show()