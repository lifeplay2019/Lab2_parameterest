import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # 保留你科研风格

# 设置风格
plt.style.use(['science', 'nature'])

# 读取P25的数据
data = pd.read_csv(r'C:\Users\shane\Desktop\Lab_2\Modle\Project_Lab2\Lab_2\data\Lab2_data\SOCOCV\P25_OCV_SOC_+25.csv')
x = data['SOC'].values
y = data['OCV'].values

# 多项式拟合 (8阶)
p = np.polyfit(x, y, 8)
x_fit = np.linspace(x.min(), x.max(), 500)
y_fit = np.polyval(p, x_fit)

fig, ax = plt.subplots(figsize=(6, 4), dpi=200)

# 散点（蓝色空心圆圈）
ax.scatter(x, y, s=38, facecolors='none', edgecolors='#1c9adb', label='Measured Data', linewidth=1)

# 拟合曲线（深橘色实线，更平滑）
ax.plot(x_fit, y_fit, color='#D2691E', linewidth=2.2, label='Polynomial Fit')

# 坐标轴/标题美化
ax.set_xlabel('SOC')
ax.set_ylabel('OCV (V)')
ax.set_title('OCV-SOC Relationship at 25°C')

# 坐标轴范围，可按需微调
ax.set_xlim([0, 1])
ax.set_ylim([2.4, 4.2])

# 图例（右下）
ax.legend(fontsize=9, frameon=False, loc='lower right')
plt.tight_layout()
plt.show()