import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

# 温度批次字典
batch_temp = {
    'N30': -30, 'N25': -25, 'N20': -20, 'N15': -15, 'N10': -10, 'P25': 25
}
batches = ['N30', 'N25', 'N20', 'N15', 'N10', 'P25']
directory_path = '../data/Lab2_data/SOCOCV/'
files = os.listdir(directory_path)

SOC_all, T_all, OCV_all = [], [], []

for batch in batches:
    for f in files:
        if batch in f and f.endswith('.csv'):
            path = os.path.join(directory_path, f)
            try:
                data = pd.read_csv(path)
                # 去除SOC=0数据 (一定容忍浮点误差)
                data = data[np.abs(data['SOC'] - 0.0) > 1e-6]
                soc = data['SOC'].values
                ocv = data['OCV'].values
                temp = batch_temp[batch]
                SOC_all.extend(soc)
                OCV_all.extend(ocv)
                T_all.extend([temp] * len(soc))
            except Exception as e:
                print(f"Error processing {f}: {e}")

SOC_all = np.array(SOC_all)
T_all = np.array(T_all)
OCV_all = np.array(OCV_all)

# 极高分辨率
T_min, T_max = T_all.min(), T_all.max()
SOC_min, SOC_max = SOC_all.min(), SOC_all.max()
OCV_min, OCV_max = OCV_all.min(), OCV_all.max()

T_grid = np.linspace(T_min, T_max, 200)
SOC_grid = np.linspace(SOC_min, SOC_max, 200)
T_mesh, SOC_mesh = np.meshgrid(T_grid, SOC_grid)

OCV_grid = griddata(
    (T_all, SOC_all), OCV_all,
    (T_mesh, SOC_mesh),
    method='cubic'
)

fig = plt.figure(figsize=(8,4), dpi=150, facecolor='none')
ax = fig.add_subplot(111, projection='3d', facecolor='none')

# 画插值表面
surf = ax.plot_surface(
    T_mesh, SOC_mesh, OCV_grid,
    cmap=cm.jet,      # 也可试cm.turbo
    edgecolor='k',
    linewidth=0.2,
    rstride=2, cstride=2,
    antialiased=True,
    alpha=1.0
)

# 缩小色条
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(OCV_grid)
cbar = fig.colorbar(m, ax=ax, shrink=0.50, aspect=12, pad=0.06)
cbar.set_label('Voltage/V', fontsize=6)
cbar.ax.tick_params(labelsize=6)    # <--- 就是这里，使colorbar数字变小

# 坐标轴设置
ax.set_xlabel("T (°C)", fontsize=8, labelpad=10)
ax.set_ylabel("SOC", fontsize=8, labelpad=10)
ax.set_zlabel("OCV(V)", fontsize=8, labelpad=10)
ax.set_xlim(T_min, T_max)
ax.set_ylim(SOC_min, SOC_max)
ax.set_zlim(OCV_min, OCV_max)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


# 刻度字体变小
ax.tick_params(axis='x', labelsize=6)
ax.tick_params(axis='y', labelsize=6)
ax.tick_params(axis='z', labelsize=6)

# 视角
ax.view_init(elev=8, azim=-130)

plt.tight_layout()
plt.show()