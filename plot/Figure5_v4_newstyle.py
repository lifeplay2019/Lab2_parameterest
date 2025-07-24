import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

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
                # 去除SOC=0的数据（包括浮点误差）
                data = data[np.abs(data['SOC']) > 1e-6]
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

# mesh为200，温度T从高到低
T_min, T_max = T_all.min(), T_all.max()
SOC_min, SOC_max = SOC_all.min(), SOC_all.max()
OCV_min, OCV_max = OCV_all.min(), OCV_all.max()

T_grid = np.linspace(T_max, T_min, 200)  # 这里高到低
SOC_grid = np.linspace(SOC_min, SOC_max, 200)
T_mesh, SOC_mesh = np.meshgrid(T_grid, SOC_grid)

# 采用linear插值，避免过拟合
OCV_grid = griddata(
    (T_all, SOC_all), OCV_all,
    (T_mesh, SOC_mesh),
    method='linear'
)

fig = plt.figure(figsize=(10,6), dpi=170)
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    T_mesh, SOC_mesh, OCV_grid,
    cmap=cm.jet,           # 鲜艳色彩
    edgecolor='k',
    linewidth=0.2,
    rstride=2, cstride=2,
    antialiased=True,
    alpha=1.0
)

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(OCV_grid)
cbar = fig.colorbar(m, ax=ax, shrink=0.68, aspect=18, pad=0.06)
cbar.set_label('Voltage/V', fontsize=13)

ax.scatter(
    T_all, SOC_all, OCV_all,
    color='k', marker='o', s=13,
    alpha=0.95, zorder=10
)

ax.set_xlabel("T (°C)", fontsize=15, labelpad=10)
ax.set_ylabel("SOC", fontsize=15, labelpad=10)
ax.set_zlabel("OCV(V)", fontsize=15, labelpad=10)
ax.set_xlim(T_max, T_min)          # 温度从高(右)到低(左)
ax.set_ylim(SOC_min, SOC_max)
ax.set_zlim(OCV_min, OCV_max)
ax.view_init(elev=24, azim=-117)

plt.tight_layout()
plt.show()