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

# 超高分辨率插值网格
T_grid = np.linspace(-30, 25, 500)     # 这里用1000，mesh极细腻
SOC_grid = np.linspace(0, 1, 500)
T_mesh, SOC_mesh = np.meshgrid(T_grid, SOC_grid)

# linear插值
OCV_grid = griddata(
    (T_all, SOC_all), OCV_all,
    (T_mesh, SOC_mesh),
    method='linear'
)

fig = plt.figure(figsize=(8,5), dpi=150)
ax = fig.add_subplot(111, projection='3d')

# 为平滑观感，可以适当降低alpha让mesh视觉更细腻
surf = ax.plot_surface(
    T_mesh, SOC_mesh, OCV_grid,
    cmap=cm.jet,
    linewidth=0,
    antialiased=True,
    alpha=0.97,        # 可根据需要0.95-1.0调整
    rstride=1, cstride=1 # mesh每行每列都画
)

cbar = fig.colorbar(surf, shrink=0.55, aspect=13, pad=0.12)
cbar.set_label('Voltage/V', fontsize=12)

ax.set_xlabel('Temperature/°C')
ax.set_ylabel('SOC')
ax.set_zlabel('OCV (V)')
ax.set_title('3D Surface of OCV-T-SOC (High Resolution)')
ax.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.show()