import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

batch_temp = {
    'N30': -30,
    'N25': -25,
    'N20': -20,
    'N15': -15,
    'N10': -10,
    'P25': 25
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
                # 只保留SOC不等于0.1的行（建议用足够小的误差容忍）
                data = data[np.abs(data['SOC'] - 0) > 1e-6]
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

T_grid = np.linspace(-30, 25, 400)
SOC_grid = np.linspace(0, 1, 400)
T_mesh, SOC_mesh = np.meshgrid(T_grid, SOC_grid)

OCV_grid = griddata(
    (T_all, SOC_all), OCV_all,
    (T_mesh, SOC_mesh),
    method='cubic'
)

fig = plt.figure(figsize=(8,6), dpi=150)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(T_mesh, SOC_mesh, OCV_grid,
                       cmap=cm.jet,
                       linewidth=0, antialiased=True,
                       alpha=0.97)
cbar = fig.colorbar(surf, shrink=0.55, aspect=13, pad=0.12)
cbar.set_label('Voltage/V', fontsize=12)

ax.set_xlabel('Temperature/°C')
ax.set_ylabel('SOC')
ax.set_zlabel('OCV (V)')
ax.set_title('3D Surface of OCV-T-SOC (exclude SOC=0.1)')
ax.view_init(elev=25, azim=-60)

plt.tight_layout()
plt.show()