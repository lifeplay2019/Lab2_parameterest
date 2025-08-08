import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.mplot3d import Axes3D

root = '../data/Lab2_data/Temperature/TempCSV/'
files = os.listdir(root)

batches = ['P25', 'N10', 'N15', 'N20', 'N25', 'N30']
batch_temps = {  # 每组的实际温度，单位：摄氏度
    'P25': 20,
    'N10': -10,
    'N15': -15,
    'N20': -20,
    'N25': -25,
    'N30': -30
}

fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection='3d')

line_width = 0.5
sigma = 50

colors = ['#003399', '#D2691E', '#005C00', '#A80000', '#827717', '#654321']

for idx, batch in enumerate(batches):
    for f in files:
        if batch in f:
            path = os.path.join(root, f)
            data = pd.read_csv(path)
            if data.empty or 't' not in data.columns or 'temperature' not in data.columns or 'value' not in data.columns:
                print(f"Warning: {f} 缺少必要数据.")
                continue
            data.dropna(subset=['t', 'temperature', 'value'], inplace=True)
            time = data['t'].values / 1e4  # 时间处理
            value = data['value'].values
            value_filtered = gaussian_filter1d(value, sigma=sigma)
            temp_value = batch_temps[batch]  # 不同温度对应不同y
            y = np.ones_like(time) * temp_value
            ax.plot(time, y, value_filtered, color=colors[idx], lw=line_width, label=f"{temp_value}°C")

ax.set_xlabel('Time ($\\times10^4$ s)')
ax.set_ylabel('Temperature (°C)')
ax.set_zlabel('Value')  # 替换为实际参数名，比如"Capacity (mAh)"等
ax.set_title('Battery Parameter vs Time at Different Temperatures')
ax.legend()
plt.tight_layout()
plt.show()