# -*- coding: utf-8 -*-
# Voltage 与 Current 合并在同一子图；不含小视窗
# 扩大 Current 轴上方留白，使电流曲线主体更偏向图像下半部分

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

# ===== 基础设置（期刊风格 + 小字号） =====
mpl.rcParams.update({
    'font.family': 'DejaVu Sans',   # 如系统有 Arial 可改 'Arial'
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'savefig.dpi': 600,
    'figure.dpi': 200,
})

# 色盲友好配色
COL = {'V': '#1f77b4', 'I': '#2ca02c', 'Ta': '#d62728', 'Ts': '#ff7f0e'}

# ===== 读取数据 =====
excel_path = r'D:\Battery_Lab2\Battery_parameter\Lab2_parameterest\data\Lab2_data\RLS\hppc_18650_p25_env.xlsx'
df = pd.read_excel(excel_path)
df = df.rename(columns={'t': 't', 'v': 'v', 'i': 'i', 'Ta': 'Ta', 'Ts': 'Ts'})
df = df.dropna(subset=['t', 'v', 'i', 'Ta', 'Ts']).reset_index(drop=True)

t  = df['t'].to_numpy()
v  = df['v'].to_numpy()
i  = df['i'].to_numpy()
Ta = df['Ta'].to_numpy()
Ts = df['Ts'].to_numpy()

# ===== 可选：绘图抽稀（仅影响绘制） =====
def downsample_uniform(n, max_points=8000):
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)

idx = downsample_uniform(len(df), max_points=8000)
t_plot, v_plot, i_plot, Ta_plot, Ts_plot = t[idx], v[idx], i[idx], Ta[idx], Ts[idx]

# ===== 时间轴单位自动（秒/分/小时） =====
def time_axis(tsec):
    tmax = float(np.nanmax(tsec)) if len(tsec) else 0.0
    if tmax >= 2 * 3600:
        return tsec / 3600.0, 'Time (h)'
    elif tmax >= 120:
        return tsec / 60.0, 'Time (min)'
    else:
        return tsec, 'Time (s)'

tx, xlabel = time_axis(t_plot)

# ===== 图形：两行子图（上：V+I 双轴；下：Ta+Ts） =====
fig = plt.figure(figsize=(7.0, 4.8), constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.2, 1])

axVI = fig.add_subplot(gs[0, 0])   # Voltage 左轴
axVI_t = axVI.twinx()               # Current 右轴
axT = fig.add_subplot(gs[1, 0], sharex=axVI)

def beautify_left_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(which='major', color='0.85', linewidth=0.5)
    ax.grid(which='minor', color='0.92', linewidth=0.4)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())

beautify_left_ax(axVI)
beautify_left_ax(axT)

axVI_t.spines['top'].set_visible(False)
axVI_t.yaxis.set_major_locator(MaxNLocator(nbins=5))
axVI_t.yaxis.set_minor_locator(AutoMinorLocator())

# ===== 绘图 =====
lV, = axVI.plot(tx, v_plot, color=COL['V'], lw=1.15, label='Voltage')
lI, = axVI_t.plot(tx, i_plot, color=COL['I'], lw=1.0, label='Current')
axVI_t.axhline(0, color='0.6', lw=0.6, alpha=0.6)

lTa, = axT.plot(tx, Ta_plot, color=COL['Ta'], lw=1.0, label='Ta')
lTs, = axT.plot(tx, Ts_plot, color=COL['Ts'], lw=1.0, label='Ts')

# 轴标题与刻度（小字号）
axVI.set_ylabel('Voltage (V)', color=COL['V'], fontsize=9)
axVI_t.set_ylabel('Current (A)', color=COL['I'], fontsize=9)
axT.set_ylabel('Temperature (°C)', fontsize=9)
axT.set_xlabel(xlabel, fontsize=9)

axVI.tick_params(axis='y', colors=COL['V'], labelsize=7)
axVI_t.tick_params(axis='y', colors=COL['I'], labelsize=7)
axVI.tick_params(axis='x', labelsize=7)
axT.tick_params(axis='both', labelsize=7)

axVI.spines['left'].set_color(COL['V'])
axVI_t.spines['right'].set_color(COL['I'])

# ===== 留白设置：让 Current 主体位于图像下半部分 =====
def set_current_ylim_lower_half(ax, y, bottom_margin=0.05, top_space_factor=1.2):
    y = np.asarray(y)
    m = np.isfinite(y)
    if not np.any(m):
        ax.set_ylim(-1, 1)
        return
    cmin = float(np.min(y[m]))
    cmax = float(np.max(y[m]))
    rng = cmax - cmin
    if rng <= 0:
        pad = max(1.0, abs(cmax) * 0.2)
        ax.set_ylim(cmin - pad, cmax + 3 * pad)
        return
    ymin = cmin - bottom_margin * rng
    ymax = cmax + top_space_factor * rng   # 增大上方留白
    ax.set_ylim(ymin, ymax)

def nice_ylim(ax, margin=0.04):
    ymin, ymax = ax.get_ylim()
    rng = ymax - ymin
    if rng > 0:
        ax.set_ylim(ymin - rng * margin, ymax + rng * margin)

nice_ylim(axVI, 0.04)  # Voltage
set_current_ylim_lower_half(axVI_t, i_plot, bottom_margin=0.05, top_space_factor=1.2)
nice_ylim(axT, 0.04)   # Temperature

# ===== 图例（小字号） =====
handles = [lV, lI, lTa, lTs]
labels = ['Voltage', 'Current', 'Ta', 'Ts']
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, 1.02), prop={'size': 8})

# 数值格式
for ax in [axVI, axVI_t, axT]:
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='plain')

# ===== 保存 =====
plt.savefig('battery_vi_temp_dualaxis_noinset.pdf', bbox_inches='tight')
plt.savefig('battery_vi_temp_dualaxis_noinset.png', bbox_inches='tight', dpi=600)

plt.show()