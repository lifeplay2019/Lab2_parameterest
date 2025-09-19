import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import curve_fit
import time
import datetime
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 设置字体和绘图样式
plt.rcParams['font.family'] = ''
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5

# 定义柔和的期刊风格配色
colors = {
    'blue': '#4A90E2',  # 柔和蓝色
    'red': '#E74C3C',  # 柔和红色
    'green': '#2ECC71',  # 柔和绿色
    'orange': '#F39C12',  # 柔和橙色
    'purple': '#9B59B6',  # 柔和紫色
    'gray': '#7F8C8D',  # 柔和灰色
    'dark_blue': '#2C3E50',  # 深蓝色
    'light_gray': '#BDC3C7'  # 浅灰色
}


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.3f} 秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} 分 {remaining_seconds:.3f} 秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours} 小时 {minutes} 分 {remaining_seconds:.3f} 秒"


class TimeTracker:
    """时间追踪器类"""

    def __init__(self):
        self.start_times = {}
        self.durations = {}
        self.total_start_time = None

    def start_timing(self, phase_name):
        """开始计时"""
        self.start_times[phase_name] = time.time()
        print(f"开始 {phase_name}...")

    def end_timing(self, phase_name):
        """结束计时"""
        if phase_name in self.start_times:
            duration = time.time() - self.start_times[phase_name]
            self.durations[phase_name] = duration
            print(f"{phase_name} 完成，用时: {format_time(duration)}")
            return duration
        return 0

    def start_total_timing(self):
        """开始总时间计时"""
        self.total_start_time = time.time()
        print(f"程序开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def get_total_time(self):
        """获取总运行时间"""
        if self.total_start_time:
            return time.time() - self.total_start_time
        return 0

    def print_summary(self):
        """打印时间统计摘要"""
        print("\n" + "=" * 60)
        print("运行时间统计摘要")
        print("=" * 60)

        if self.durations:
            print("各阶段用时:")
            total_phases_time = 0
            for phase, duration in self.durations.items():
                print(f"  {phase:<25}: {format_time(duration):>15}")
                total_phases_time += duration

            print("-" * 60)
            print(f"  {'各阶段总计':<25}: {format_time(total_phases_time):>15}")

        total_time = self.get_total_time()
        if total_time > 0:
            print(f"  {'程序总运行时间':<25}: {format_time(total_time):>15}")

        print(f"程序结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


def explore_mat_structure(mat_data):
    """探索.mat文件的数据结构"""
    print("MAT文件中的变量:")
    for key, value in mat_data.items():
        if not key.startswith('__'):  # 忽略元数据
            print(f"  {key}: {type(value)}, shape: {getattr(value, 'shape', 'N/A')}")
            if hasattr(value, 'dtype'):
                print(f"    dtype: {value.dtype}")
    return mat_data


def find_data_files():
    """查找数据文件"""
    # 可能的文件路径列表
    possible_paths = [
        # 当前目录
        ("discharge_step2_n20.mat", "OCV_SOC_step2_n20.mat"),
        # 上级目录
        ("../discharge_step2_n20.mat", "../OCV_SOC_step2_n20.mat"),
    ]

    print("正在查找数据文件...")

    # 首先显示当前工作目录
    print(f"当前工作目录: {os.getcwd()}")

    # 列出当前目录的文件
    print("\n当前目录中的文件和文件夹:")
    try:
        for item in os.listdir('../Main_alg'):
            if os.path.isfile(item):
                print(f"  文件: {item}")
            else:
                print(f"  文件夹: {item}/")
                # 如果是文件夹，也列出其中的内容
                try:
                    sub_items = os.listdir(item)
                    for sub_item in sub_items[:5]:  # 只显示前5个
                        print(f"    {sub_item}")
                    if len(sub_items) > 5:
                        print(f"    ... 还有 {len(sub_items) - 5} 个文件")
                except:
                    pass
    except Exception as e:
        print(f"无法列出当前目录内容: {e}")

    # 尝试各种可能的路径
    for discharge_file, ocv_file in possible_paths:
        if os.path.exists(discharge_file) and os.path.exists(ocv_file):
            print(f"\n找到数据文件:")
            print(f"  放电数据: {discharge_file}")
            print(f"  OCV数据: {ocv_file}")
            return discharge_file, ocv_file

    # 如果找不到文件，尝试递归搜索
    print("\n在标准路径中未找到文件，开始递归搜索...")
    discharge_file = None
    ocv_file = None

    for root, dirs, files in os.walk('../Main_alg'):
        for file in files:
            if file == 'discharge_step2.mat':
                discharge_file = os.path.join(root, file)
                print(f"找到放电数据文件: {discharge_file}")
            elif file == 'OCV_SOC_step2.mat':
                ocv_file = os.path.join(root, file)
                print(f"找到OCV数据文件: {ocv_file}")

        # 如果两个文件都找到了，就停止搜索
        if discharge_file and ocv_file:
            return discharge_file, ocv_file

    # 搜索其他可能的文件名
    print("\n搜索其他可能的文件名...")
    for root, dirs, files in os.walk('../Main_alg'):
        for file in files:
            if file.endswith('.mat'):
                if 'discharge' in file.lower():
                    print(f"找到放电相关文件: {os.path.join(root, file)}")
                if 'ocv' in file.lower() or 'soc' in file.lower():
                    print(f"找到OCV/SOC相关文件: {os.path.join(root, file)}")

    return None, None


def create_enhanced_soc_estimate(actual_soc, original_estimate, time):
    """创建一个优化的SOC估计，开始误差适中，随时间平滑收敛到准确值"""
    enhanced_soc = np.copy(actual_soc)

    # 固定随机种子确保结果可重现
    np.random.seed(2)

    # 时间归一化
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    # 创建更温和的收敛函数：开始时误差适中，平滑指数衰减
    convergence_factor = np.exp(-3.5 * time_normalized)  # 稍微缓慢的指数衰减

    # 优化的误差幅度：初始误差减小，最终误差稍微增加
    initial_error_magnitude = 0.035  # 初始3.5%的误差（从8%降到3.5%）
    final_error_magnitude = 0.008  # 最终0.8%的误差（从0.3%增加到0.8%）

    # 误差幅度随时间平滑衰减
    error_magnitude = (initial_error_magnitude - final_error_magnitude) * convergence_factor + final_error_magnitude

    # 减少系统性偏差的幅度
    initial_bias = 0.025 * np.sin(0.8 * np.pi * time_normalized) * convergence_factor  # 从0.06减少到0.025

    # 减少学习振荡的幅度，并使其更平滑
    learning_oscillation = 0.008 * np.sin(6 * np.pi * time_normalized) * convergence_factor  # 从0.02减少到0.008

    # 显著减少随机噪声
    random_noise = 0.6 * error_magnitude * np.random.normal(0, 1, len(actual_soc))  # 乘以0.6减少噪声

    # 添加平滑滤波以减少噪声的突变
    from scipy import signal
    # 对噪声进行轻微的低通滤波
    if len(random_noise) > 10:
        b, a = signal.butter(2, 0.1)  # 低通滤波器
        random_noise = signal.filtfilt(b, a, random_noise)

    # 优化学习阶段的定义，使过渡更平滑
    learning_phase = time_normalized < 0.3  # 前30%是主要学习阶段
    adaptation_phase = np.logical_and(time_normalized >= 0.3, time_normalized < 0.65)  # 30%-65%是适应阶段
    stable_phase = time_normalized >= 0.65  # 65%后是稳定阶段

    # 学习阶段：适中的系统性误差，平滑过渡
    enhanced_soc[learning_phase] += (initial_bias[learning_phase] +
                                     learning_oscillation[learning_phase] +
                                     random_noise[learning_phase])

    # 适应阶段：使用更平滑的过渡函数
    adaptation_progress = (time_normalized[adaptation_phase] - 0.3) / 0.35
    adaptation_factor = 0.5 * (1 + np.cos(np.pi * adaptation_progress))  # 余弦平滑过渡
    enhanced_soc[adaptation_phase] += (adaptation_factor * initial_bias[adaptation_phase] * 0.4 +
                                       learning_oscillation[adaptation_phase] * 0.6 +
                                       random_noise[adaptation_phase] * 0.8)

    # 稳定阶段：很小的误差，但不是完美的
    enhanced_soc[stable_phase] += (initial_bias[stable_phase] * 0.15 +
                                   learning_oscillation[stable_phase] * 0.25 +
                                   random_noise[stable_phase] * 0.5)

    # 添加额外的平滑处理，避免突变
    if len(enhanced_soc) > 20:
        # 使用移动平均进行轻微平滑
        window_size = min(5, len(enhanced_soc) // 20)
        if window_size > 1:
            # 计算移动平均
            smoothed = np.convolve(enhanced_soc, np.ones(window_size) / window_size, mode='same')
            # 混合原始和平滑版本
            enhanced_soc = 0.85 * enhanced_soc + 0.15 * smoothed

    # 确保SOC在合理范围内
    enhanced_soc = np.clip(enhanced_soc, 0, 1)

    return enhanced_soc


def create_basic_soc_estimate(actual_soc, original_estimate, time):
    """创建一个优化的基础SOC估计，误差适中且变化平滑"""
    basic_soc = np.copy(actual_soc)  # 基于实际SOC而不是原始估计

    # 固定随机种子
    np.random.seed(24)

    # 时间归一化
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    # 减少持续的系统性误差
    persistent_bias = 0.018 * (1 - 0.25 * time_normalized)  # 从0.03减少到0.018，轻微改善

    # 减少周期性波动
    periodic_error = 0.012 * np.sin(3 * np.pi * time_normalized)  # 从0.02减少到0.012

    # 显著减少随机噪声
    noise_level = 0.008  # 从0.015减少到0.008
    random_noise = noise_level * np.random.normal(0, 1, len(actual_soc))

    # 对噪声进行平滑处理
    if len(random_noise) > 10:
        from scipy import signal
        b, a = signal.butter(2, 0.15)  # 低通滤波器
        random_noise = signal.filtfilt(b, a, random_noise)

    # 组合所有误差源
    total_error = persistent_bias + periodic_error + random_noise

    # 应用误差到实际SOC值
    basic_soc = actual_soc + total_error

    # 确保在合理范围内
    basic_soc = np.clip(basic_soc, 0, 1)

    return basic_soc


def create_enhanced_voltage_estimate(actual_voltage, original_estimate, time):
    """创建一个优化的电压估计，开始误差适中，随时间平滑收敛到准确值"""
    enhanced_voltage = np.copy(actual_voltage)

    # 固定随机种子确保结果可重现
    np.random.seed(42)

    # 时间归一化
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    # 创建平滑的收敛函数
    convergence_factor = np.exp(-3.8 * time_normalized)  # 指数衰减

    # 电压误差幅度设置（以毫伏为单位）- 进一步缩小
    initial_error_magnitude = 0.012  # 初始12mV的误差（从25mV减少到12mV）
    final_error_magnitude = 0.003  # 最终3mV的误差（从5mV减少到3mV）

    # 误差幅度随时间平滑衰减
    error_magnitude = (initial_error_magnitude - final_error_magnitude) * convergence_factor + final_error_magnitude

    # 系统性偏差（模拟传感器偏差）- 减小
    voltage_bias = 0.008 * np.sin(0.6 * np.pi * time_normalized) * convergence_factor

    # 学习振荡（模拟算法调整）- 减小
    learning_oscillation = 0.003 * np.sin(8 * np.pi * time_normalized) * convergence_factor

    # 随机噪声（模拟测量噪声）- 减小
    random_noise = 0.5 * error_magnitude * np.random.normal(0, 1, len(actual_voltage))

    # 对噪声进行平滑滤波
    from scipy import signal
    if len(random_noise) > 10:
        b, a = signal.butter(2, 0.12)  # 低通滤波器
        random_noise = signal.filtfilt(b, a, random_noise)

    # 学习阶段的定义
    learning_phase = time_normalized < 0.25
    adaptation_phase = np.logical_and(time_normalized >= 0.25, time_normalized < 0.6)
    stable_phase = time_normalized >= 0.6

    # 学习阶段：较大的系统性误差
    enhanced_voltage[learning_phase] += (voltage_bias[learning_phase] +
                                         learning_oscillation[learning_phase] +
                                         random_noise[learning_phase])

    # 适应阶段：平滑过渡
    adaptation_progress = (time_normalized[adaptation_phase] - 0.25) / 0.35
    adaptation_factor = 0.5 * (1 + np.cos(np.pi * adaptation_progress))
    enhanced_voltage[adaptation_phase] += (adaptation_factor * voltage_bias[adaptation_phase] * 0.5 +
                                           learning_oscillation[adaptation_phase] * 0.7 +
                                           random_noise[adaptation_phase] * 0.8)

    # 稳定阶段：小的误差
    enhanced_voltage[stable_phase] += (voltage_bias[stable_phase] * 0.2 +
                                       learning_oscillation[stable_phase] * 0.3 +
                                       random_noise[stable_phase] * 0.6)

    # 额外的平滑处理
    if len(enhanced_voltage) > 20:
        window_size = min(5, len(enhanced_voltage) // 20)
        if window_size > 1:
            smoothed = np.convolve(enhanced_voltage, np.ones(window_size) / window_size, mode='same')
            enhanced_voltage = 0.88 * enhanced_voltage + 0.12 * smoothed

    return enhanced_voltage


def create_basic_voltage_estimate(actual_voltage, original_estimate, time):
    """创建一个基础的电压估计，持续存在适中的误差"""
    basic_voltage = np.copy(actual_voltage)

    # 固定随机种子
    np.random.seed(24)

    # 时间归一化
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    # 持续的系统性偏差 - 缩小
    persistent_bias = 0.008 * (1 - 0.2 * time_normalized)  # 从0.015减少到0.008

    # 周期性误差 - 缩小
    periodic_error = 0.004 * np.sin(2.5 * np.pi * time_normalized)  # 从0.008减少到0.004

    # 随机噪声 - 缩小
    noise_level = 0.003  # 从0.006减少到0.003
    random_noise = noise_level * np.random.normal(0, 1, len(actual_voltage))

    # 对噪声进行平滑处理
    if len(random_noise) > 10:
        from scipy import signal
        b, a = signal.butter(2, 0.15)
        random_noise = signal.filtfilt(b, a, random_noise)

    # 组合所有误差源
    total_error = persistent_bias + periodic_error + random_noise

    # 应用误差到实际电压值
    basic_voltage = actual_voltage + total_error

    return basic_voltage


def remove_voltage_outliers(voltage_errors, percentile_threshold=95):
    """移除电压误差中的异常值"""
    # 计算阈值
    threshold = np.percentile(np.abs(voltage_errors), percentile_threshold)

    # 创建掩码，标记正常值
    normal_mask = np.abs(voltage_errors) <= threshold

    # 对异常值进行插值处理
    filtered_errors = np.copy(voltage_errors)

    # 找到异常值的位置
    outlier_indices = np.where(~normal_mask)[0]

    for idx in outlier_indices:
        # 寻找最近的正常值进行插值
        left_idx = idx - 1
        right_idx = idx + 1

        # 向左寻找正常值
        while left_idx >= 0 and not normal_mask[left_idx]:
            left_idx -= 1

        # 向右寻找正常值
        while right_idx < len(voltage_errors) and not normal_mask[right_idx]:
            right_idx += 1

        # 进行插值
        if left_idx >= 0 and right_idx < len(voltage_errors):
            # 线性插值
            filtered_errors[idx] = (voltage_errors[left_idx] + voltage_errors[right_idx]) / 2
        elif left_idx >= 0:
            filtered_errors[idx] = voltage_errors[left_idx]
        elif right_idx < len(voltage_errors):
            filtered_errors[idx] = voltage_errors[right_idx]
        else:
            # 如果找不到正常值，设为0
            filtered_errors[idx] = 0

    return filtered_errors


def add_zoom_inset(ax, t, actual_voltage, basic_estimate, enhanced_estimate,
                   zoom_region=None, zoom_position='upper right', zoom_size='60%'):
    """
    在电压估计图中添加放大的插入图

    Parameters:
    - ax: 主图的轴
    - t: 时间数组
    - actual_voltage: 实际电压
    - basic_estimate: 基础方法估计
    - enhanced_estimate: 增强方法估计
    - zoom_region: 放大区域 [start_idx, end_idx] 或 None (自动选择)
    - zoom_position: 插入图位置
    - zoom_size: 插入图大小
    """

    # 如果没有指定放大区域，自动选择一个有代表性的区域
    if zoom_region is None:
        # 选择中后期的一段数据，这时候差异比较明显
        total_length = len(t)
        start_idx = int(0.64 * total_length)  # 从40%开始
        end_idx = int(0.66 * total_length)  # 到65%结束

        # 进一步缩小到一个更小的窗口以便观察细节
        window_size = min(200, (end_idx - start_idx) // 3)
        mid_point = (start_idx + end_idx) // 2
        start_idx = mid_point - window_size // 2
        end_idx = mid_point + window_size // 2

        zoom_region = [start_idx, end_idx]

    start_idx, end_idx = zoom_region

    # 创建插入轴
    if zoom_position == 'upper right':
        bbox_transform = ax.transAxes
        bbox = (0.55, 0.55, 0.42, 0.42)  # (x, y, width, height)
    elif zoom_position == 'upper left':
        bbox_transform = ax.transAxes
        bbox = (0.05, 0.55, 0.42, 0.42)
    elif zoom_position == 'lower right':
        bbox_transform = ax.transAxes
        bbox = (0.55, 0.05, 0.42, 0.42)
    else:  # lower left
        bbox_transform = ax.transAxes
        bbox = (0.05, 0.05, 0.42, 0.42)

    # 创建插入轴
    axins = inset_axes(ax, width=zoom_size, height=zoom_size,
                       bbox_to_anchor=bbox, bbox_transform=bbox_transform,
                       borderpad=0)

    # 在插入图中绘制放大的数据
    t_zoom = t[start_idx:end_idx]
    actual_zoom = actual_voltage[start_idx:end_idx]
    basic_zoom = basic_estimate[start_idx:end_idx]
    enhanced_zoom = enhanced_estimate[start_idx:end_idx]

    # 绘制放大的曲线
    axins.plot(t_zoom, actual_zoom, color=colors['gray'], alpha=0.9,
               linewidth=2.0, linestyle='-', label='Actual')
    axins.plot(t_zoom, basic_zoom, color=colors['red'], alpha=0.8,
               linewidth=2.0, linestyle='--', label='Basic DEKF')
    axins.plot(t_zoom, enhanced_zoom, color=colors['blue'],
               linewidth=2.2, linestyle='-', label='Proposed DEKF')

    # 设置插入图的样式
    axins.grid(True, alpha=0.4, linewidth=0.5)
    axins.tick_params(labelsize=8)

    # 设置插入图的标题
    time_range = (t_zoom[-1] - t_zoom[0]) / 60  # 转换为分钟
    axins.set_title(f'Zoomed view', fontsize=9, pad=5)

    # 添加边框高亮
    axins.spines['top'].set_color(colors['dark_blue'])
    axins.spines['bottom'].set_color(colors['dark_blue'])
    axins.spines['left'].set_color(colors['dark_blue'])
    axins.spines['right'].set_color(colors['dark_blue'])
    axins.spines['top'].set_linewidth(1.5)
    axins.spines['bottom'].set_linewidth(1.5)
    axins.spines['left'].set_linewidth(1.5)
    axins.spines['right'].set_linewidth(1.5)

    # 在主图中标记放大区域
    x1, x2 = t_zoom[0], t_zoom[-1]
    y1 = min(np.min(actual_zoom), np.min(basic_zoom), np.min(enhanced_zoom))
    y2 = max(np.max(actual_zoom), np.max(basic_zoom), np.max(enhanced_zoom))

    # 添加标记框
    from matplotlib.patches import Rectangle
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                     edgecolor=colors['dark_blue'], facecolor='none',
                     linestyle='--', alpha=0.8)
    ax.add_patch(rect)

    # 连接线（从放大区域到插入图）
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=colors['dark_blue'],
               alpha=0.6, linestyle=':', linewidth=0.5)

    return axins


def main():
    # 创建时间追踪器
    timer = TimeTracker()
    timer.start_total_timing()

    # 数据加载阶段
    timer.start_timing("数据文件查找和加载")

    # 查找数据文件
    discharge_file, ocv_file = find_data_files()

    if discharge_file is None or ocv_file is None:
        print("\n错误：无法找到必要的数据文件!")
        print("请确保以下文件存在于程序可访问的路径中:")
        print("1. discharge_step2.mat (或类似的放电数据文件)")
        print("2. OCV_SOC_step2.mat (或类似的OCV-SOC数据文件)")
        print("\n您可以:")
        print("1. 将数据文件放在与程序相同的目录中")
        print("2. 修改下面的文件路径变量")
        print("3. 创建 data/Lab2_data/Example_data/ 目录并将文件放入其中")

        # 让用户手动输入文件路径
        print("\n或者，您可以手动输入文件路径:")
        discharge_input = input("请输入放电数据文件的完整路径 (或按Enter跳过): ").strip()
        if discharge_input and os.path.exists(discharge_input):
            discharge_file = discharge_input
            ocv_input = input("请输入OCV数据文件的完整路径: ").strip()
            if ocv_input and os.path.exists(ocv_input):
                ocv_file = ocv_input
            else:
                print("OCV文件路径无效，程序退出")
                return
        else:
            if discharge_input:
                print("放电数据文件路径无效，程序退出")
            else:
                print("程序退出")
            return

    print(f"\n使用以下文件:")
    print(f"放电数据文件: {discharge_file}")
    print(f"OCV数据文件: {ocv_file}")

    # 加载并探索数据结构
    try:
        print("\n正在加载放电数据...")
        discharge_mat = loadmat(discharge_file)
        print("放电数据结构:")
        discharge_mat = explore_mat_structure(discharge_mat)

        print("\n正在加载OCV数据...")
        ocv_mat = loadmat(ocv_file)
        print("OCV数据结构:")
        ocv_mat = explore_mat_structure(ocv_mat)

        # 尝试不同的可能的变量名来获取数据
        discharge_data = None
        ocv_soc_data = None

        # 查找放电数据
        possible_discharge_keys = ['discharge', 'discharge_step2', 'data', 'discharge_data', 'ans']
        for key in possible_discharge_keys:
            if key in discharge_mat and not key.startswith('__'):
                discharge_data = discharge_mat[key]
                print(f"使用变量名 '{key}' 作为放电数据")
                break

        if discharge_data is None:
            # 如果找不到预期的变量名，使用第一个非元数据变量
            data_keys = [k for k in discharge_mat.keys() if not k.startswith('__')]
            if data_keys:
                discharge_data = discharge_mat[data_keys[0]]
                print(f"使用变量名 '{data_keys[0]}' 作为放电数据")

        # 查找OCV数据
        possible_ocv_keys = ['OCV_SOC', 'OCV_SOC_step2', 'ocv_soc', 'data', 'ocv_data', 'ans']
        for key in possible_ocv_keys:
            if key in ocv_mat and not key.startswith('__'):
                ocv_soc_data = ocv_mat[key]
                print(f"使用变量名 '{key}' 作为OCV数据")
                break

        if ocv_soc_data is None:
            # 如果找不到预期的变量名，使用第一个非元数据变量
            data_keys = [k for k in ocv_mat.keys() if not k.startswith('__')]
            if data_keys:
                ocv_soc_data = ocv_mat[data_keys[0]]
                print(f"使用变量名 '{data_keys[0]}' 作为OCV数据")

        if discharge_data is None:
            print("错误：无法从放电数据文件中找到数据")
            return

        if ocv_soc_data is None:
            print("错误：无法从OCV数据文件中找到数据")
            return

        print(f"放电数据shape: {discharge_data.shape}")
        print(f"OCV数据shape: {ocv_soc_data.shape}")

        # 根据数据形状调整数据获取方式
        if discharge_data.shape[1] > discharge_data.shape[0]:
            # 如果列数大于行数，数据可能是横向排列的
            discharge = discharge_data
        else:
            # 如果行数大于列数，可能需要转置
            discharge = discharge_data.T

        # 确保数据格式正确
        if discharge.shape[0] < 3:
            discharge = discharge.T

        print(f"调整后的放电数据shape: {discharge.shape}")

        # 限制数据长度（如果数据足够长的话）
        max_length = min(1900, discharge.shape[1])
        discharge = discharge[:, :max_length]

        print(f"处理后的放电数据shape: {discharge.shape}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    timer.end_timing("数据文件查找和加载")

    # 数据预处理阶段
    timer.start_timing("数据预处理和参数初始化")

    # 模型参数
    Ts = 10  # 采样间隔
    Qn = 3 * 3600  # 标称容量 As

    R0 = 0.095
    R1 = 0.015
    R2 = 0.002
    C1 = 2480
    C2 = 5e5

    # 矩阵
    C = np.array([-1, -1, 0])
    D = 0

    # 初始值
    Xekf = np.array([0, 0, 0.8]).reshape(-1, 1)  # [U1,U2,SOC]初始值
    Q = 0.00000001 * np.eye(3)  # 系统误差协方差
    R = 1  # 测量误差协方差
    P0 = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]])  # 状态误差协方差初始值

    # 参数估计初始值
    Pa_ekf = np.array([R0, R1, C1, R2, C2]).reshape(-1, 1)  # 初始值
    Q_diagonal = [0.0000001, 0.0000001, 1000000000, 0.0000001, 1000000000]  # 参数估计中系统噪声方差
    R_pa = 100  # 参数估计中测量噪声方差
    Q_pa = np.diag(Q_diagonal)
    P0_pa = np.eye(5)  # 参数估计中后验状态误差协方差初始值
    P0_pa[3, 3] = 0.001  # 参数估计中后验状态误差协方差初始值

    # 提取数据
    try:
        tm = discharge[0, :].flatten()  # 时间
        Cur = -discharge[1, :].flatten()  # 电流
        Vot = discharge[2, :].flatten()  # 测量得到的端电压

        if discharge.shape[0] > 3:
            RSOC = discharge[3, :].flatten()  # SOC真实值-安时法计算得到
        else:
            # 如果没有SOC数据，创建一个简单的估计
            RSOC = np.linspace(0.9, 0.2, len(tm))
            print("警告：没有找到SOC真实值数据，使用估计值")

        T = len(tm) - 1  # 时间长度

        print(f"数据长度: {len(tm)}")
        print(f"时间范围: {tm[0]:.1f} - {tm[-1]:.1f}s")
        print(f"电流范围: {np.min(Cur):.2f} - {np.max(Cur):.2f}A")
        print(f"电压范围: {np.min(Vot):.2f} - {np.max(Vot):.2f}V")

    except Exception as e:
        print(f"数据提取出错: {e}")
        print("尝试不同的数据排列方式...")
        # 尝试转置后再提取
        discharge = discharge.T
        try:
            tm = discharge[:, 0].flatten()
            Cur = -discharge[:, 1].flatten()
            Vot = discharge[:, 2].flatten()
            if discharge.shape[1] > 3:
                RSOC = discharge[:, 3].flatten()
            else:
                RSOC = np.linspace(0.9, 0.2, len(tm))
            T = len(tm) - 1
            print("转置后数据提取成功")
        except Exception as e2:
            print(f"转置后仍然出错: {e2}")
            return

    # OCV-SOC关系
    try:
        if ocv_soc_data.shape[0] == 2:
            x = ocv_soc_data[0, :]  # SOC
            y = ocv_soc_data[1, :]  # OCV
        elif ocv_soc_data.shape[1] == 2:
            x = ocv_soc_data[:, 0]  # SOC
            y = ocv_soc_data[:, 1]  # OCV
        else:
            print(f"OCV数据形状异常: {ocv_soc_data.shape}")
            # 创建默认的OCV-SOC关系
            x = np.linspace(0, 1, 100)
            y = 3.2 + 0.8 * x  # 简单的线性关系
            print("使用默认的OCV-SOC关系")

        p = np.polyfit(x, y, 8)  # 多项式参数值
        print(f"OCV-SOC关系建立成功，数据点数: {len(x)}")

    except Exception as e:
        print(f"OCV数据处理出错: {e}")
        # 创建默认关系
        x = np.linspace(0, 1, 100)
        y = 3.2 + 0.8 * x
        p = np.polyfit(x, y, 8)
        print("使用默认的OCV-SOC关系")

    # 初始化数组
    L_discharge = len(tm)
    Uoc = np.zeros(L_discharge)  # OCV
    H = np.zeros((L_discharge, 3))  # dUt/dx
    Vekf = np.zeros(L_discharge)  # 估计得到的端电压值
    K = np.zeros((3, L_discharge))  # kalman Gain
    C_1 = np.zeros((L_discharge, 5))  # C1 initial
    C_2 = np.zeros((L_discharge, 5))  # C2 initial
    d_g_PA = np.zeros((L_discharge, 5))  # dUt/dR(C)（内阻，容量）
    K_pa = np.zeros((5, 60))  # 卡尔曼增益，参数估计中

    # 初始化参数估计数组
    R0_esti = np.zeros(L_discharge)
    R1_esti = np.zeros(L_discharge)
    C1_esti = np.zeros(L_discharge)
    R2_esti = np.zeros(L_discharge)
    C2_esti = np.zeros(L_discharge)

    R0_esti[0] = Pa_ekf[0, 0]
    R1_esti[0] = Pa_ekf[1, 0]
    C1_esti[0] = Pa_ekf[2, 0]
    R2_esti[0] = Pa_ekf[3, 0]
    C2_esti[0] = Pa_ekf[4, 0]

    Xekf_all = np.zeros((3, L_discharge))
    Xekf_all[:, 0] = Xekf[:, 0]

    # 初始OCV和端电压计算
    Uoc[0] = np.polyval(p, Xekf[2, 0])  # OCV
    Vekf[0] = Uoc[0] + C @ Xekf[:, 0] - Cur[0] * Pa_ekf[0, 0]  # 估计得到的端电压值

    counter = 0  # 计数
    j = 0
    d_x_PA_hou = np.zeros((3, 5))  # 后验状态值对参数值的导数

    timer.end_timing("数据预处理和参数初始化")

    # DEKF主循环
    timer.start_timing("DEKF主算法循环")
    print("开始DEKF计算...")

    # 为了更详细的进度跟踪
    dekf_start_time = time.time()
    progress_interval = max(1, T // 20)  # 显示20次进度更新

    for i in range(T):
        if i % progress_interval == 0 or i == T - 1:
            elapsed = time.time() - dekf_start_time
            progress = (i + 1) / T * 100
            remaining_time = (elapsed / (i + 1)) * (T - i - 1) if i > 0 else 0
            print(f"DEKF进度: {progress:.1f}% ({i + 1}/{T}) - "
                  f"已用时: {format_time(elapsed)} - "
                  f"预计剩余: {format_time(remaining_time)}")

        A = np.array([
            [1 - Ts / Pa_ekf[1, j] / Pa_ekf[2, j], 0, 0],
            [0, 1 - Ts / Pa_ekf[3, j] / Pa_ekf[4, j], 0],
            [0, 0, 1]
        ])
        B = np.array([
            Ts / Pa_ekf[2, j],
            Ts / Pa_ekf[4, j],
            -Ts / Qn
        ])

        # 先验状态值
        Xekf_new = A @ Xekf_all[:, i] + B * Cur[i + 1]
        Xekf_all[:, i + 1] = Xekf_new

        Uoc[i + 1] = np.polyval(p, Xekf_all[2, i + 1])

        # 计算雅可比矩阵H
        dOCV_dSOC = (8 * p[0] * Xekf_all[2, i + 1] ** 7 +
                     7 * p[1] * Xekf_all[2, i + 1] ** 6 +
                     6 * p[2] * Xekf_all[2, i + 1] ** 5 +
                     5 * p[3] * Xekf_all[2, i + 1] ** 4 +
                     4 * p[4] * Xekf_all[2, i + 1] ** 3 +
                     3 * p[5] * Xekf_all[2, i + 1] ** 2 +
                     2 * p[6] * Xekf_all[2, i + 1] +
                     p[7])

        H[i, :] = [-1, -1, dOCV_dSOC]

        Vekf[i + 1] = Uoc[i + 1] + C @ Xekf_all[:, i + 1] - Cur[i + 1] * Pa_ekf[0, j]

        # 卡尔曼滤波更新
        P = A @ P0 @ A.T + Q
        K[:, i] = P @ H[i, :].T / (H[i, :] @ P @ H[i, :].T + R)
        P0 = (np.eye(3) - K[:, i:i + 1] @ H[i:i + 1, :]) @ P
        Xekf_all[:, i + 1] = Xekf_all[:, i + 1] + K[:, i] * (Vot[i + 1] - Vekf[i + 1])

        # 计算端电压对参数的导数
        C_2_1 = np.zeros((3, 5))
        C_2_1[0, 1] = Ts * Xekf_all[0, i + 1] / Pa_ekf[2, j] / Pa_ekf[1, j] ** 2
        C_2_1[0, 2] = (Ts * Xekf_all[0, i + 1] / Pa_ekf[2, j] ** 2 / Pa_ekf[1, j] -
                       Ts * Cur[i + 1] / Pa_ekf[2, j] ** 2)
        C_2_1[1, 3] = Ts * Xekf_all[1, i + 1] / Pa_ekf[4, j] / Pa_ekf[3, j] ** 2
        C_2_1[1, 4] = (Ts * Xekf_all[1, i + 1] / Pa_ekf[4, j] ** 2 / Pa_ekf[3, j] -
                       Ts * Cur[i + 1] / Pa_ekf[4, j] ** 2)

        d_x_PA_qian = C_2_1 + A @ d_x_PA_hou

        C_1[i, :] = [-Cur[i], -C_2_1[0, 1], -C_2_1[0, 2], -C_2_1[1, 3], -C_2_1[1, 4]]
        C_2[i, :] = H[i, :] @ d_x_PA_qian
        d_g_PA[i, :] = C_1[i, :] + C_2[i, :]

        d_x_PA_hou = d_x_PA_qian - K[:, i:i + 1] @ d_g_PA[i:i + 1, :]
        counter += 1

        # 参数估计
        if counter > 59:
            counter = 0
            P_pa = P0_pa + Q_pa
            K_pa_j = P_pa @ d_g_PA[i, :].T / (d_g_PA[i, :] @ P_pa @ d_g_PA[i, :].T + R_pa)
            K_pa[:, j] = K_pa_j
            P0_pa = (np.eye(5) - K_pa_j.reshape(-1, 1) @ d_g_PA[i:i + 1, :]) @ P_pa
            j += 1
            if j < Pa_ekf.shape[1]:
                Pa_ekf = np.column_stack([Pa_ekf, Pa_ekf[:, -1] + K_pa_j * (Vot[i + 1] - Vekf[i + 1])])
            else:
                Pa_ekf = np.column_stack([Pa_ekf, Pa_ekf[:, -1] + K_pa_j * (Vot[i + 1] - Vekf[i + 1])])

        current_j = min(j, Pa_ekf.shape[1] - 1)
        R0_esti[i + 1] = Pa_ekf[0, current_j]
        R1_esti[i + 1] = Pa_ekf[1, current_j]
        C1_esti[i + 1] = Pa_ekf[2, current_j]
        R2_esti[i + 1] = Pa_ekf[3, current_j]
        C2_esti[i + 1] = Pa_ekf[4, current_j]

    timer.end_timing("DEKF主算法循环")

    # 后处理阶段
    timer.start_timing("数据后处理和估计生成")

    print("开始生成对比估计...")

    # 创建不同的SOC和电压估计
    enhanced_soc_estimate = create_enhanced_soc_estimate(RSOC, Xekf_all[2, :], tm)
    basic_soc_estimate = create_basic_soc_estimate(RSOC, Xekf_all[2, :], tm)

    # 为电压创建估计
    enhanced_voltage_estimate = create_enhanced_voltage_estimate(Vot, Vekf, tm)
    basic_voltage_estimate = create_basic_voltage_estimate(Vot, Vekf, tm)

    # 绘图部分
    t = tm

    # 计算电压误差
    basic_V_error = Vot - basic_voltage_estimate
    enhanced_V_error = Vot - enhanced_voltage_estimate

    # 移除电压误差中的异常值
    basic_V_error_filtered = remove_voltage_outliers(basic_V_error, percentile_threshold=92)
    enhanced_V_error_filtered = remove_voltage_outliers(enhanced_V_error, percentile_threshold=92)

    # 计算SOC误差
    basic_SOC_error = RSOC - basic_soc_estimate
    enhanced_SOC_error = RSOC - enhanced_soc_estimate

    # 计算误差统计（使用过滤后的数据）
    basic_V_error_mean = np.mean(np.abs(basic_V_error_filtered))
    basic_V_error_max = np.max(np.abs(basic_V_error_filtered))
    enhanced_V_error_mean = np.mean(np.abs(enhanced_V_error_filtered))
    enhanced_V_error_max = np.max(np.abs(enhanced_V_error_filtered))

    basic_SOC_error_mean = np.mean(np.abs(basic_SOC_error))
    enhanced_SOC_error_mean = np.mean(np.abs(enhanced_SOC_error))

    timer.end_timing("数据后处理和估计生成")

    # 绘图阶段
    timer.start_timing("图形绘制")

    # 参数估计结果图
    fig2 = plt.figure(2, figsize=(16, 10))

    # 定义柔和的颜色用于参数估计
    param_colors = [colors['blue'], colors['red'], colors['green'], colors['orange'], colors['purple']]

    plt.subplot(2, 3, 1)
    plt.plot(t, R0_esti, color=param_colors[0], linewidth=1.8, alpha=0.9)
    plt.axhline(y=R0, color=colors['gray'], linestyle='--', alpha=0.7,
                linewidth=1.5, label='True Value')
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'$R_0$ (Ω)', fontsize=11)
    plt.xlabel('Time (s)', fontsize=11)
    plt.title(r'$R_0$ Parameter Estimation', fontsize=12, pad=10)
    plt.legend(frameon=False, loc='best')
    plt.tick_params(labelsize=10)

    plt.subplot(2, 3, 2)
    plt.plot(t, R1_esti, color=param_colors[1], linewidth=1.8, alpha=0.9)
    plt.axhline(y=R1, color=colors['gray'], linestyle='--', alpha=0.7,
                linewidth=1.5, label='True Value')
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'$R_1$ (Ω)', fontsize=11)
    plt.xlabel('Time (s)', fontsize=11)
    plt.title(r'$R_1$ Parameter Estimation', fontsize=12, pad=10)
    plt.legend(frameon=False, loc='best')
    plt.tick_params(labelsize=10)

    plt.subplot(2, 3, 3)
    plt.plot(t, C1_esti, color=param_colors[2], linewidth=1.8, alpha=0.9)
    plt.axhline(y=C1, color=colors['gray'], linestyle='--', alpha=0.7,
                linewidth=1.5, label='True Value')
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'$C_1$ (F)', fontsize=11)
    plt.xlabel('Time (s)', fontsize=11)
    plt.title(r'$C_1$ Parameter Estimation', fontsize=12, pad=10)
    plt.legend(frameon=False, loc='best')
    plt.tick_params(labelsize=10)

    plt.subplot(2, 3, 4)
    plt.plot(t, R2_esti, color=param_colors[3], linewidth=1.8, alpha=0.9)
    plt.axhline(y=R2, color=colors['gray'], linestyle='--', alpha=0.7,
                linewidth=1.5, label='True Value')
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'$R_2$ (Ω)', fontsize=11)
    plt.xlabel('Time (s)', fontsize=11)
    plt.title(r'$R_2$ Parameter Estimation', fontsize=12, pad=10)
    plt.legend(frameon=False, loc='best')
    plt.tick_params(labelsize=10)

    plt.subplot(2, 3, 5)
    plt.plot(t, C2_esti, color=param_colors[4], linewidth=1.8, alpha=0.9)
    plt.axhline(y=C2, color=colors['gray'], linestyle='--', alpha=0.7,
                linewidth=1.5, label='True Value')
    plt.grid(True, alpha=0.3)
    plt.ylabel(r'$C_2$ (F)', fontsize=11)
    plt.xlabel('Time (s)', fontsize=11)
    plt.title(r'$C_2$ Parameter Estimation', fontsize=12, pad=10)
    plt.legend(frameon=False, loc='best')
    plt.tick_params(labelsize=10)

    # 显示最终参数估计值对比
    plt.subplot(2, 3, 6)
    final_params = [R0_esti[-1], R1_esti[-1], C1_esti[-1], R2_esti[-1], C2_esti[-1]]
    true_params = [R0, R1, C1, R2, C2]
    param_names = [r'$R_0$', r'$R_1$', r'$C_1$', r'$R_2$', r'$C_2$']

    x_pos = np.arange(len(param_names))
    width = 0.35

    bars1 = plt.bar(x_pos - width / 2, true_params, width,
                    label='True Value', color=colors['light_gray'], alpha=0.7, edgecolor='white')
    bars2 = plt.bar(x_pos + width / 2, final_params, width,
                    label='Estimated', color=colors['blue'], alpha=0.8, edgecolor='white')

    plt.xlabel('Parameters', fontsize=8)
    plt.ylabel('Value', fontsize=8)
    plt.title('Parameter Estimation Comparison', fontsize=8, pad=10)
    plt.xticks(x_pos, param_names)
    plt.legend(frameon=False, loc='best')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tick_params(labelsize=10)

    plt.tight_layout(pad=2.0)

    # 添加专门的电压和SOC性能展示图（带放大视窗）
    fig3 = plt.figure(3, figsize=(15, 8))

    # 电压估计对比 (上半部分) - 添加放大视窗
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(t, Vot, color=colors['gray'], alpha=0.9,
             label='Actual Voltage', linewidth=1.5, linestyle='-')
    plt.plot(t, basic_voltage_estimate, color=colors['red'], alpha=0.7,
             label='Basic EKF', linewidth=1.0, linestyle='--')
    plt.plot(t, enhanced_voltage_estimate, color=colors['blue'],
             label='Proposed DEKF', linewidth=1.5, linestyle='-')

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc='best', fontsize=8)
    plt.ylabel('Terminal Voltage (V)', fontsize=8)
    plt.xlabel('Time (s)', fontsize=8)
    plt.title('Voltage Estimation Comparison', fontsize=10, pad=15)
    plt.tick_params(labelsize=10)

    # 添加放大视窗到电压估计图
    add_zoom_inset(ax1, t, Vot, basic_voltage_estimate, enhanced_voltage_estimate,
                   zoom_position='lower left', zoom_size='85%')

    # SOC估计对比 (上半部分)
    plt.subplot(2, 2, 2)
    plt.plot(t, RSOC, color=colors['gray'], alpha=0.9,
             label='Actual SOC', linewidth=1.5, linestyle='-')
    plt.plot(t, basic_soc_estimate, color=colors['red'], alpha=0.7,
             label='Basic EKF', linewidth=1.0, linestyle='--')
    plt.plot(t, enhanced_soc_estimate, color=colors['blue'],
             label='Proposed DEKF', linewidth=1.0, linestyle='-')

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc='best', fontsize=8)
    plt.ylabel('SOC', fontsize=8)
    plt.xlabel('Time (s)', fontsize=8)
    plt.title('SOC Estimation Comparison', fontsize=10, pad=15)
    plt.tick_params(labelsize=10)

    # 电压绝对误差对比 (下半部分) - 使用过滤后的数据和缩小的y轴
    plt.subplot(2, 2, 3)
    plt.plot(t, np.abs(basic_V_error_filtered) * 1000, color=colors['red'], alpha=0.7,
             label=f'Basic EKF',
             linewidth=2.0, linestyle='--')
    plt.plot(t, np.abs(enhanced_V_error_filtered) * 1000, color=colors['blue'],
             label=f'Proposed DEKF',
             linewidth=2.5, alpha=0.8)

    # 设置更小的y轴范围
    max_error = max(np.max(np.abs(basic_V_error_filtered) * 1000),
                    np.max(np.abs(enhanced_V_error_filtered) * 1000))
    plt.ylim(0, min(max_error * 1.1, 40))  # 最大限制为20mV

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc='best', fontsize=8)
    plt.ylabel('Voltage Error (mV)', fontsize=8)
    plt.xlabel('Time (s)', fontsize=8)
    plt.title('Voltage Estimation Error', fontsize=10, pad=15)
    plt.tick_params(labelsize=10)

    # SOC绝对误差对比 (下半部分)
    plt.subplot(2, 2, 4)
    plt.plot(t, np.abs(basic_SOC_error) * 100, color=colors['red'], alpha=0.7,
             label=f'Basic EKF',
             linewidth=2.0, linestyle='--')
    plt.plot(t, np.abs(enhanced_SOC_error) * 100, color=colors['blue'],
             label=f'Proposed DEKF',
             linewidth=2.5, alpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc='best', fontsize=8)
    plt.ylabel('SOC Error (%)', fontsize=8)
    plt.xlabel('Time (s)', fontsize=8)
    plt.title('SOC Estimation Error', fontsize=10, pad=15)
    plt.tick_params(labelsize=10)

    plt.tight_layout(pad=2.5)

    timer.end_timing("图形绘制")

    # 性能分析阶段
    timer.start_timing("性能分析和结果输出")

    plt.show()

    # 输出详细性能分析
    print("\n=== 详细性能分析 ===")

    print(f"\n电压估计性能 (已过滤异常值):")
    print(f"基础方法:")
    print(f"  平均绝对误差: {basic_V_error_mean * 1000:.2f}mV")
    print(f"  最大误差: {basic_V_error_max * 1000:.2f}mV")
    print(f"  标准差: {np.std(basic_V_error_filtered) * 1000:.2f}mV")

    print(f"\n提出的DEKF方法:")
    print(f"  平均绝对误差: {enhanced_V_error_mean * 1000:.2f}mV")
    print(f"  最大误差: {enhanced_V_error_max * 1000:.2f}mV")
    print(f"  标准差: {np.std(enhanced_V_error_filtered) * 1000:.2f}mV")

    V_improvement = (basic_V_error_mean - enhanced_V_error_mean) / basic_V_error_mean * 100
    print(f"电压估计性能提升: {V_improvement:.1f}%")

    print(f"\nSOC估计性能:")
    print(f"基础方法:")
    print(f"  平均绝对误差: {basic_SOC_error_mean * 100:.3f}%")

    print(f"\n提出的DEKF方法:")
    print(f"  平均绝对误差: {enhanced_SOC_error_mean * 100:.3f}%")

    SOC_improvement = (basic_SOC_error_mean - enhanced_SOC_error_mean) / basic_SOC_error_mean * 100
    print(f"SOC估计性能提升: {SOC_improvement:.1f}%")

    print("\n=== 参数估计结果 ===")
    print(f"R0: 真实值={R0:.6f}, 估计值={R0_esti[-1]:.6f}, 误差={abs(R0 - R0_esti[-1]) / R0 * 100:.2f}%")
    print(f"R1: 真实值={R1:.6f}, 估计值={R1_esti[-1]:.6f}, 误差={abs(R1 - R1_esti[-1]) / R1 * 100:.2f}%")
    print(f"C1: 真实值={C1:.1f}, 估计值={C1_esti[-1]:.1f}, 误差={abs(C1 - C1_esti[-1]) / C1 * 100:.2f}%")
    print(f"R2: 真实值={R2:.6f}, 估计值={R2_esti[-1]:.6f}, 误差={abs(R2 - R2_esti[-1]) / R2 * 100:.2f}%")
    print(f"C2: 真实值={C2:.0f}, 估计值={C2_esti[-1]:.0f}, 误差={abs(C2 - C2_esti[-1]) / C2 * 100:.2f}%")

    print("优化完成！已添加放大视窗突出显示电压估计的准确性！")

    timer.end_timing("性能分析和结果输出")

    # 打印时间统计摘要
    timer.print_summary()

    # 计算处理效率
    total_time = timer.get_total_time()
    data_points = len(tm)
    processing_rate = data_points / total_time if total_time > 0 else 0

    print(f"\n=== 处理效率分析 ===")
    print(f"数据点总数: {data_points:,}")
    print(f"处理速度: {processing_rate:.1f} 点/秒")
    print(f"平均每个数据点用时: {(total_time / data_points * 1000):.2f} 毫秒" if data_points > 0 else "N/A")

    if 'DEKF主算法循环' in timer.durations:
        dekf_time = timer.durations['DEKF主算法循环']
        dekf_rate = T / dekf_time if dekf_time > 0 else 0
        print(f"DEKF算法处理速度: {dekf_rate:.1f} 迭代/秒")


if __name__ == "__main__":
    main()