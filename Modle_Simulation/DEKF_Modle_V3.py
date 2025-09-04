import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体显示和解决负号显示问题
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    # 设置字体大小
    plt.rcParams['font.size'] = 10
except:
    pass


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
        ("discharge_step2.mat", "OCV_SOC_step2.mat"),
        ("data/discharge_step2.mat", "data/OCV_SOC_step2.mat"),
        ("../discharge_step2.mat", "../OCV_SOC_step2.mat"),
    ]

    print("正在查找数据文件...")
    print(f"当前工作目录: {os.getcwd()}")

    # 列出当前目录的文件
    print("\n当前目录中的文件和文件夹:")
    try:
        for item in os.listdir('.'):
            if os.path.isfile(item):
                print(f"  文件: {item}")
            else:
                print(f"  文件夹: {item}/")
    except Exception as e:
        print(f"无法列出当前目录内容: {e}")

    # 尝试各种可能的路径
    for discharge_file, ocv_file in possible_paths:
        if os.path.exists(discharge_file) and os.path.exists(ocv_file):
            print(f"\n找到数据文件:")
            print(f"  放电数据: {discharge_file}")
            print(f"  OCV数据: {ocv_file}")
            return discharge_file, ocv_file

    return None, None


def adaptive_noise_estimation(innovation_sequence, window_size=100):
    """自适应噪声估计"""
    if len(innovation_sequence) < window_size:
        return 0.05

    # 使用滑动窗口计算方差，并添加遗忘因子
    recent_innovations = innovation_sequence[-window_size:]
    R_adaptive = np.var(recent_innovations)

    # 限制噪声估计范围，避免过大或过小
    R_adaptive = max(min(R_adaptive, 1.0), 0.001)
    return R_adaptive


def estimate_initial_parameters(tm, Cur, Vot, ocv_interp, initial_soc=0.9):
    """基于数据估计初始参数"""
    print("开始估计初始参数...")

    # 找到电流变化较大的时刻来估计内阻
    current_changes = np.abs(np.diff(Cur))
    large_change_indices = np.where(current_changes > 0.1)[0]

    if len(large_change_indices) > 0:
        # 估计R0（欧姆内阻）
        idx = large_change_indices[0]
        if idx > 0 and idx < len(Vot) - 1:
            dV = Vot[idx + 1] - Vot[idx]
            dI = Cur[idx + 1] - Cur[idx]
            if abs(dI) > 0.01:
                R0_est = abs(dV / dI)
                R0_est = max(0.01, min(R0_est, 0.5))  # 限制范围
            else:
                R0_est = 0.08
        else:
            R0_est = 0.08
    else:
        R0_est = 0.08

    # 根据电池类型设置合理的初始参数
    R1_est = 0.02  # 极化内阻1
    R2_est = 0.005  # 极化内阻2
    C1_est = 3000  # 极化电容1
    C2_est = 300000  # 极化电容2

    print(f"估计的初始参数:")
    print(f"  R0: {R0_est:.6f} Ω")
    print(f"  R1: {R1_est:.6f} Ω")
    print(f"  R2: {R2_est:.6f} Ω")
    print(f"  C1: {C1_est:.0f} F")
    print(f"  C2: {C2_est:.0f} F")

    return R0_est, R1_est, R2_est, C1_est, C2_est


def main():
    # 查找数据文件
    discharge_file, ocv_file = find_data_files()

    if discharge_file is None or ocv_file is None:
        print("\n错误：无法找到必要的数据文件!")
        return

    print(f"\n使用以下文件:")
    print(f"放电数据文件: {discharge_file}")
    print(f"OCV数据文件: {ocv_file}")

    # 加载数据
    try:
        print("\n正在加载放电数据...")
        discharge_mat = loadmat(discharge_file)
        print("放电数据结构:")
        discharge_mat = explore_mat_structure(discharge_mat)

        print("\n正在加载OCV数据...")
        ocv_mat = loadmat(ocv_file)
        print("OCV数据结构:")
        ocv_mat = explore_mat_structure(ocv_mat)

        # 获取数据
        discharge_data = discharge_mat['discharge']
        ocv_soc_data = ocv_mat['OCV_SOC']

        print(f"放电数据shape: {discharge_data.shape}")
        print(f"OCV数据shape: {ocv_soc_data.shape}")

        # 处理放电数据
        discharge = discharge_data
        max_length = min(1900, discharge.shape[1])
        discharge = discharge[:, :max_length]
        print(f"处理后的放电数据shape: {discharge.shape}")

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 提取数据
    try:
        tm = discharge[0, :].flatten()  # 时间
        Cur = -discharge[1, :].flatten()  # 电流（取负值）
        Vot = discharge[2, :].flatten()  # 测量端电压
        RSOC = discharge[3, :].flatten()  # SOC真实值

        T = len(tm) - 1  # 时间长度
        print(f"数据长度: {len(tm)}")
        print(f"时间范围: {tm[0]:.1f} - {tm[-1]:.1f}s")
        print(f"电流范围: {np.min(Cur):.2f} - {np.max(Cur):.2f}A")
        print(f"电压范围: {np.min(Vot):.2f} - {np.max(Vot):.2f}V")

    except Exception as e:
        print(f"数据提取出错: {e}")
        return

    # 优化OCV-SOC关系
    try:
        x = ocv_soc_data[0, :]  # SOC
        y = ocv_soc_data[1, :]  # OCV

        # 使用更高阶多项式和插值
        p = np.polyfit(x, y, 10)  # 10阶多项式
        ocv_interp = interp1d(x, y, kind='cubic', bounds_error=False, fill_value='extrapolate')
        print(f"OCV-SOC关系建立成功，数据点数: {len(x)}")

    except Exception as e:
        print(f"OCV数据处理出错: {e}")
        return

    # 基于数据估计初始参数
    R0_init, R1_init, R2_init, C1_init, C2_init = estimate_initial_parameters(
        tm, Cur, Vot, ocv_interp, initial_soc=RSOC[0])

    # 模型参数
    Ts = 10  # 采样间隔
    Qn = 3 * 3600  # 标称容量 As

    # 使用真实参考值（用于对比）
    R0 = 0.095
    R1 = 0.015
    R2 = 0.002
    C1 = 2480
    C2 = 5e5

    # 矩阵
    C_matrix = np.array([-1, -1, 0])

    # 状态初始值 [U1, U2, SOC]
    Xekf = np.array([0, 0, RSOC[0]]).reshape(-1, 1)  # 使用实际初始SOC

    # 优化的噪声协方差矩阵
    Q = np.diag([1e-9, 1e-9, 1e-11])  # 系统噪声
    R_noise = 0.01  # 初始测量噪声
    P0 = np.diag([0.001, 0.001, 0.01])  # 初始协方差

    # 参数估计初始值（使用估计的参数作为初值）
    Pa_ekf = np.array([R0_init, R1_init, C1_init, R2_init, C2_init]).reshape(-1, 1)

    # 优化参数估计噪声
    Q_diagonal = [1e-8, 1e-8, 1e4, 1e-8, 1e4]
    R_pa = 1.0  # 参数估计测量噪声
    Q_pa = np.diag(Q_diagonal)
    P0_pa = 0.01 * np.eye(5)  # 初始参数协方差
    P0_pa[2, 2] = 1e4  # C1
    P0_pa[4, 4] = 1e8  # C2

    # 初始化数组
    L_discharge = len(tm)
    Uoc = np.zeros(L_discharge)
    H = np.zeros((L_discharge, 3))
    Vekf = np.zeros(L_discharge)
    K = np.zeros((3, L_discharge))

    # 参数估计相关数组
    max_param_updates = max(100, L_discharge // 15)
    K_pa = np.zeros((5, max_param_updates))
    d_g_PA = np.zeros((L_discharge, 5))

    # 参数估计结果数组
    R0_esti = np.zeros(L_discharge)
    R1_esti = np.zeros(L_discharge)
    C1_esti = np.zeros(L_discharge)
    R2_esti = np.zeros(L_discharge)
    C2_esti = np.zeros(L_discharge)

    # 设置初始参数估计值
    R0_esti[0] = Pa_ekf[0, 0]
    R1_esti[0] = Pa_ekf[1, 0]
    C1_esti[0] = Pa_ekf[2, 0]
    R2_esti[0] = Pa_ekf[3, 0]
    C2_esti[0] = Pa_ekf[4, 0]

    Xekf_all = np.zeros((3, L_discharge))
    Xekf_all[:, 0] = Xekf[:, 0]

    # 初始OCV和端电压
    Uoc[0] = ocv_interp(Xekf[2, 0])
    Vekf[0] = Uoc[0] + C_matrix @ Xekf[:, 0] - Cur[0] * Pa_ekf[0, 0]

    # 循环变量
    counter = 0
    j = 0
    innovation_sequence = []
    R_adaptive = R_noise
    update_interval = 20  # 参数更新间隔

    # 添加遗忘因子用于参数估计
    forgetting_factor = 0.99

    print("开始优化的DEKF计算...")

    # DEKF主循环
    for i in range(T):
        if i % 300 == 0:
            print(f"处理进度: {i}/{T} ({100 * i / T:.1f}%)")

        # 当前参数索引
        current_j = min(j, Pa_ekf.shape[1] - 1)

        # 参数边界约束（更严格）
        Pa_ekf[0, current_j] = np.clip(Pa_ekf[0, current_j], 0.01, 0.5)  # R0
        Pa_ekf[1, current_j] = np.clip(Pa_ekf[1, current_j], 0.001, 0.1)  # R1
        Pa_ekf[2, current_j] = np.clip(Pa_ekf[2, current_j], 500, 20000)  # C1
        Pa_ekf[3, current_j] = np.clip(Pa_ekf[3, current_j], 0.0001, 0.05)  # R2
        Pa_ekf[4, current_j] = np.clip(Pa_ekf[4, current_j], 50000, 2e6)  # C2

        # 状态转移矩阵
        tau1 = Pa_ekf[1, current_j] * Pa_ekf[2, current_j]
        tau2 = Pa_ekf[3, current_j] * Pa_ekf[4, current_j]

        # 避免除零
        if tau1 < 1e-6:
            tau1 = 1e-6
        if tau2 < 1e-6:
            tau2 = 1e-6

        A = np.array([
            [np.exp(-Ts / tau1), 0, 0],
            [0, np.exp(-Ts / tau2), 0],
            [0, 0, 1]
        ])

        B = np.array([
            Pa_ekf[1, current_j] * (1 - np.exp(-Ts / tau1)),
            Pa_ekf[3, current_j] * (1 - np.exp(-Ts / tau2)),
            -Ts / Qn
        ])

        # 预测步
        Xekf_pred = A @ Xekf_all[:, i] + B * Cur[i + 1]

        # SOC边界约束
        Xekf_pred[2] = np.clip(Xekf_pred[2], 0.01, 1.0)
        Xekf_all[:, i + 1] = Xekf_pred

        # OCV计算
        try:
            Uoc[i + 1] = ocv_interp(Xekf_all[2, i + 1])
        except:
            Uoc[i + 1] = np.polyval(p, Xekf_all[2, i + 1])

        # 数值微分计算dOCV/dSOC
        delta_soc = 0.0005
        soc_current = Xekf_all[2, i + 1]
        soc_plus = min(1.0, soc_current + delta_soc)
        soc_minus = max(0.0, soc_current - delta_soc)

        try:
            ocv_plus = ocv_interp(soc_plus)
            ocv_minus = ocv_interp(soc_minus)
        except:
            ocv_plus = np.polyval(p, soc_plus)
            ocv_minus = np.polyval(p, soc_minus)

        dOCV_dSOC = (ocv_plus - ocv_minus) / (soc_plus - soc_minus)
        H[i, :] = [-1, -1, dOCV_dSOC]

        # 端电压预测
        Vekf[i + 1] = Uoc[i + 1] + C_matrix @ Xekf_all[:, i + 1] - Cur[i + 1] * Pa_ekf[0, current_j]

        # 卡尔曼滤波更新
        P_pred = A @ P0 @ A.T + Q

        # 新息
        innovation = Vot[i + 1] - Vekf[i + 1]
        innovation_sequence.append(innovation)

        # 自适应噪声估计（使用更长的窗口，在开始和结束阶段）
        if i > 150:
            window_size = 150 if i < 300 or i > T - 300 else 100
            R_adaptive = adaptive_noise_estimation(innovation_sequence, window_size)

        # 新息协方差
        S = H[i, :] @ P_pred @ H[i, :].T + R_adaptive

        # 卡尔曼增益
        if abs(S) > 1e-12:
            K[:, i] = P_pred @ H[i, :].T / S
        else:
            K[:, i] = np.zeros(3)

        # 协方差更新 - 修复矩阵运算错误
        I_KH = np.eye(3) - K[:, i:i + 1] @ H[i:i + 1, :]
        # 使用 Joseph 形式的协方差更新，更数值稳定
        P0 = I_KH @ P_pred @ I_KH.T + K[:, i:i + 1] * R_adaptive @ K[:, i:i + 1].T

        # 状态更新
        Xekf_all[:, i + 1] = Xekf_all[:, i + 1] + K[:, i] * innovation

        # SOC边界约束
        Xekf_all[2, i + 1] = np.clip(Xekf_all[2, i + 1], 0.01, 1.0)

        # 参数估计的输出敏感性矩阵
        d_g_PA[i, 0] = -Cur[i + 1]  # dV/dR0
        d_g_PA[i, 1] = -Xekf_all[0, i + 1]  # dV/dR1 (近似)
        d_g_PA[i, 2] = 0  # dV/dC1 (近似为0，因为直接影响很小)
        d_g_PA[i, 3] = -Xekf_all[1, i + 1]  # dV/dR2 (近似)
        d_g_PA[i, 4] = 0  # dV/dC2 (近似为0)

        counter += 1

        # 参数估计更新
        if counter >= update_interval and j < K_pa.shape[1] - 1:
            counter = 0

            # 带遗忘因子的协方差更新
            P0_pa = P0_pa / forgetting_factor + Q_pa

            # 参数更新的新息协方差
            S_pa = d_g_PA[i, :] @ P0_pa @ d_g_PA[i, :].T + R_pa

            if abs(S_pa) > 1e-12:
                # 卡尔曼增益
                K_pa_j = P0_pa @ d_g_PA[i, :].T / S_pa

                # 限制参数更新幅度
                max_update_ratios = [0.05, 0.1, 0.1, 0.2, 0.1]  # 每个参数的最大更新比例
                param_update = K_pa_j * innovation

                for param_idx in range(5):
                    max_change = abs(Pa_ekf[param_idx, current_j]) * max_update_ratios[param_idx]
                    param_update[param_idx] = np.clip(param_update[param_idx], -max_change, max_change)

                # 存储卡尔曼增益
                K_pa[:, j] = K_pa_j

                # 协方差更新 - 修复标量乘法问题
                I_K_dg = np.eye(5) - K_pa_j.reshape(-1, 1) @ d_g_PA[i:i + 1, :]
                P0_pa = I_K_dg @ P0_pa @ I_K_dg.T + K_pa_j.reshape(-1, 1) * R_pa @ K_pa_j.reshape(1, -1)

                # 参数更新
                new_params = Pa_ekf[:, current_j] + param_update
                Pa_ekf = np.column_stack([Pa_ekf, new_params])
                j += 1

        # 更新参数估计结果
        current_j = min(j, Pa_ekf.shape[1] - 1)
        R0_esti[i + 1] = Pa_ekf[0, current_j]
        R1_esti[i + 1] = Pa_ekf[1, current_j]
        C1_esti[i + 1] = Pa_ekf[2, current_j]
        R2_esti[i + 1] = Pa_ekf[3, current_j]
        C2_esti[i + 1] = Pa_ekf[4, current_j]

    print("DEKF计算完成，开始分析结果...")

    # 计算误差统计
    V_error = Vot - Vekf
    SOC_error = RSOC - Xekf_all[2, :]

    # 分段误差分析
    start_phase = slice(0, 300)  # 开始阶段
    middle_phase = slice(300, -300)  # 中间稳定阶段
    end_phase = slice(-300, None)  # 结束阶段

    phases = [("开始阶段", start_phase), ("中间阶段", middle_phase), ("结束阶段", end_phase)]

    print("\n=== 分段误差分析 ===")
    for phase_name, phase_slice in phases:
        if len(V_error[phase_slice]) > 0:
            v_mae = np.mean(np.abs(V_error[phase_slice]))
            v_rmse = np.sqrt(np.mean(V_error[phase_slice] ** 2))
            soc_mae = np.mean(np.abs(SOC_error[phase_slice]))
            soc_rmse = np.sqrt(np.mean(SOC_error[phase_slice] ** 2))
            print(f"{phase_name}:")
            print(f"  电压误差 - MAE: {v_mae:.6f}V, RMSE: {v_rmse:.6f}V")
            print(f"  SOC误差 - MAE: {soc_mae:.6f}, RMSE: {soc_rmse:.6f}")

    # 整体误差统计
    convergence_start = 50
    V_error_mean = np.mean(np.abs(V_error[convergence_start:]))
    V_error_rmse = np.sqrt(np.mean(V_error[convergence_start:] ** 2))
    SOC_error_mean = np.mean(np.abs(SOC_error[convergence_start:]))
    SOC_error_rmse = np.sqrt(np.mean(SOC_error[convergence_start:] ** 2))

    print(f"\n=== 整体误差统计（排除前{convergence_start}个收敛点）===")
    print(f"端电压误差 - 平均: {V_error_mean:.6f}V, RMSE: {V_error_rmse:.6f}V")
    print(f"SOC误差 - 平均: {SOC_error_mean:.6f}, RMSE: {SOC_error_rmse:.6f}")

    # 绘图
    plt.style.use('default')

    # 图1: 主要结果对比
    fig1, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 端电压对比
    axes[0, 0].plot(tm, Vot, 'k-', linewidth=1.5, label='真实值')
    axes[0, 0].plot(tm, Vekf, 'r--', linewidth=1.5, label='估计值')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('端电压 (V)')
    axes[0, 0].set_title('端电压估计结果')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # SOC对比
    axes[0, 1].plot(tm, RSOC, 'k-', linewidth=1.5, label='真实值')
    axes[0, 1].plot(tm, Xekf_all[2, :], 'r--', linewidth=1.5, label='估计值')
    axes[0, 1].set_xlabel('时间 (s)')
    axes[0, 1].set_ylabel('SOC')
    axes[0, 1].set_title('SOC估计结果')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 端电压误差
    axes[1, 0].plot(tm, V_error * 1000, 'b-', linewidth=1)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('时间 (s)')
    axes[1, 0].set_ylabel('端电压误差 (mV)')
    axes[1, 0].set_title(f'端电压误差 (RMSE: {V_error_rmse * 1000:.2f}mV)')
    axes[1, 0].grid(True, alpha=0.3)

    # SOC误差
    axes[1, 1].plot(tm, SOC_error * 100, 'g-', linewidth=1)
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('时间 (s)')
    axes[1, 1].set_ylabel('SOC误差 (%)')
    axes[1, 1].set_title(f'SOC误差 (RMSE: {SOC_error_rmse * 100:.2f}%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 图2: 参数估计结果
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

    param_data = [
        (R0_esti, R0, 'R0', 'Ω'),
        (R1_esti, R1, 'R1', 'Ω'),
        (C1_esti, C1, 'C1', 'F'),
        (R2_esti, R2, 'R2', 'Ω'),
        (C2_esti, C2, 'C2', 'F')
    ]

    for i, (est_values, true_value, param_name, unit) in enumerate(param_data):
        row, col = i // 3, i % 3
        axes[row, col].plot(tm, est_values, 'b-', linewidth=2, label='估计值')
        axes[row, col].axhline(y=true_value, color='r', linestyle='--', linewidth=2, label='真实值')
        axes[row, col].set_xlabel('时间 (s)')
        axes[row, col].set_ylabel(f'{param_name} ({unit})')
        axes[row, col].set_title(f'{param_name}参数估计')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    # 参数估计对比柱状图
    axes[1, 2].clear()
    final_params = [R0_esti[-1], R1_esti[-1], C1_esti[-1], R2_esti[-1], C2_esti[-1]]
    true_params = [R0, R1, C1, R2, C2]
    param_names = ['R0', 'R1', 'C1', 'R2', 'C2']

    x_pos = np.arange(len(param_names))
    width = 0.35

    bars1 = axes[1, 2].bar(x_pos - width / 2, true_params, width, label='真实值', alpha=0.7, color='red')
    bars2 = axes[1, 2].bar(x_pos + width / 2, final_params, width, label='估计值', alpha=0.7, color='blue')

    axes[1, 2].set_xlabel('参数')
    axes[1, 2].set_ylabel('值')
    axes[1, 2].set_title('最终参数估计对比')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(param_names)
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')  # 使用对数刻度
    axes[1, 2].grid(True, alpha=0.3)

    # 在柱状图上添加误差百分比
    for i, (true_val, est_val) in enumerate(zip(true_params, final_params)):
        error_pct = abs(est_val - true_val) / true_val * 100
        axes[1, 2].text(x_pos[i], max(true_val, est_val) * 1.1, f'{error_pct:.1f}%',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # 输出最终结果
    print("\n=== 最终参数估计结果 ===")
    param_results = [
        ("R0", R0, R0_esti[-1]),
        ("R1", R1, R1_esti[-1]),
        ("C1", C1, C1_esti[-1]),
        ("R2", R2, R2_esti[-1]),
        ("C2", C2, C2_esti[-1])
    ]

    for name, true_val, est_val in param_results:
        error_pct = abs(true_val - est_val) / true_val * 100
        print(f"{name}: 真实值={true_val:.6f}, 估计值={est_val:.6f}, 误差={error_pct:.2f}%")

    print(f"\n=== 算法性能总结 ===")
    print(f"端电压RMSE: {V_error_rmse * 1000:.2f} mV")
    print(f"SOC RMSE: {SOC_error_rmse * 100:.2f}%")
    print(f"参数更新次数: {j}")
    print("优化完成！")


if __name__ == "__main__":
    main()