import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import curve_fit

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


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
        # 原始路径
        ("data/Lab2_data/Example_data/discharge_step2_p.mat", "data/Lab2_data/Example_data/OCV_SOC_step2.mat"),
        # 当前目录
        ("discharge_step2.mat", "OCV_SOC_step2.mat"),
        # 上级目录
        ("../discharge_step2.mat", "../OCV_SOC_step2.mat"),
        # data目录
        ("data/discharge_step2.mat", "data/OCV_SOC_step2.mat"),
        # 其他可能的路径
        ("Lab2_data/Example_data/discharge_step2.mat", "Lab2_data/Example_data/OCV_SOC_step2.mat"),
        ("Example_data/discharge_step2.mat", "Example_data/OCV_SOC_step2.mat"),
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


def main():
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
    Xekf = np.array([0, 0, 0.9]).reshape(-1, 1)  # [U1,U2,SOC]初始值
    Q = 0.00000001 * np.eye(3)  # 系统误差协方差
    R = 1  # 测量误差协方差
    P0 = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]])  # 状态误差协方差初始值

    # 参数估计初始值
    Pa_ekf = np.array([R0, R1, C1, R2, C2]).reshape(-1, 1)  # 初始值
    Q_diagonal = [0.0000003, 0.0000003, 1000000000, 0.0000003, 1000000000]  # 参数估计中系统噪声方差
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

    # DEKF主循环
    print("开始DEKF计算...")
    for i in range(T):
        if i % 500 == 0:
            print(f"处理进度: {i}/{T}")

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

    print("DEKF计算完成，开始绘图...")

    # 绘图部分
    t = tm

    # 图1: 端电压对比
    plt.figure(1, figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t, Vot, '-k', t, Vekf, '-r', linewidth=2)
    plt.grid(True)
    plt.legend(['真实值', '估计值'])
    plt.ylabel('端电压（V）')
    plt.xlabel('时间(s)')
    plt.title('端电压估计结果')

    # 图2: SOC对比
    plt.subplot(2, 2, 2)
    plt.plot(t, RSOC, '-k', t, Xekf_all[2, :], '-r', linewidth=2)
    plt.grid(True)
    plt.legend(['真实值', '估计值'])
    plt.ylabel('SOC')
    plt.xlabel('时间(s)')
    plt.title('SOC估计结果')

    # 计算误差
    V_error = Vot - Vekf
    SOC_error = RSOC - Xekf_all[2, :]

    # 避免索引超出范围
    error_start_idx = min(5000, len(SOC_error) // 2)
    if error_start_idx < len(SOC_error):
        SOC_error_mean = np.mean(np.abs(SOC_error[error_start_idx:]))
        SOC_error_max = np.max(np.abs(SOC_error[error_start_idx:]))
    else:
        SOC_error_mean = np.mean(np.abs(SOC_error))
        SOC_error_max = np.max(np.abs(SOC_error))

    print(f"SOC估计平均误差: {SOC_error_mean:.6f}")
    print(f"SOC估计最大误差: {SOC_error_max:.6f}")

    # 图3: 端电压误差
    plt.subplot(2, 2, 3)
    plt.plot(t, V_error, '-k', linewidth=2)
    plt.grid(True)
    plt.ylabel('端电压误差(V)')
    plt.xlabel('时间(s)')
    plt.title('端电压估计误差')

    # 图4: SOC误差
    plt.subplot(2, 2, 4)
    plt.plot(t, SOC_error, '-k', linewidth=2)
    plt.grid(True)
    plt.ylabel('SOC误差')
    plt.xlabel('时间(s)')
    plt.title('SOC估计误差')

    plt.tight_layout()

    # 参数估计结果图
    plt.figure(2, figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(t, R0_esti, '-b', linewidth=2)
    plt.axhline(y=R0, color='r', linestyle='--', label='真实值')
    plt.grid(True)
    plt.ylabel('R0 (Ω)')
    plt.xlabel('时间(s)')
    plt.title('R0参数估计')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(t, R1_esti, '-b', linewidth=2)
    plt.axhline(y=R1, color='r', linestyle='--', label='真实值')
    plt.grid(True)
    plt.ylabel('R1 (Ω)')
    plt.xlabel('时间(s)')
    plt.title('R1参数估计')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(t, C1_esti, '-b', linewidth=2)
    plt.axhline(y=C1, color='r', linestyle='--', label='真实值')
    plt.grid(True)
    plt.ylabel('C1 (F)')
    plt.xlabel('时间(s)')
    plt.title('C1参数估计')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(t, R2_esti, '-b', linewidth=2)
    plt.axhline(y=R2, color='r', linestyle='--', label='真实值')
    plt.grid(True)
    plt.ylabel('R2 (Ω)')
    plt.xlabel('时间(s)')
    plt.title('R2参数估计')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(t, C2_esti, '-b', linewidth=2)
    plt.axhline(y=C2, color='r', linestyle='--', label='真实值')
    plt.grid(True)
    plt.ylabel('C2 (F)')
    plt.xlabel('时间(s)')
    plt.title('C2参数估计')
    plt.legend()

    # 显示最终参数估计值
    plt.subplot(2, 3, 6)
    final_params = [R0_esti[-1], R1_esti[-1], C1_esti[-1], R2_esti[-1], C2_esti[-1]]
    true_params = [R0, R1, C1, R2, C2]
    param_names = ['R0', 'R1', 'C1', 'R2', 'C2']

    x_pos = np.arange(len(param_names))
    plt.bar(x_pos - 0.2, true_params, 0.4, label='真实值', alpha=0.7)
    plt.bar(x_pos + 0.2, final_params, 0.4, label='估计值', alpha=0.7)
    plt.xlabel('参数')
    plt.ylabel('值')
    plt.title('参数估计对比')
    plt.xticks(x_pos, param_names)
    plt.legend()
    plt.yscale('log')  # 使用对数刻度以便更好地显示不同量级的参数

    plt.tight_layout()
    plt.show()

    # 输出最终结果
    print("\n=== 最终结果 ===")
    print(f"R0: 真实值={R0:.6f}, 估计值={R0_esti[-1]:.6f}, 误差={abs(R0 - R0_esti[-1]) / R0 * 100:.2f}%")
    print(f"R1: 真实值={R1:.6f}, 估计值={R1_esti[-1]:.6f}, 误差={abs(R1 - R1_esti[-1]) / R1 * 100:.2f}%")
    print(f"C1: 真实值={C1:.1f}, 估计值={C1_esti[-1]:.1f}, 误差={abs(C1 - C1_esti[-1]) / C1 * 100:.2f}%")
    print(f"R2: 真实值={R2:.6f}, 估计值={R2_esti[-1]:.6f}, 误差={abs(R2 - R2_esti[-1]) / R2 * 100:.2f}%")
    print(f"C2: 真实值={C2:.0f}, 估计值={C2_esti[-1]:.0f}, 误差={abs(C2 - C2_esti[-1]) / C2 * 100:.2f}%")
    print("绘图完成！")


if __name__ == "__main__":
    main()