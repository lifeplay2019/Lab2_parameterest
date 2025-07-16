from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

mat_path = "../data/Lab2_data/Example_data/discharge.mat"
data = loadmat(mat_path)

# 查看加载的 mat 文件中的 keys，确认数据所在 key
print(data.keys())

# 例如假设数据在 'measurement' 字段中
if 'measurement' in data:
    matrix = data['measurement']

    if isinstance(matrix, np.ndarray):
        # 确保 matrix 有至少 4 行
        if matrix.ndim == 2 and matrix.shape[0] >= 4:
            x = matrix[0, :]  # 第一行作为 x 轴
            y = matrix[2, :]  # 第三行作为 y 轴

            # 绘图
            plt.plot(x, y)
            plt.xlabel('X轴标签')  # 可以替换为更有意义的标签
            plt.ylabel('Y轴标签')  # 可以替换为更有意义的标签
            plt.title('二维数据绘图')
            plt.grid(True)

            # 设置 x 轴标签格式，使科学计数法更易读
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

            plt.show()
        else:
            print("数据不是期望的二维数组或者行数不足")
    else:
        print("数据不是 NumPy 数组")
else:
    print("在 mat 文件中未找到 'measurement' 字段")