# 导入必要的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 可视化
import heapq  # 优先队列
from math import factorial  # 阶乘计算（用于贝塞尔曲线）

# ------------------------ 数据初始化 ------------------------
file_path = 'Highdensity.xlsx'
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
grid = sheet1.values  # 转换为numpy数组

# 处理起终点坐标
start_pos = np.argwhere(grid == 'Start')[0]  # 查找起点坐标
end_pos = np.argwhere(grid == 'End')[0]  # 查找终点坐标
grid[start_pos[0], start_pos[1]] = 0  # 重置起点值
grid[end_pos[0], end_pos[1]] = 0  # 重置终点值
grid = grid.astype(int)  # 转换为整数矩阵


# ------------------------ 工具函数 ------------------------
def get_neighbors(pos, rows, cols):
    """获取四邻域有效邻居坐标
    参数:
        pos - 当前坐标(x,y)
        rows - 网格总行数
        cols - 网格总列数
    返回:
        有效邻居坐标列表"""
    neighbors = []
    # 四方向偏移量：上、下、左、右
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = pos[0] + dx, pos[1] + dy
        if 0 <= x < rows and 0 <= y < cols:  # 边界检查
            neighbors.append((x, y))
    return neighbors


def reconstruct_path(came_from, current):
    """路径回溯重建
    参数:
        came_from - 父节点映射字典
        current - 终点坐标
    返回:
        从起点到终点的有序路径列表"""
    path = [current]
    while current in came_from:  # 反向追踪父节点
        current = came_from[current]
        path.append(current)
    return path[::-1]  # 反转得到正向路径


# ------------------------ A*算法核心 ------------------------
def a_star(grid, start, end):
    """A*路径规划算法（曼哈顿距离）
    参数:
        grid - 二维网格地图
        start - 起点坐标
        end - 终点坐标
    返回:
        规划路径 或 None"""
    rows, cols = grid.shape
    open_heap = []  # 优先队列（开放列表）
    heapq.heappush(open_heap, (0, start))  # 初始化队列

    # 路径记录和代价字典
    came_from = {}  # 父节点映射
    g_score = {start: 0}  # 实际移动代价
    f_score = {start: 0}  # 总评估代价

    def heuristic(a, b):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_heap:
        current = heapq.heappop(open_heap)[1]  # 取出最小代价节点

        if current == end:  # 到达终点
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, rows, cols):
            if grid[neighbor[0], neighbor[1]] == 1:  # 障碍物检测
                continue

            # 计算临时移动代价（假设每步代价为1）
            tentative_g = g_score[current] + 1

            # 更新更优路径
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None  # 无可用路径


# ------------------------ 贝塞尔曲线优化 ------------------------
def comb(n, k):
    """组合数计算（用于多项式）
    参数:
        n - 总点数
        k - 选取点数
    返回:
        组合数值"""
    return factorial(n) // (factorial(k) * factorial(n - k))


def bezier_curve(points, num=100):
    """生成贝塞尔曲线
    参数:
        points - 控制点列表（A*路径点）
        num - 曲线采样点数
    返回:
        平滑后的路径坐标数组"""
    n = len(points) - 1
    if n < 1:  # 处理单点或空点情况
        return points

    t = np.linspace(0, 1, num)  # 参数t取值范围
    curve = []
    for ti in t:
        x, y = 0, 0
        # 计算各控制点的函数加权和
        for i, (px, py) in enumerate(points):
            bern = comb(n, i) * (ti ** i) * (1 - ti) ** (n - i)
            x += px * bern
            y += py * bern
        curve.append((x, y))
    return np.array(curve)  # 转换为numpy数组


# ------------------------ 可视化对比 ------------------------
def plot_compare(original_path, bezier_path):
    """路径对比可视化
    参数:
        original_path - 原始A*路径
        bezier_path - 贝塞尔优化路径"""
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='Accent', origin='upper')  # 显示栅格地图

    # 绘制起终点
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])
    plt.scatter(start[1], start[0], c='red', edgecolor='black',
                s=150, label='Start')
    plt.scatter(end[1], end[0], c='yellow', edgecolor='black',
                s=150, label='End')

    # 绘制原始路径
    path_array = np.array(original_path)
    plt.plot(path_array[:, 1], path_array[:, 0], 'b-o',
             markersize=5, linewidth=2, label='Original Path')

    # 绘制贝塞尔路径
    bezier_array = bezier_path
    plt.plot(bezier_array[:, 1], bezier_array[:, 0], 'r--',
             linewidth=2, label='Optimized Path')

    plt.legend()
    plt.title("Path Planning: A* vs Bezier Optimization")
    plt.show()


# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])

    path = a_star(grid, start, end)
    if path:
        # 生成贝塞尔曲线优化路径
        bezier_path = bezier_curve(path)
        # 可视化对比
        plot_compare(path, bezier_path)
    else:
        print("No valid path found!")