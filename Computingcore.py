# 导入必要的库
import pandas as pd  # 用于Excel文件读取
import numpy as np  # 数值计算和数组处理
import matplotlib.pyplot as plt  # 数据可视化
import heapq  # 优先队列实现（用于A*算法）

# ------------------------ 数据读取与栅格初始化 ------------------------
file_path = 'Mediumdensity.xlsx'  # Excel文件路径
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None)  # 读取Sheet1数据
grid = sheet1.values  # 将DataFrame转换为numpy数组

# 处理起点终点坐标
start_pos = np.argwhere(grid == 'Start')[0]  # 查找"Start"标记的位置
end_pos = np.argwhere(grid == 'End')[0]  # 查找"End"标记的位置
grid[start_pos[0], start_pos[1]] = 0  # 将起点位置值设为0
grid[end_pos[0], end_pos[1]] = 0  # 将终点位置值设为0
grid = grid.astype(int)  # 将网格转换为整数类型


# ------------------------ 通用工具函数 ------------------------
def get_neighbors(pos, rows, cols):
    """获取四邻域有效邻居
    参数:
        pos - 当前位置坐标 (x, y)
        rows - 网格行数
        cols - 网格列数
    返回:
        有效邻居坐标列表
    """
    neighbors = []
    # 四个移动方向：上、下、左、右
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = pos[0] + dx, pos[1] + dy
        # 检查边界有效性
        if 0 <= x < rows and 0 <= y < cols:
            neighbors.append((x, y))
    return neighbors


def reconstruct_path(came_from, current):
    """路径回溯重建
    参数:
        came_from - 记录节点来源的字典
        current - 当前位置
    返回:
        有序路径列表（从起点到终点）
    """
    path = [current]
    # 通过字典回溯找到完整路径
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # 反转得到从起点到终点的路径


# ------------------------ A*算法实现 ------------------------
def a_star(grid, start, end, heuristic_type):
    """通用A*算法框架
    参数:
        grid - 二维网格地图
        start - 起点坐标 (x, y)
        end - 终点坐标 (x, y)
        heuristic_type - 启发函数类型（'manhattan'或'euclidean'）
    返回:
        规划路径列表 或 None（无路径时）
    """
    rows, cols = grid.shape
    open_heap = []  # 优先队列（开放列表）
    heapq.heappush(open_heap, (0, start))  # 初始节点入队

    # 路径记录字典和代价字典初始化
    came_from = {}  # 记录节点父节点
    g_score = {start: 0}  # 实际移动代价（起点到当前节点的代价）
    f_score = {start: 0}  # 总代价（g_score + 启发函数值）

    # 定义启发函数
    def heuristic(a, b):
        """启发函数计算"""
        if heuristic_type == 'manhattan':
            # 曼哈顿距离：水平和垂直移动的绝对值之和
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        elif heuristic_type == 'euclidean':
            # 欧式距离：直线距离
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    while open_heap:
        current = heapq.heappop(open_heap)[1]  # 取出总代价最小的节点

        if current == end:  # 到达终点
            return reconstruct_path(came_from, current)

        # 遍历四个邻居方向
        for neighbor in get_neighbors(current, rows, cols):
            if grid[neighbor[0], neighbor[1]] == 1:  # 障碍物检测（假设1为障碍）
                continue

            # 计算临时移动代价（假设每步代价为1）
            tentative_g = g_score[current] + 1

            # 发现更优路径时更新
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None  # 开放列表为空时表示无路径


# ------------------------ 可视化配置 ------------------------
# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_grid(title, path=None, color='b'):
    """绘制栅格地图及路径
    参数:
        title - 图表标题
        path - 路径坐标列表（可选）
        color - 路径颜色（默认蓝色）
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='Accent', origin='upper')  # 显示网格

    # 绘制起点和终点
    plt.scatter(start[1], start[0], c='red', edgecolor='black',
                s=150, label='起点')
    plt.scatter(end[1], end[0], c='yellow', edgecolor='black',
                s=150, label='终点')

    # 绘制路径（如果有）
    if path:
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0],
                 marker='o', markersize=5,
                 color=color, linewidth=2,
                 linestyle='-', label='规划路径')

    plt.legend(loc='upper right')  # 显示图例
    plt.title(title)  # 设置标题
    plt.show()


# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    # 获取处理后的起点终点坐标
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])

    # 显示原始栅格图
    plot_grid(title="原始栅格地图")

    # 曼哈顿距离路径规划
    path_manhattan = a_star(grid, start, end, 'manhattan')
    plot_grid(title="曼哈顿距离路径规划",
              path=path_manhattan, color='blue')

    # 欧式距离路径规划
    path_euclidean = a_star(grid, start, end, 'euclidean')
    plot_grid(title="欧式距离路径规划",
              path=path_euclidean, color='green')