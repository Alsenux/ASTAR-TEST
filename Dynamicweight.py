# 导入必要的库
import pandas as pd  # 数据处理
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 可视化
import heapq  # 优先队列

# ------------------------ 数据初始化 ------------------------
file_path = 'Mediumdensity.xlsx'  # Excel文件路径
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None)  # 读取数据
grid = sheet1.values  # 转换为numpy数组

# 处理起点终点坐标
start_pos = np.argwhere(grid == 'Start')[0]  # 查找起点坐标（返回第一个匹配位置）
end_pos = np.argwhere(grid == 'End')[0]  # 查找终点坐标
grid[start_pos[0], start_pos[1]] = 0  # 重置起点位置值为0
grid[end_pos[0], end_pos[1]] = 0  # 重置终点位置值为0
grid = grid.astype(int)  # 转换为整数类型矩阵


# ------------------------ 工具函数 ------------------------
def get_neighbors(pos, rows, cols):
    """获取四邻域有效邻居坐标
    参数:
        pos - 当前坐标元组 (x, y)
        rows - 网格总行数
        cols - 网格总列数
    返回:
        有效邻居坐标列表"""
    neighbors = []
    # 四方向偏移量：上、下、左、右
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = pos[0] + dx, pos[1] + dy
        # 边界有效性检查
        if 0 <= x < rows and 0 <= y < cols:
            neighbors.append((x, y))
    return neighbors


def reconstruct_path(came_from, current):
    """路径回溯重建
    参数:
        came_from - 父节点映射字典
        current - 当前节点坐标
    返回:
        从起点到终点的有序路径列表"""
    path = [current]
    # 通过字典反向追踪路径
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # 反转得到正向路径


# ------------------------ 改进的A*算法 ------------------------
def weighted_a_star(grid, start, end, beta=1):
    """带权重的A*路径规划算法
    参数:
        grid - 二维网格地图（0可通过，1为障碍）
        start - 起点坐标元组 (x, y)
        end - 终点坐标元组 (x, y)
        beta - 启发式权重系数（默认1）
    返回:
        规划路径列表 或 None"""
    rows, cols = grid.shape
    open_heap = []  # 优先队列（开放列表）
    heapq.heappush(open_heap, (0, start))  # 初始节点入队

    # 路径记录和代价字典
    came_from = {}  # 记录每个节点的父节点
    g_score = {start: 0}  # 实际移动代价（起点到当前节点的累计代价）
    f_score = {start: 0}  # 总评估代价（g + β*h）

    def heuristic(a, b):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_heap:
        # 取出当前最小代价节点
        current = heapq.heappop(open_heap)[1]

        if current == end:  # 到达终点
            return reconstruct_path(came_from, current)

        # 遍历四邻域
        for neighbor in get_neighbors(current, rows, cols):
            if grid[neighbor[0], neighbor[1]] == 1:  # 障碍物检测
                continue

            # 计算临时移动代价（假设每步代价为1）
            tentative_g = g_score[current] + 1

            # 发现更优路径时更新
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # 关键修改：引入启发式权重系数β
                f_score[neighbor] = tentative_g + beta * heuristic(neighbor, end)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None  # 无可用路径


# ------------------------ 可视化对比 ------------------------
def plot_compare(path_original, path_weighted):
    """路径对比可视化
    参数:
        path_original - 原始A*路径（β=1）
        path_weighted - 加权A*路径（β=2）"""
    plt.figure(figsize=(12, 6))  # 创建画布

    # 原始路径子图
    plt.subplot(1, 2, 1)
    plt.imshow(grid, cmap='Accent', origin='upper')  # 显示网格地图
    # 绘制起终点
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])
    plt.scatter(start[1], start[0], c='red', s=100, label='Start')
    plt.scatter(end[1], end[0], c='yellow', s=100, label='End')
    # 绘制原始路径（注意坐标转换：矩阵行->y轴，列->x轴）
    path_array = np.array(path_original)
    plt.plot(path_array[:, 1], path_array[:, 0], 'b-o',
             linewidth=2, markersize=4)
    plt.title(f"Original A* (β=1)\nPath Length: {len(path_original)}")
    plt.legend()

    # 加权路径子图
    plt.subplot(1, 2, 2)
    plt.imshow(grid, cmap='Accent', origin='upper')
    plt.scatter(start[1], start[0], c='red', s=100)
    plt.scatter(end[1], end[0], c='yellow', s=100)
    path_weighted_array = np.array(path_weighted)
    plt.plot(path_weighted_array[:, 1], path_weighted_array[:, 0], 'r--o',
             linewidth=2, markersize=4)
    plt.title(f"Weighted A* (β=2)\nPath Length: {len(path_weighted)}")

    plt.tight_layout()  # 自动调整子图间距
    plt.show()


# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    # 获取处理后的坐标元组
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])

    # 计算两种路径
    path_original = weighted_a_star(grid, start, end, beta=1)  # 标准A*
    path_weighted = weighted_a_star(grid, start, end, beta=2)  # 加权A*

    if path_original and path_weighted:
        plot_compare(path_original, path_weighted)
    else:
        print("Path not found!")  # 路径查找失败处理
