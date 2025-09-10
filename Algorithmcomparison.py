# 导入必要的库
import pandas as pd  # 数据处理和Excel文件读取
import numpy as np  # 数值计算和数组操作
import matplotlib.pyplot as plt  # 数据可视化
import heapq  # 优先队列实现（用于A*算法）
from collections import deque  # 双端队列实现（用于BFS算法）

# 设置中文字体显示（包含多个备选字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常问题

# ------------------------ 数据初始化 ------------------------
file_path = 'Highdensity.xlsx'  # Excel文件路径
sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None)  # 读取指定工作表
grid = sheet1.values  # 将DataFrame转换为NumPy数组

# 处理起点终点坐标
start_pos = np.argwhere(grid == 'Start')[0]  # 查找"Start"标记的第一个出现位置
end_pos = np.argwhere(grid == 'End')[0]  # 查找"End"标记的第一个出现位置
grid[start_pos[0], start_pos[1]] = 0  # 将起点位置值设为0（可通行）
grid[end_pos[0], end_pos[1]] = 0  # 将终点位置值设为0（可通行）
grid = grid.astype(int)  # 将网格数据转换为整数类型（确保数据类型一致性）


# ------------------------ 通用工具函数 ------------------------
def get_neighbors(pos, rows, cols):
    """获取四邻域有效邻居坐标
    参数:
        pos: 当前位置元组 (x, y)
        rows: 网格总行数
        cols: 网格总列数
    返回:
        有效邻居坐标列表（排除边界外位置）"""
    neighbors = []
    # 定义四个移动方向：上、下、左、右
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x, y = pos[0] + dx, pos[1] + dy
        # 检查新坐标是否在有效范围内
        if 0 <= x < rows and 0 <= y < cols:
            neighbors.append((x, y))
    return neighbors


def reconstruct_path(came_from, current):
    """路径回溯重建函数
    参数:
        came_from: 记录每个节点父节点的字典
        current: 当前节点坐标（终点）
    返回:
        从起点到终点的有序路径列表"""
    path = [current]
    # 通过字典反向追踪父节点直到起点
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # 反转列表得到从起点到终点的顺序


# ------------------------ 路径规划算法实现 ------------------------

def a_star(grid, start, end):
    """A*路径规划算法（曼哈顿距离启发式）
    参数:
        grid: 二维网格地图（0可通行，1为障碍）
        start: 起点坐标元组 (x, y)
        end: 终点坐标元组 (x, y)
    返回:
        规划路径列表 或 None（无路径时）"""
    rows, cols = grid.shape
    open_heap = []  # 优先队列（开放列表）
    heapq.heappush(open_heap, (0, start))  # 初始节点入队（优先级，坐标）

    came_from = {}  # 记录父节点关系的字典
    g_score = {start: 0}  # 从起点到各节点的实际移动代价

    while open_heap:
        current = heapq.heappop(open_heap)[1]  # 取出优先级最高的节点

        if current == end:  # 到达终点
            return reconstruct_path(came_from, current)

        # 探索当前节点的四个邻域
        for neighbor in get_neighbors(current, rows, cols):
            if grid[neighbor[0], neighbor[1]] == 1:  # 障碍物检测
                continue

            # 计算临时移动代价（假设每步代价为1）
            tentative_g = g_score[current] + 1

            # 发现更优路径或新节点时更新
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # 计算曼哈顿距离启发值
                h_score = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                f_score = tentative_g + h_score  # A*总评估值
                heapq.heappush(open_heap, (f_score, neighbor))

    return None  # 开放列表为空时表示无可用路径


def bfs(grid, start, end):
    """广度优先搜索算法（BFS）
    参数:
        grid: 二维网格地图
        start: 起点坐标
        end: 终点坐标
    返回:
        最短路径列表 或 None"""
    rows, cols = grid.shape
    queue = deque()  # 使用双端队列实现FIFO
    queue.append(start)  # 初始化队列
    came_from = {}  # 父节点记录字典
    visited = set()  # 已访问节点集合
    visited.add(start)  # 标记起点已访问

    while queue:
        current = queue.popleft()  # 取出队列首元素

        if current == end:  # 找到终点
            return reconstruct_path(came_from, current)

        # 遍历当前节点的所有邻域
        for neighbor in get_neighbors(current, rows, cols):
            # 跳过障碍物和已访问节点
            if grid[neighbor[0], neighbor[1]] == 1 or neighbor in visited:
                continue

            visited.add(neighbor)  # 标记为已访问
            came_from[neighbor] = current  # 记录父节点
            queue.append(neighbor)  # 加入队列尾部

    return None  # 队列排空未找到路径


def dfs(grid, start, end):
    """深度优先搜索算法（DFS）
    参数:
        grid: 二维网格地图
        start: 起点坐标
        end: 终点坐标
    返回:
        找到的路径列表 或 None"""
    rows, cols = grid.shape
    stack = [start]  # 使用列表模拟栈（LIFO）
    came_from = {}  # 父节点记录字典
    visited = set()  # 已访问节点集合
    visited.add(start)  # 标记起点已访问

    while stack:
        current = stack.pop()  # 弹出栈顶元素

        if current == end:  # 找到终点
            return reconstruct_path(came_from, current)

        # 反转邻域顺序保证搜索方向一致性（与原顺序相同）
        for neighbor in reversed(get_neighbors(current, rows, cols)):
            if grid[neighbor[0], neighbor[1]] == 1 or neighbor in visited:
                continue

            visited.add(neighbor)  # 标记已访问
            came_from[neighbor] = current  # 记录父节点
            stack.append(neighbor)  # 压入栈顶

    return None  # 栈排空未找到路径


# ------------------------ 可视化函数 ------------------------
def plot_single_algorithm(grid, path, title, color='blue'):
    """单个算法结果可视化函数
    参数:
        grid: 二维网格地图
        path: 路径坐标列表
        title: 图表标题
        color: 路径颜色（默认蓝色）"""
    plt.figure(figsize=(6, 6))
    # 绘制网格地图（注意坐标系方向设置）
    plt.imshow(grid, cmap='Accent', origin='upper')

    # 绘制起点终点（注意坐标转换：矩阵行->y轴，列->x轴）
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])
    plt.scatter(start[1], start[0], c='red', edgecolor='black',
                s=150, label='起点')
    plt.scatter(end[1], end[0], c='yellow', edgecolor='black',
                s=150, label='终点')

    # 绘制路径（如果存在）
    if path:
        path_array = np.array(path)
        plt.plot(path_array[:, 1], path_array[:, 0], marker='o', markersize=3,
                 color=color, linewidth=1.5, linestyle='-', label='路径')

    plt.legend()  # 显示图例
    plt.title(title)  # 设置标题
    plt.show()


# ------------------------ 主程序 ------------------------
if __name__ == "__main__":
    # 获取处理后的坐标元组
    start = (start_pos[0], start_pos[1])
    end = (end_pos[0], end_pos[1])

    # 计算各算法路径
    path_a_star = a_star(grid, start, end)  # A*算法
    path_bfs = bfs(grid, start, end)  # 广度优先搜索
    path_dfs = dfs(grid, start, end)  # 深度优先搜索

    # 分别绘制三个独立图表进行对比
    plot_single_algorithm(grid, path_a_star, "A*算法（曼哈顿距离）", 'blue')
    plot_single_algorithm(grid, path_bfs, "广度优先搜索（BFS）", 'green')
    plot_single_algorithm(grid, path_dfs, "深度优先搜索（DFS）", 'purple')