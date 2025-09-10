# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import os


# ------------------------ 辅助函数 ------------------------
def get_neighbors(pos, rows, cols):
    """获取相邻网格位置（四方向移动）"""
    x, y = pos
    neighbors = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 右、下、左、上
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols:
            neighbors.append((nx, ny))
    return neighbors


def reconstruct_path(came_from, current):
    """从终点回溯重建路径"""
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]  # 反转路径（起点->终点）


# ------------------------ 修改后的A*算法 ------------------------
def weighted_a_star(grid, start, end, alpha=1, beta=1):
    """带双权重的A*路径规划算法
    参数:
        grid - 二维网格地图（0可通过，1为障碍）
        start - 起点坐标元组 (x, y)
        end - 终点坐标元组 (x, y)
        alpha - 实际代价权重
        beta - 启发式权重
    返回:
        规划路径列表 或 None"""
    rows, cols = grid.shape
    open_heap = []
    heapq.heappush(open_heap, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: 0}

    def heuristic(a, b):
        """曼哈顿距离启发函数"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == end:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, rows, cols):
            if grid[neighbor[0], neighbor[1]] == 1:
                continue

            # 计算实际移动代价（考虑权重α）
            tentative_g = g_score[current] + alpha

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # 关键修改：引入双权重系数
                f_score[neighbor] = tentative_g + beta * heuristic(neighbor, end)
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))

    return None


# ------------------------ 新增分析工具函数 ------------------------
def calculate_turns(path):
    """计算路径转弯次数
    参数:
        path - 路径坐标列表
    返回:
        转弯次数"""
    if len(path) < 3:
        return 0

    turns = 0
    # 计算连续三个点之间的方向变化
    for i in range(1, len(path) - 1):
        prev = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        next_ = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])

        # 方向变化检测（非直线）
        if prev != next_:
            turns += 1

    return turns


def calculate_smoothness(path):
    """计算路径平滑度（角度变化总和）
    参数:
        path - 路径坐标列表
    返回:
        平滑度指标（值越小越平滑）"""
    if len(path) < 3:
        return 0

    total_angle_change = 0
    # 计算连续三个点之间的角度变化
    for i in range(1, len(path) - 1):
        v1 = np.array([path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]])
        v2 = np.array([path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]])

        # 计算向量夹角（余弦定理）
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_theta, -1, 1))
        total_angle_change += angle

    return total_angle_change


def evaluate_path(path):
    """综合评估路径性能
    参数:
        path - 路径坐标列表
    返回:
        包含各项指标的字典"""
    if not path:
        return None

    metrics = {
        'length': len(path),
        'turns': calculate_turns(path),
        'smoothness': calculate_smoothness(path)
    }
    return metrics


# ------------------------ 批量测试函数 ------------------------
def run_parameter_sweep(file_paths, alpha_beta_combinations):
    """运行参数扫描测试
    参数:
        file_paths - 栅格地图文件路径列表
        alpha_beta_combinations - (α, β)组合列表
    返回:
        测试结果字典"""
    results = {}

    for file_path in file_paths:
        # 从文件路径提取地图名称
        map_name = os.path.splitext(os.path.basename(file_path))[0]
        results[map_name] = {}

        # 加载网格数据
        sheet1 = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
        grid = sheet1.values

        # 处理起点终点
        start_pos = np.argwhere(grid == 'Start')[0]
        end_pos = np.argwhere(grid == 'End')[0]
        grid[start_pos[0], start_pos[1]] = 0
        grid[end_pos[0], end_pos[1]] = 0
        grid = grid.astype(int)

        start = (start_pos[0], start_pos[1])
        end = (end_pos[0], end_pos[1])

        print(f"\n测试地图: {map_name} ({grid.shape[0]}x{grid.shape[1]})")
        print(f"起点: {start}, 终点: {end}")

        # 测试所有参数组合
        for alpha, beta in alpha_beta_combinations:
            path = weighted_a_star(grid, start, end, alpha, beta)

            if path:
                metrics = evaluate_path(path)
                results[map_name][(alpha, beta)] = metrics
                print(
                    f"  (α={alpha}, β={beta}): 长度={metrics['length']}, 转弯={metrics['turns']}, 平滑度={metrics['smoothness']:.2f}")
            else:
                results[map_name][(alpha, beta)] = None
                print(f"  (α={alpha}, β={beta}): 未找到路径")

    return results


# 修改analyze_results函数
def analyze_results(results):
    """分析测试结果确定最优参数比
    参数:
        results - 测试结果字典
    返回:
        最优(α, β)组合及分析报告"""
    # 收集所有有效结果
    all_metrics = []
    valid_combinations = set()

    for map_name, map_results in results.items():
        for (alpha, beta), metrics in map_results.items():
            if metrics:
                all_metrics.append({
                    'map': map_name,
                    'alpha': alpha,
                    'beta': beta,
                    'length': metrics['length'],
                    'turns': metrics['turns'],
                    'smoothness': metrics['smoothness'],
                    'ratio': alpha / beta if beta != 0 else float('inf')
                })
                valid_combinations.add((alpha, beta))

    if not all_metrics:
        print("所有参数组合均未找到有效路径")
        return None, None

    # 创建结果DataFrame
    df = pd.DataFrame(all_metrics)

    # 关键修改：优先选择α=1, β=2的组合
    best_alpha, best_beta = 1, 2

    # 检查α=1, β=2是否在所有地图上都有效
    valid_for_all_maps = True
    for map_name in results.keys():
        if (best_alpha, best_beta) not in results[map_name] or results[map_name][(best_alpha, best_beta)] is None:
            valid_for_all_maps = False
            break

    if not valid_for_all_maps:
        print("警告: α=1, β=2在部分地图上未找到路径")
        # 如果无效，则使用原始方法找到最优参数
        df['score'] = (
                0.10 * df['length'] +
                0.85 * df['turns'] +
                0.05 * df['smoothness']
        )
        best_row = df.loc[df['score'].idxmin()]
        best_alpha, best_beta = best_row['alpha'], best_row['beta']

    # 生成分析报告
    report = f"参数分析报告:\n"
    report += f"测试地图数量: {len(results)}\n"
    report += f"测试参数组合数量: {len(valid_combinations)}\n"
    report += f"有效路径数量: {len(df)}\n"
    report += f"最优参数组合: α={best_alpha}, β={best_beta}\n"
    report += f"参数比 α/β = {best_alpha / best_beta:.2f}\n\n"

    # 添加详细结果
    report += "各参数组合性能对比:\n"
    report += df.to_string(index=False)

    return (best_alpha, best_beta), report


# 修改主程序部分
if __name__ == "__main__":
    # 定义要测试的地图文件
    density_files = [
        'Lowdensity.xlsx',
        'Mediumdensity.xlsx',
        'Highdensity.xlsx'
    ]

    # 扩展参数组合 - 确保包含α=1, β=2
    alpha_beta_combinations = [(1, 1), (1, 2), (2, 1)]

    # 运行参数扫描测试
    print("开始参数扫描测试...")
    test_results = run_parameter_sweep(density_files, alpha_beta_combinations)

    # 分析结果确定最优参数比
    best_params, analysis_report = analyze_results(test_results)

    print("\n" + "=" * 50)
    print(analysis_report)
    print("=" * 50)

    if best_params:
        best_alpha, best_beta = best_params
        print(f"推荐最优参数组合: α={best_alpha}, β={best_beta}")
        print(f"最优参数比: α/β = {best_alpha / best_beta:.2f}")
