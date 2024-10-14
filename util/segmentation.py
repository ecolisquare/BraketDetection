from shapely.geometry import LineString, Point, Polygon
import numpy as np
import matplotlib.pyplot as plt

# 定义函数，使用膨胀来处理噪声容差
def find_intersection_with_tolerance(line, other, tolerance=0.01):
    line_buffer = line.buffer(tolerance)
    other_buffer = other.buffer(tolerance)
    return line_buffer.intersection(other_buffer)

# 追踪路径并寻找闭合形状的辅助函数
def trace_path(start, visited_edges, lines_arcs_dict, tolerance):
    path = [start]
    current = start
    while True:
        found_next = False
        for neighbor in lines_arcs_dict[current]:
            edge = (current, neighbor)
            reverse_edge = (neighbor, current)
            if edge not in visited_edges and reverse_edge not in visited_edges:
                # 标记边的访问方向
                path.append(neighbor)
                visited_edges.add(edge)  # 记录边的方向
                current = neighbor
                found_next = True
                break
        if not found_next or current == start:  # 如果没有下一个点，或者回到起点
            break
    return path if current == start else None  # 仅返回闭合路径

# 获取图中所有的闭合图形
def find_closed_shapes(lines, arcs, tolerance=0.01):
    closed_shapes = []
    lines_arcs = lines + arcs  # 合并所有线段和圆弧
    lines_arcs_dict = {}  # 用于存储每个端点的连接关系
    
    # Step 1: 构建连接图，存储每个线段和圆弧的交点
    for i, line in enumerate(lines_arcs):
        for j, other_line in enumerate(lines_arcs[i+1:], i+1):
            intersection = find_intersection_with_tolerance(line, other_line, tolerance)
            if not intersection.is_empty:
                if line not in lines_arcs_dict:
                    lines_arcs_dict[line] = []
                if other_line not in lines_arcs_dict:
                    lines_arcs_dict[other_line] = []
                lines_arcs_dict[line].append(other_line)
                lines_arcs_dict[other_line].append(line)

    # Step 2: 处理共享边的情况，确保每条边仅使用一次
    visited_edges = set()  # 追踪已经使用过的边，带有方向
    for start in lines_arcs_dict.keys():
        while True:
            path = trace_path(start, visited_edges, lines_arcs_dict, tolerance)
            if path:
                # 创建闭合的多边形
                polygon_coords = []
                for segment in path:
                    polygon_coords.extend(list(segment.coords))  # 添加线段或圆弧的坐标
                closed_shapes.append(Polygon(polygon_coords))
            else:
                break

    return closed_shapes

# 测试数据
def create_test_data():
    # 创建几条线段构成一个矩形，并共享一条边
    line1 = LineString([(0, 0), (2, 0)])
    line2 = LineString([(2, 0), (2, 2)])
    line3 = LineString([(2, 2), (0, 2)])
    line4 = LineString([(0, 2), (0, 0)])

    # 第二个矩形，共用 line3 的上边
    line5 = LineString([(0, 2), (2, 2)])  # 共享线
    line6 = LineString([(2, 2), (2, 4)])
    line7 = LineString([(2, 4), (0, 4)])
    line8 = LineString([(0, 4), (0, 2)])

    return [line1, line2, line3, line4, line5, line6, line7, line8], []

# 可视化结果
def plot_shapes(lines, arcs, closed_shapes):
    fig, ax = plt.subplots()

    # 绘制线段
    for line in lines:
        x, y = line.xy
        ax.plot(x, y, color='blue')

    # 绘制圆弧
    for arc in arcs:
        x, y = arc.xy
        ax.plot(x, y, color='green')

    # 绘制闭合图形
    for shape in closed_shapes:
        x, y = shape.exterior.xy
        ax.fill(x, y, color='lightblue', alpha=0.5)

    plt.savefig('./ex.png')

# 主程序
if __name__ == "__main__":
    lines, arcs = create_test_data()
    tolerance = 0.05  # 设置允许的误差范围
    closed_shapes = find_closed_shapes(lines, arcs, tolerance)
    plot_shapes(lines, arcs, closed_shapes)
