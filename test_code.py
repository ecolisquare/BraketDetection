

# from bracket_parameter_extraction import *
# # Example usage
# if __name__ == "__main__":
#     labels = [
#         # (" 45 ", "no annotation line", False), (" text 120", "no annotation line", False), ("120%DH ", "top", False), ("other 45DH", "bottom", False),
#         # ("  B100X10CH ", "top", False), ("  FB120X10   ", "bottom", False), ("FL150", "bottom", False),
#         # (" BK01 extra ", "top", False), ("R300", "no annotation line", False), ("~DH120", "top", False),
#         # ("FB150X10", "bottom", True), ("FL200", "bottom", False),
#         # (" FB150X12 ~ DH ", "bottom", False),
#         # ("   AC ", "bottom", False)
#         ("20.0X185.5", "top", False),("$~ F123", "bottom", False),("100.8X20.6 &~ AH", "bottom", True),("  100.0456 ", "other", False),
#         ("8.0X100.0X150.2","top",False),("8.0X150.0 ~& DH","bottom",False),("8.0X150.0 ~& DH","bottom",True),(" $&%~~ F1180","bottom",False),
#     ]
#     for label, position, is_fb_flag in labels:
#         result = parse_elbow_plate(label, annotation_position=position, is_fb=is_fb_flag)
#         print(f"Parse Result ({position}, is_fb={is_fb_flag}):", result)
#         print(is_useful_text(label))



from element import *

def get_segment_blocks(segment: DSegment, rect, M, N):
    """
    计算给定线段在矩形框划分的 MxN 网格中占据的块。
    
    参数:
        segment: DSegment 实例，线段。
        rect: (rect_x_min, rect_x_max, rect_y_min, rect_y_max)，矩形框范围。
        M: 列数（块的宽方向划分）。
        N: 行数（块的高方向划分）。
    
    返回:
        set: 占据的块索引集合，形式为 (row, col)。
    """
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect
    cell_width = (rect_x_max - rect_x_min) / M
    cell_height = (rect_y_max - rect_y_min) / N

    # 获取线段的起点和终点
    x0, y0 = segment.start_point.x, segment.start_point.y
    x1, y1 = segment.end_point.x, segment.end_point.y

    # 计算步长比例
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # 按网格调整步长
    sx = cell_width if x0 < x1 else -cell_width
    sy = cell_height if y0 < y1 else -cell_height

    # 初始化误差项（考虑网格尺寸）
    err = dx / cell_width - dy / cell_height

    # 存储占据的块索引
    grids = set()
    final_col_index = int((x1 - rect_x_min) / cell_width)
    final_row_index = int((y1 - rect_y_min) / cell_height)
    while True:
        # 计算当前点所属的块索引
        col_index = int((x0 - rect_x_min) / cell_width)
        row_index = int((y0 - rect_y_min) / cell_height)

        # 确保索引在范围内
        if 0 <= col_index < M and 0 <= row_index < N:
            grids.add((row_index, col_index))

        # 终止条件
        if col_index == final_col_index and row_index == final_row_index:
            break

        # 更新误差项和当前点
        e2 = 2 * err
        if e2 > -dy / cell_height:
            err -= dy / cell_height
            x0 += sx
        if e2 < dx / cell_width:
            err += dx / cell_width
            y0 += sy

    return grids



# 定义矩形框范围和网格划分
rect = (0, 10000, 0, 8000)  # 矩形框 (x_min, x_max, y_min, y_max)
M, N = 10, 10 # 分成 5x5 网格

# 定义线段
start = DPoint(1000, 800)
end = DPoint(9999, 7895)
segment = DSegment(start, end)

# 计算占据的块
blocks = get_segment_blocks(segment, rect, M, N)
print("占据的网格块:", blocks)



import matplotlib.pyplot as plt

# Visualize the grid and segment

def visualize_grid_and_segment(segments, poly,rect, M, N, blocks):
    """
    Visualizes the grid and highlights the segment and occupied blocks.

    Parameters:
        segment: The DSegment instance to visualize.
        rect: Tuple defining the rectangular area (x_min, x_max, y_min, y_max).
        M: Number of columns in the grid.
        N: Number of rows in the grid.
        blocks: The set of occupied blocks as (row, col) indices.
    """
    rect_x_min, rect_x_max, rect_y_min, rect_y_max = rect
    cell_width = (rect_x_max - rect_x_min) / M
    cell_height = (rect_y_max - rect_y_min) / N

    # Create the grid
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(M + 1):
        x = rect_x_min + i * cell_width
        ax.plot([x, x], [rect_y_min, rect_y_max], color='black', linestyle='--', linewidth=0.5)
    for j in range(N + 1):
        y = rect_y_min + j * cell_height
        ax.plot([rect_x_min, rect_x_max], [y, y], color='black', linestyle='--', linewidth=0.5)

    # Highlight the occupied blocks
    for row, col in blocks:
        x = rect_x_min + col * cell_width
        y = rect_y_min + row * cell_height
        ax.add_patch(plt.Rectangle((x, y), cell_width, cell_height, color='blue', alpha=0.3))

    # Plot the segment
    for s in segments:
        x0, y0 = segment.start_point.x, segment.start_point.y
        x1, y1 = segment.end_point.x, segment.end_point.y
        ax.plot([x0, x1], [y0, y1], color='green', linewidth=2, label='Segment')
    for s in poly:
        x0, y0 = segment.start_point.x, segment.start_point.y
        x1, y1 = segment.end_point.x, segment.end_point.y
        ax.plot([x0, x1], [y0, y1], color='red', linewidth=2, label='poly')
    # Set axis limits and labels
    ax.set_xlim(rect_x_min, rect_x_max)
    ax.set_ylim(rect_y_min, rect_y_max)
    ax.set_aspect('equal')
    ax.set_title("Grid and Segment Visualization")
    ax.legend()
    plt.show()


# Visualize the example
visualize_grid_and_segment(segment, rect, M, N, blocks)