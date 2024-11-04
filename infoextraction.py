from  element import *
from plot_geo import plot_geometry,plot_polys, plot_info_poly
import os

def log_to_file(filename, content):
    """将内容写入指定的文件。"""
    with open(filename, 'a') as file:  # 以追加模式打开文件
        file.write(content + '\n')  # 写入内容并换行

def clear_file(file_path):
    """清空指定的文件"""
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # 清空文件内容
    except Exception as e:
        print(f"无法清空文件 '{file_path}'。错误: {e}")


# 计算几何中心坐标
def calculate_poly_centroid(poly):
    points = []
    for segment in poly:
        points.append(segment.start_point)
        points.append(segment.end_point)
        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
    return (x, y)

def are_equal_with_tolerance(k1, k2, tolerance = 0.1):
    return abs(k1 - k2) <= tolerance

def calculate_poly_refs(poly):
    refs = []
    for segment in poly:
        if isinstance(segment.ref, DArc) and len(refs) != 0 and isinstance(refs[-1].ref, DArc):
            if(segment.ref.start_angle, segment.ref.end_angle, segment.ref.center, segment.ref.radius) == (refs[-1].ref.start_angle, refs[-1].ref.end_angle, refs[-1].ref.center, refs[-1].ref.radius):
                continue
        elif isinstance(segment.ref, DArc):
            refs.append(segment)
        else:
            if len(refs) != 0:
                dx_1 = segment.end_point.x - segment.start_point.x
                dy_1 = segment.end_point.y - segment.start_point.y
                k_1 = float('inf') if dx_1 == 0 else dy_1 / dx_1
                dx_2 = refs[-1].end_point.x - refs[-1].start_point.x
                dy_2 = refs[-1].end_point.y - refs[-1].start_point.y
                k_2 = float('inf') if dx_2 == 0 else dy_2 / dx_2

                # 判断斜率是否相等
                if are_equal_with_tolerance(k_1, k_2) or (dx_1 == 0 and dx_2 == 0):
                    new_segment = DSegment(
                        refs[-1].start_point,     # 保留原本的起点
                        segment.end_point,        # 当前segment的终点
                        refs[-1].ref              # 保留原本的ref
                    )
                    
                    # 替换refs中的最后一个Segment
                    refs[-1] = new_segment
                    continue
            refs.append(segment)
    # 末尾和首个seg的合并判断
    if isinstance(refs[0].ref, DArc)  and isinstance(refs[-1].ref, DArc):
        if(refs[0].ref.start_angle, refs[0].ref.end_angle, refs[0].ref.center, refs[0].ref.radius) == (refs[-1].ref.start_angle, refs[-1].ref.end_angle, refs[-1].ref.center, refs[-1].ref.radius):
            del refs[-1]
    elif isinstance(refs[0].ref, DArc):
        pass
    else:
        if len(refs) != 0:
            dx_1 = refs[0].end_point.x - refs[0].start_point.x
            dy_1 = refs[0].end_point.y - refs[0].start_point.y
            k_1 = float('inf') if dx_1 == 0 else dy_1 / dx_1
            dx_2 = refs[-1].end_point.x - refs[-1].start_point.x
            dy_2 = refs[-1].end_point.y - refs[-1].start_point.y
            k_2 = float('inf') if dx_2 == 0 else dy_2 / dx_2

            # 判断斜率是否相等
            if are_equal_with_tolerance(k_1, k_2) or (dx_1 == 0 and dx_2 == 0):
                new_segment = DSegment(
                    refs[-1].start_point,     # 保留原本的起点
                    refs[0].end_point,        # 当前segment的终点
                    refs[-1].ref              # 保留原本的ref
                )
                
                # 替换refs中的第一个Segment
                refs[0] = new_segment
                del refs[-1]
    return refs


def outputPolyInfo(poly, segments, segmentation_config, point_map, index):
    # step1: 计算几何中心坐标
    poly_centroid = calculate_poly_centroid(poly)

    # step2: 合并边界线
    poly_refs = calculate_poly_refs(poly)

    # step3: 标记角隅孔
    corner_holes = []
    for i,segment in enumerate(poly_refs):
        # 如果是角隅孔(圆弧角隅孔和直线角隅孔)
        if isinstance(segment.ref, DArc) and segment.ref.radius > 20 and segment.ref.radius < 160:
            poly_refs[i].isCornerhole = True
            segment.isCornerhole = True
            corner_holes.append(segment)
        # TODO:判断并标记直线角隅孔
        # elif isinstance(segment.ref, DLine):
        #     continue
    
    # TODO: 根据星形角隅孔的位置，将角隅孔的坐标标记到相邻segment的StarCornerhole属性中，同时将该边标记为固定边

    # step4: 标记固定边
    for i, segment in enumerate(poly_refs):
        # 颜色确定
        if segment.ref.color == 3:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 角隅孔确定
        elif poly_refs[(i - 1) % len(poly_refs)].isCornerhole or poly_refs[(i + 1) % len(poly_refs)].isCornerhole:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 平行线确定
        else:
            for other in segments:
                # 检查是否平行
                dx_1 = segment.end_point.x - segment.start_point.x
                dy_1 = segment.end_point.y - segment.start_point.y
                dx_2 = other.end_point.x - other.start_point.x
                dy_2 = other.end_point.y - other.start_point.y
                
                # 计算斜率
                k_1 = float('inf') if dx_1 == 0 else dy_1 / dx_1
                k_2 = float('inf') if dx_2 == 0 else dy_2 / dx_2

                # 如果平行
                if (dx_1 == 0 and dx_2 == 0):
                    # 计算两条平行线之间的距离
                    distance = abs(other.end_point.x - segment.end_point.x)
                    # 如果距离在指定范围内，标记为固定边
                    if distance > 15 and distance < 140:
                        segment.isConstraint = True
                        poly_refs[i].isConstraint = True
                elif are_equal_with_tolerance(k_1, k_2) or (dx_1 == 0 and dx_2 == 0):
                    # 计算直线的系数A, B, C
                    A1 = dy_1
                    B1 = -dx_1
                    C1 = (dx_1 * segment.start_point.y) - (dy_1 * segment.start_point.x)
                    A2 = dy_2
                    B2 = -dx_2
                    C2 = (dx_2 * other.start_point.y) - (dy_2 * other.start_point.x)
                    # 计算两条平行线之间的距离
                    distance = abs(C1 - C2) / ((A1**2 + B1**2) ** 0.5)
                    # 如果距离在指定范围内，标记为固定边
                    if distance > 15 and distance < 140:
                        segment.isConstraint = True
                        poly_refs[i].isConstraint = True

    # 属于同一参考线的边只要有一个是固定边，则所有都是固定边
    for a in range(len(poly_refs)):
        for b in range(len(poly_refs)):
            if isinstance(poly_refs[a].ref, DLwpolyline) and isinstance(poly_refs[b].ref, DLwpolyline):
                if poly_refs[a].ref.points[0] == poly_refs[b].ref.points[0] and poly_refs[a].ref.points[-1] == poly_refs[b].ref.points[-1]:
                    poly_refs[b].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint
                    poly_refs[a].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint
            if isinstance(poly_refs[a].ref, DLine) and isinstance(poly_refs[b].ref, DLine):
                if poly_refs[a].ref.start_point == poly_refs[b].ref.start_point and poly_refs[a].ref.end_point == poly_refs[b].ref.end_point:
                    poly_refs[b].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint
                    poly_refs[a].isConstraint = poly_refs[a].isConstraint or poly_refs[b].isConstraint

    # step:5 确定自由边，合并相邻的固定边
    constraint_edges = []
    free_edges = []
    edges = []

    # 初始化合并边的列表和当前的合并状态
    current_edge = []
    is_current_constraint = None

    for segment in poly_refs:
        # 如果当前段是角隅孔，直接处理
        if segment.isCornerhole:
            # 如果有合并的边，保存到相应的列表
            if current_edge:
                if is_current_constraint:
                    constraint_edges.append(current_edge)
                    edges.append(current_edge)
                else:
                    free_edges.append(current_edge)
                    edges.append(current_edge)
            # 重置合并状态
            edges.append([segment])
            current_edge = []
            is_current_constraint = None
            continue

        # 处理固定边
        if segment.isConstraint:
            if is_current_constraint is None:  # 如果当前没有边在合并
                current_edge = [segment]  # 开始一个新的合并边
                is_current_constraint = True
            elif is_current_constraint:  # 如果当前是固定边
                if current_edge[-1].ref.color == segment.ref.color:  # 如果颜色相同
                    current_edge.append(segment)  # 添加到当前边
                else:
                    # 保存当前合并边，并开始新的合并边
                    constraint_edges.append(current_edge)
                    edges.append(current_edge)
                    current_edge = [segment]  # 开始新的合并边
            else:
                # 如果当前是自由边，保存当前合并边，并开始新的合并边
                constraint_edges.append(current_edge)
                edges.append(current_edge)
                current_edge = [segment]  # 开始新的合并边
                is_current_constraint = True

        # 处理自由边
        else:
            if is_current_constraint is None:  # 如果当前没有边在合并
                current_edge = [segment]  # 开始一个新的合并边
                is_current_constraint = False
            elif not is_current_constraint:  # 如果当前是自由边
                if current_edge[-1].ref.color == segment.ref.color:  # 如果颜色相同
                    current_edge.append(segment)  # 添加到当前边
                else:
                    # 保存当前合并边，并开始新的合并边
                    free_edges.append(current_edge)
                    edges.append(current_edge)
                    current_edge = [segment]  # 开始新的合并边
            else:
                # 如果当前是固定边，保存当前合并边，并开始新的合并边
                free_edges.append(current_edge)
                edges.append(current_edge)
                current_edge = [segment]  # 开始新的合并边
                is_current_constraint = False

    # 添加最后一个合并边，并检查与第一条边的属性
    if current_edge:
        if is_current_constraint:
            # 检查最后一个合并边的属性是否与第一条边相同
            if poly_refs[0].isConstraint and not poly_refs[0].isCornerhole:
                constraint_edges[0].insert(0, *current_edge)  # 合并到第一条固定边
                edges[0].insert(0, *current_edge)
            else:
                constraint_edges.append(current_edge)
                edges.append(current_edge)
        else:
            # 检查最后一个合并边的属性是否与第一条边相同
            if not poly_refs[0].isConstraint and not poly_refs[0].isCornerhole:
                free_edges[0].insert(0, *current_edge)  # 合并到第一条自由边
                edges[0].insert(0, *current_edge)
            else:
                free_edges.append(current_edge)
                edges.append(current_edge)


    # 如果有多于两条的自由边则判定为不是肘板，不进行输出
    if len(free_edges) > 1:
        print(f"回路{index}超过两条自由边！")
        return

    # step6: 绘制对边分类后的几何图像
    plot_info_poly(poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'))


    # step7: 输出几何中心和边界信息
    file_path = os.path.join(segmentation_config.poly_info_dir, f'info{index}.txt')
    clear_file(file_path)
    log_to_file(file_path, f"几何中心坐标：{poly_centroid}")
    log_to_file(file_path, f"边界数量（含自由边）：{len(constraint_edges) + len(free_edges)}")
    log_to_file(file_path, "边界信息（固定边）：")
    for i, edge in enumerate(constraint_edges):
        log_to_file(file_path, f"边界颜色{i + 1}: {edge[0].ref.color}")
        log_to_file(file_path, f"边界轮廓{i + 1}: ")
        for seg in edge:
            if isinstance(seg.ref, DArc):
                log_to_file(file_path, f"起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）")
            else:
                log_to_file(file_path, f"起点：{seg.start_point}、终点{seg.end_point}（直线）")

    # step8: 输出自由边信息
    log_to_file(file_path, "自由边轮廓：")
    for seg in free_edges[0]:
        if isinstance(seg.ref, DArc):
            log_to_file(file_path, f"起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）")
        else:
            log_to_file(file_path, f"起点：{seg.start_point}、终点{seg.end_point}（直线）")

    # step9: 输出角隅孔信息
    cornerhole_index = 1
    for edge in edges:
        # 圆弧角隅孔和直线角隅孔
        seg = edge[0]
        if seg.isCornerhole:
            log_to_file(file_path, f"角隅孔{cornerhole_index}轮廓：")
            if isinstance(seg.ref, DArc):
                log_to_file(file_path, f"起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）")
            else:
                log_to_file(file_path, f"起点：{seg.start_point}、终点{seg.end_point}（直线）")
            cornerhole_index += 1
        # 星形角隅孔
        if seg.StarCornerhole is not None:
            log_to_file(file_path, f"角隅孔{cornerhole_index}轮廓：")
            log_to_file(file_path, f"坐标：{seg.StarCornerhole}（星形角隅孔）")
            cornerhole_index += 1

    # step10: TODO：输出周围标注信息

    # step11: 输出角隅孔和边界之间的关系
    cornerhole_index = 1
    edge_index = 0
    for edge in edges:
        seg = edge[0]
        # 如果是固定边，边界计数加一
        if not seg.isCornerhole and seg.isConstraint:
            edge_index += 1

        # 圆弧角隅孔和直线角隅孔
        if seg.isCornerhole:
            pre = len(constraint_edges) if edge_index == 0 else edge_index
            nex = (pre + 1) % len(constraint_edges)
            nex = len(constraint_edges) if nex == 0 else nex
            log_to_file(file_path, f"角隅孔{cornerhole_index}位于边界{pre}和边界{nex}之间")
            cornerhole_index += 1

        # 星形角隅孔
        if seg.StarCornerhole is not None:
            pre = len(constraint_edges) if edge_index == 0 else edge_index
            nex = (pre + 1) % len(constraint_edges)
            nex = len(constraint_edges) if nex == 0 else nex
            log_to_file(file_path, f"角隅孔{cornerhole_index}位于边界{pre}和边界{nex}之间")
            cornerhole_index += 1
