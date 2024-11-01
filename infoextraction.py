from  element import *
from plot_geo import plot_geometry,plot_polys, plot_info_poly
import os

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

    # step2: 合并边界
    poly_refs = calculate_poly_refs(poly)

    # step3: 标记角隅孔
    corner_holes = []
    for i,segment in enumerate(poly_refs):
        # 如果是角隅孔(圆弧角隅孔和直线角隅孔)
        if isinstance(segment.ref, DArc) and segment.ref.radius > 20 and segment.ref.radius < 160:
            poly_refs[i].isCornerhole = True
            segment.isCornerhole = True
            corner_holes.append(segment)
        # TODO：判断并标记直线角隅孔
        # elif isinstance(segment.ref, DLine):
        #     continue

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
    

    # 如果有多余两条的自由边则判定为不是肘板，不进行输出

    # 绘制对边分类后的几何图像
    plot_info_poly(poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'))

    # step6: 对相邻的同类型的边进行合并


    # step7: 输出几何中心和边界信息
    print("几何中心坐标：", poly_centroid)

    # step8: 输出自由边信息

    # step9: 输出角隅孔信息
