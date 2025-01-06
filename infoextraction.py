from  element import *
from plot_geo import plot_geometry,plot_polys, plot_info_poly
import os
from utils import segment_intersection,computeBoundingBox,is_parallel
from classifier import poly_classifier
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from bracket_parameter_extraction import parse_elbow_plate
def is_point_in_polygon(point, polygon_edges):
    
    polygon_points = set()  # Concave polygon example
    for edge in polygon_edges:
        vs,ve=edge.start_point,edge.end_point
        polygon_points.add((vs.x,vs.y))
        polygon_points.add((ve.x,ve.y))

    polygon_points = list(polygon_points)

    polygon = Polygon(polygon_points)

    point = Point(point.x, point.y)

    # Check if the point is inside the polygon
    is_inside = polygon.contains(point)

    return is_inside
def log_to_file(filename, content):
    """将内容写入指定的文件。"""
    with open(filename, 'a', encoding='utf-8') as file:  # 以追加模式打开文件
        file.write(content + '\n')  # 写入内容并换行

def clear_file(file_path):
    """清空指定的文件"""
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # 清空文件内容
    except Exception as e:
        print(f"无法清空文件 '{file_path}'。错误: {e}")


def polygon_area(points):
    """
    计算多边形的面积，基于顶点的顺序。
    :param points: 多边形顶点列表 [(x1, y1), (x2, y2), ...]。
    :return: 多边形面积（绝对值）。
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0

def is_near_convex(points, i, tolerance=0.1):
    """
    判断多边形是否是近似凸多边形（基于面积差）。
    :param points: 多边形顶点列表，按顺序排列 [(x1, y1), (x2, y2), ...]。
    :param i: 当前多边形的索引，用于命名图片文件。
    :param tolerance: 面积差的允许百分比，默认为 5%。
    :return: True 如果是近似凸多边形，否则 False。
    """
    if len(points) < 3:
        return False  # 不构成多边形

    # 确保输入点集是一个二维数组
    points = [(float(x), float(y)) for x, y in points]

    # 计算输入多边形面积
    poly_area = polygon_area(points)

    # 计算凸包的面积
    convex_hull = ConvexHull(points)
    hull_area = convex_hull.volume  # ConvexHull 的面积
    # 检查面积差
    #print(poly_area,hull_area,abs(poly_area - hull_area) / hull_area)
    if abs(poly_area - hull_area) / hull_area > tolerance:
        return False

    return True

def is_near_rectangle(points, area, tolerance = 0.01):
    if len(points) < 3:
        return False  # 不构成多边形

    # 确保输入点集是一个二维数组
    points = [(float(x), float(y)) for x, y in points]

    # 计算输入多边形面积
    poly_area = polygon_area(points)
    
    if abs(poly_area - area) / area > tolerance:
        return False

    return True


def calculate_angle(point1, point2, point3):
    """
    计算由三点确定的两向量之间的夹角（以度为单位）。

    Parameters:
        point1, point2, point3: 各为一个元组，表示点的坐标 (x, y)。

    Returns:
        angle: 两向量之间的夹角（较小的角，范围为 0-180°）。
    """
    def vector(a, b):
        return (b[0] - a[0], b[1] - a[1])

    def magnitude(v):
        return math.sqrt(v[0]**2 + v[1]**2)

    def dot_product(v1, v2):
        return v1[0] * v2[0] + v1[1] * v2[1]

    v1 = vector(point2, point1)
    v2 = vector(point2, point3)

    mag_v1 = magnitude(v1)
    mag_v2 = magnitude(v2)

    if mag_v1 == 0 or mag_v2 == 0:
        return 0  # 防止除零错误

    cos_theta = dot_product(v1, v2) / (mag_v1 * mag_v2)
    cos_theta = max(-1, min(1, cos_theta))  # 修正浮点误差

    angle = math.acos(cos_theta) * (180 / math.pi)  # 弧度转角度
    return angle

# 计算几何中心坐标
def calculate_poly_centroid(poly):
    points = []
    for segment in poly:
        points.append(segment.start_point)
        points.append(segment.end_point)
        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
    return (x, y)


def calculate_poly_refs(poly,segmentation_config):
    refs = []
    
    for segment in poly:
        if isinstance(segment.ref, DArc):
            if len(refs) != 0 and isinstance(refs[-1].ref, DArc):
                if (segment.ref.start_angle, segment.ref.end_angle, segment.ref.center, segment.ref.radius) == (
                    refs[-1].ref.start_angle, refs[-1].ref.end_angle, refs[-1].ref.center, refs[-1].ref.radius):
                    continue
            refs.append(segment)
        else:
            if len(refs) != 0:
                last_segment = refs[-1]
                # 判断是否平行
                if is_parallel(last_segment, segment,segmentation_config.is_parallel_tolerance):
                    new_segment = DSegment(
                        start_point=last_segment.start_point,
                        end_point=segment.end_point,
                        ref=last_segment.ref
                    )
                    refs[-1] = new_segment
                    continue
            refs.append(segment)

    # 判断首尾是否可以合并
    if len(refs) > 1:
        first_segment = refs[0]
        last_segment = refs[-1]

        if isinstance(first_segment.ref, DArc) and isinstance(last_segment.ref, DArc):
            if (first_segment.ref.start_angle, first_segment.ref.end_angle, first_segment.ref.center, first_segment.ref.radius) == (
                last_segment.ref.start_angle, last_segment.ref.end_angle, last_segment.ref.center, last_segment.ref.radius):
                refs.pop()
        elif not isinstance(first_segment.ref, DArc) and not isinstance(last_segment.ref, DArc):
            if is_parallel(first_segment, last_segment,segmentation_config.is_parallel_tolerance):
                new_segment = DSegment(
                    start_point=last_segment.start_point,
                    end_point=first_segment.end_point,
                    ref=last_segment.ref
                )
                refs[0] = new_segment
                refs.pop()

    return refs


def computePolygon(poly,tolerance = 0.1):
    polygon_points = set()  # Concave polygon example
    for edge in poly:
        vs,ve=edge.start_point,edge.end_point
        polygon_points.add((vs.x,vs.y))
        polygon_points.add((ve.x,ve.y))
    polygon_points = list(polygon_points)
    polygon = Polygon(polygon_points)
    
    polygon_with_tolerance = polygon.buffer(tolerance)

    return polygon_with_tolerance
def point_is_inside(point,polygon):
    point_=Point(point.x,point.y)
    return polygon.contains(point_)


def stiffenersInPoly(stiffeners,poly,segmentation_config):
    # Define your polygon (list of vertices)
    sf=[]
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    # polygon_points = set()  # Concave polygon example
    # for edge in poly:
    #     vs,ve=edge.start_point,edge.end_point
    #     polygon_points.add((vs.x,vs.y))
    #     polygon_points.add((ve.x,ve.y))
    # polygon_points = list(polygon_points)
    # polygon = Polygon(polygon_points)
    # tolerance = 0.1 # 误差值
    # polygon_with_tolerance = polygon.buffer(tolerance)
    for stiffener in stiffeners:
        mid=stiffener.mid_point()
        # point = Point(mid.x,mid.y)
        
        # if polygon_with_tolerance.contains(point):
        #     sf.append(stiffener)
        if x_min <= mid.x and mid.x <=x_max and y_min <=mid.y and y_max>=mid.y:
            sf.append(stiffener)
    return sf
def textsInPoly(text_map,poly,segmentation_config,is_fb):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ts=[]
    for pos,texts in text_map.items():
        for t in texts:
            if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
                if t[1]["Type"]=="FB" or t[1]["Type"]=="FL":
                    result=parse_elbow_plate(t[0].content, "bottom",is_fb)
                else:
                    result=t[1]
                ts.append([t[0],pos,result,t[2]])#element,position,result,anotation
    return ts

def braketTextInPoly(braket_texts,braket_pos,poly,segmentation_config):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ts=[]
    bps=[]
    for i,t in enumerate(braket_pos):
        pos=t
        if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
            ts.append(braket_texts[i])
            bps.append(braket_pos[i])
    return ts,bps


def dimensionsInPoly(dimensions,poly,segmentation_config):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    if segmentation_config.bracket_bbox_expand_is_ratio:
        xx=(x_max-x_min)*segmentation_config.bracket_bbox_expand_ratio
        yy=(y_max-y_min)*segmentation_config.bracket_bbox_expand_ratio
    else:
        xx=segmentation_config.bracket_bbox_expand_length
        yy=segmentation_config.bracket_bbox_expand_length
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ds=[]
    for d_t in dimensions:
        pos=d_t[1]
        d=d_t[0]
        if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
            ds.append([d,pos])
    return ds
def outputPolyInfo(poly, segments, segmentation_config, point_map, index,star_pos_map,cornor_holes,texts,dimensions,text_map,stiffeners):
    # step1: 计算几何中心坐标
    poly_centroid = calculate_poly_centroid(poly)

    # step2: 合并边界线
    poly_refs = calculate_poly_refs(poly,segmentation_config)



    # plot_polys({},poly_refs,"./check")
    # step3: 标记角隅孔
    # for corner_hole in cornor_holes:
    #     for seg in corner_hole.segments:
    #         seg.isCornerHole=True
    # print(len(cornor_holes))
    # for cornor_hole in cornor_holes:
    #     print(cornor_hole.segments)
    cornor_holes_map={}
    for cornor_hole in cornor_holes:
        for s in cornor_hole.segments:
            cornor_holes_map[s]=cornor_hole.ID
            cornor_holes_map[DSegment(s.end_point,s.start_point,s.ref)]=cornor_hole.ID
    for seg in poly_refs:
        if seg in cornor_holes_map:
            seg.isCornerhole=True
            #print(seg)
            

    
    #  根据星形角隅孔的位置，将角隅孔的坐标标记到相邻segment的StarCornerhole属性中，同时将该边标记为固定边
    star_set=set()
    for s in poly:
        if s in star_pos_map:
            for ss in star_pos_map[s]:
                star_set.add(ss)
    stars_pos=list(star_set)
    for p in stars_pos:
        x,y=p.x,p.y
        lines=[]
        lines.append(DSegment(DPoint(x,y),DPoint(x-5000,y)))
        lines.append(DSegment(DPoint(x,y),DPoint(x+5000,y)))
        lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
        lines.append(DSegment(DPoint(x,y),DPoint(x,y-5000)))
        cornor=[]
        for i, seg1 in enumerate(lines):
            dist=None
            s=None
            for j, seg2 in enumerate(poly_refs):
                p1, p2 = seg1.start_point, seg1.end_point
                q1, q2 = seg2.start_point, seg2.end_point
                intersection = segment_intersection(p1, p2, q1, q2)
                if intersection:
                    if dist is None:
                        dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
                        s=seg2
                    else:
                        if dist>(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y):
                            dist=(intersection[0]-p1.x)*(intersection[0]-p1.x)+(intersection[1]-p1.y)*(intersection[1]-p1.y)
                            s=seg2
            if s is not None:
               cornor.append((dist,s,p))
        cornor=sorted(cornor,key=lambda p:p[0])
        if len(cornor)>=2:
            cornor[0][1].isConstraint=True
            cornor[0][1].isCornerhole=False
            cornor[0][1].StarCornerhole=cornor[0][2]
            cornor[1][1].isConstraint=True
            cornor[1][1].isCornerhole=False
            cornor[1][1].StarCornerhole=cornor[1][2]
    

   
    # step4: 标记固定边
    for i, segment in enumerate(poly_refs):
        # 颜色确定
        if segment.ref.color in segmentation_config.constraint_color:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 角隅孔确定
        elif poly_refs[(i - 1) % len(poly_refs)].isCornerhole or poly_refs[(i + 1) % len(poly_refs)].isCornerhole:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 平行线确定
        # else:
        #     dx_1 = segment.end_point.x - segment.start_point.x
        #     dy_1 = segment.end_point.y - segment.start_point.y
        #     mid_point=DPoint((segment.end_point.x+segment.start_point.x)/2,(segment.end_point.y+segment.start_point.y)/2)
        #     l = (dx_1**2 + dy_1**2)**0.5
        #     v_1 = (dy_1 / l * segmentation_config.parallel_max_distance, -dx_1 / l * segmentation_config.parallel_max_distance)
        #     point1,point2,point3=DSegment(segment.start_point,mid_point).mid_point(),DSegment(segment.end_point,mid_point).mid_point(),mid_point
        #     for j, other in enumerate(segments):
        #         if segment == other:
        #             continue
                
        #         if is_parallel(segment, other,segmentation_config.is_parallel_tolerance):
        #             #print(segment,other)
        #             s1 = DSegment(
        #                 DPoint(point1.x + v_1[0], point1.y + v_1[1]),
        #                 DPoint(point1.x - v_1[0], point1.y - v_1[1])
        #             )
        #             s2 = DSegment(
        #                 DPoint(point2.x + v_1[0], point2.y + v_1[1]),
        #                 DPoint(point2.x - v_1[0], point2.y - v_1[1])
        #             )
        #             s3 = DSegment(
        #                 DPoint(mid_point.x + v_1[0], mid_point.y + v_1[1]),
        #                 DPoint(mid_point.x - v_1[0], mid_point.y - v_1[1])
        #             )

        #             i1 = segment_intersection(s1.start_point, s1.end_point, other.start_point, other.end_point)
        #             if i1 == other.end_point or i1 == other.start_point:
        #                 i1 = None
                    
        #             i2 = segment_intersection(s2.start_point, s2.end_point, other.start_point, other.end_point)
        #             if i2 == other.end_point or i2 == other.start_point:
        #                 i2 = None
        #             i3 = segment_intersection(s3.start_point, s3.end_point, other.start_point, other.end_point)
        #             if i3 == other.end_point or i3 == other.start_point:
        #                 i3 = None
        #             if i1 is not None and DSegment(i1,point1).length()<segmentation_config.parallel_min_distance:
        #                 i1=None
        #             if i2 is not None and DSegment(i2,point2).length()<segmentation_config.parallel_min_distance:
        #                 i2=None
        #             if i3 is not None and DSegment(i3,point3).length()<segmentation_config.parallel_min_distance:
        #                 i3=None
        #             if i1 is not None and i2 is not None and i3 is not None:
        #                 segment.isConstraint = True
        #                 poly_refs[i].isConstraint = True
        #                 # if poly_refs[i].length()<=27 and poly_refs[i].length()>=25:
        #                 #     print(poly_refs[i])
        #                 #     print(other)
        #                 #     print(i1,i2,i3)
        #                 #print(segment,other)
        #                 break
    
            
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
    
    #加强结构
    sfs=stiffenersInPoly(stiffeners,poly,segmentation_config)
    is_fb=False
    others=set()
    st_segments=set()
    fb_segments=set()
    fl_segments=set()
    #查找相邻结构
    for i,segment in enumerate(poly_refs):
        if True:
            dx_1 = segment.end_point.x - segment.start_point.x
            dy_1 = segment.end_point.y - segment.start_point.y
            mid_point=DPoint((segment.end_point.x+segment.start_point.x)/2,(segment.end_point.y+segment.start_point.y)/2)
            l = (dx_1**2 + dy_1**2)**0.5
            v_1 = (dy_1 / l * segmentation_config.parallel_max_distance_relax, -dx_1 / l * segmentation_config.parallel_max_distance_relax)
            point1,point2,point3=DSegment(segment.start_point,mid_point).mid_point(),DSegment(segment.end_point,mid_point).mid_point(),mid_point
            for j, other in enumerate(segments):
                if segment == other:
                    continue
                
                if is_parallel(segment, other,segmentation_config.is_parallel_tolerance):
                    #print(segment,other)
                    s1 = DSegment(
                        DPoint(point1.x + v_1[0], point1.y + v_1[1]),
                        DPoint(point1.x - v_1[0], point1.y - v_1[1])
                    )
                    s2 = DSegment(
                        DPoint(point2.x + v_1[0], point2.y + v_1[1]),
                        DPoint(point2.x - v_1[0], point2.y - v_1[1])
                    )
                    s3 = DSegment(
                        DPoint(mid_point.x + v_1[0], mid_point.y + v_1[1]),
                        DPoint(mid_point.x - v_1[0], mid_point.y - v_1[1])
                    )

                    i1 = segment_intersection(s1.start_point, s1.end_point, other.start_point, other.end_point)
                    if i1 == other.end_point or i1 == other.start_point:
                        i1 = None
                    
                    i2 = segment_intersection(s2.start_point, s2.end_point, other.start_point, other.end_point)
                    if i2 == other.end_point or i2 == other.start_point:
                        i2 = None
                    i3 = segment_intersection(s3.start_point, s3.end_point, other.start_point, other.end_point)
                    if i3 == other.end_point or i3 == other.start_point:
                        i3 = None
                    # if i1 is not None and DSegment(i1,point1).length()<segmentation_config.parallel_min_distance_relax:
                    #     i1=None
                    # if i2 is not None and DSegment(i2,point2).length()<segmentation_config.parallel_min_distance_relax:
                    #     i2=None
                    # if i3 is not None and DSegment(i3,point3).length()<segmentation_config.parallel_min_distance_relax:
                    #     i3=None
                    if i1 is not None and i2 is not None and i3 is not None:
                        distance=DSegment(i3,point3).length()
                        l1=segment.length()
                        l2=other.length()
                        if  distance<segmentation_config.parallel_min_distance:
                            continue
                        if distance < segmentation_config.parallel_max_distance:
                            #contraint
                            others.add(other)
                            segment.isPart=True
                            poly_refs[i].isPart=True
                            segment.isConstraint = True
                            poly_refs[i].isConstraint = True
                        elif l1<l2 *segmentation_config.contraint_factor:
                            #constraint
                            others.add(other)
                            segment.isPart=True
                            poly_refs[i].isPart=True
                            segment.isConstraint = True
                            poly_refs[i].isConstraint = True
                        else:

                            others.add(other)
                            segment.isPart=True
                            poly_refs[i].isPart=True
                            # st_segments.add(segment)
                               
    st_segments=list(st_segments)
    others=list(others)
    # step:5 确定自由边，合并相邻的固定边
    constraint_edges = []
    free_edges = []
    cornerhole_edges = []
    edges = []

    # 初始化合并边的列表和当前的合并状态
    current_edge = []
    current_type = None  # 使用三种类型：'constraint', 'free', 'cornerhole'

    for segment in poly_refs:
        # 处理角隅孔
        if segment.isCornerhole:
            # 如果当前是角隅孔边
            if current_type == 'cornerhole':
                current_edge.append(segment)  # 将当前段添加到当前的角隅孔边
            else:
                # 保存上一个合并边
                if current_edge:
                    if current_type == 'constraint':
                        constraint_edges.append(current_edge)
                    elif current_type == 'free':
                        free_edges.append(current_edge)
                    edges.append(current_edge)

                # 开始新的角隅孔边合并
                current_edge = [segment]
                current_type = 'cornerhole'

        # 处理固定边
        elif segment.isConstraint:
            if current_type == 'constraint':  # 如果当前在合并固定边
                if current_edge[-1].ref.color == segment.ref.color:  # 如果颜色相同
                    current_edge.append(segment)  # 添加到当前边
                else:
                    # 保存当前合并边，并开始新的合并边
                    constraint_edges.append(current_edge)
                    edges.append(current_edge)
                    current_edge = [segment]  # 开始新的合并边
            else:
                # 保存上一个合并边
                if current_edge:
                    if current_type == 'free':
                        free_edges.append(current_edge)
                    elif current_type == 'cornerhole':
                        cornerhole_edges.append(current_edge)
                    edges.append(current_edge)
                
                # 开始新的固定边合并
                current_edge = [segment]
                current_type = 'constraint'

        # 处理自由边
        else:
            if current_type == 'free':  # 如果当前在合并自由边
                if current_edge[-1].ref.color == segment.ref.color:  # 如果颜色相同
                    current_edge.append(segment)  # 添加到当前边
                else:
                    # 保存当前合并边，并开始新的合并边
                    free_edges.append(current_edge)
                    edges.append(current_edge)
                    current_edge = [segment]  # 开始新的合并边
            else:
                # 保存上一个合并边
                if current_edge:
                    if current_type == 'constraint':
                        constraint_edges.append(current_edge)
                    elif current_type == 'cornerhole':
                        cornerhole_edges.append(current_edge)
                    edges.append(current_edge)
                
                # 开始新的自由边合并
                current_edge = [segment]
                current_type = 'free'

    # 添加最后一个合并边，并检查与第一条边的属性
    if current_edge:
        if current_type == 'constraint':
            # 检查最后一个合并边的属性是否与第一条边相同
            if poly_refs[0].isConstraint and not poly_refs[0].isCornerhole and len(constraint_edges) > 0:
                constraint_edges[0] = current_edge + constraint_edges[0]  # 合并到第一条固定边
                edges[0] = current_edge + edges[0]
            else:
                constraint_edges.append(current_edge)
                edges.append(current_edge)
        elif current_type == 'free':
            if not poly_refs[0].isConstraint and not poly_refs[0].isCornerhole and len(free_edges) > 0:
                free_edges[0] = current_edge + free_edges[0]  # 合并到第一条自由边
                edges[0] = current_edge + edges[0]
            else:
                free_edges.append(current_edge)
                edges.append(current_edge)
        elif current_type == 'cornerhole':
            if poly_refs[0].isCornerhole and len(cornerhole_edges) > 0:
                cornerhole_edges[0] = current_edge + cornerhole_edges[0]
                edges[0] = current_edge + edges[0]
            else:
                cornerhole_edges.append(current_edge)
                edges.append(current_edge)


  
    # step 5.5：找到所有的标注
    
    # print(len(stiffeners))
    # print(len(sfs))
    
    # if len(sfs)>0:
    #     is_fb=True
    #     print(f"回路{index}有加强边！")
    # else:
    #     print(f"回路{index}没有加强边！")
    
    tis=textsInPoly(text_map,poly,segmentation_config,is_fb)
    ds=dimensionsInPoly(dimensions,poly,segmentation_config)
    # print(free_edges)
    # print(cornor_holes[0].segments)
    # for s in poly_refs:
    #     print(s.isCornerhole)
    # print(current_edge)
    # print(current_type)
    # # 附加：添加筛选条件
    # # 如果有多于两条的自由边则判定为不是肘板，不进行输出
    # print(constraint_edges)
    # print(free_edges)
    # print(cornerhole_edges)
    # for s in poly:
    #     print(s.isCornerhole)
 
  
        

    # print(constraint_edges)
    # print("===========")
    # print(free_edges)
    # print("=============")
    # print(cornerhole_edges)
    plot_info_poly(poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'),tis,ds,sfs,others)
    if len(free_edges) > 1:
        print(f"回路{index}超过两条自由边！")
        #return poly_refs
        return None
    if len(free_edges) == 0:
        print(f"回路{index}没有自由边！")
        #return poly_refs
        return  None
    if len(constraint_edges)==0:
        print(f"回路{index}没有约束边！")
        #return poly_refs
        return None
    
    for constarint_edge in constraint_edges:
        pass
    # 如果除去圆弧外固定边多边形不是凸多边形则不进行输出
    constraint_edge_poly = []

    for constarint_edge in constraint_edges:
        for seg in constarint_edge:
            if not (isinstance(seg.ref, DArc) and seg.ref.radius<200):
                constraint_edge_poly.append(seg.start_point)
                constraint_edge_poly.append(seg.end_point)

    if not is_near_convex(constraint_edge_poly, index,segmentation_config.near_convex_tolerance):
        print(f"回路{index}去圆弧外固定边多边形不是凸多边形")
        return None
    
    # 如果自由边轮廓中出现了夹角小于45°的折线则不进行输出
    for i in range(len(free_edges[0]) - 1):
        seg1 = free_edges[0][i]
        seg2 = free_edges[0][i + 1]
        if isinstance(seg1.ref, DArc) or isinstance(seg2.ref, DArc):
            continue
        angle = calculate_angle(seg1.start_point, seg1.end_point, seg2.end_point)
        if angle < segmentation_config.min_angle_in_free_edge:
            print(f"回路{index}自由边轮廓中出现了夹角小于45°的折线")
            return None

    # 如果自由边长度在总轮廓长度中占比不超过25%则不进行输出
    free_len = 0
    all_len = 0
    for seg in poly_refs:
        if isinstance(seg.ref, DArc):
            l = ((seg.ref.end_point.x - seg.ref.start_point.x) ** 2 +   
                (seg.ref.end_point.y - seg.ref.start_point.y) ** 2) ** 0.5
        else:
            l = seg.length()
        if not seg.isConstraint and not seg.isCornerhole:
            free_len += l
        all_len += l
    if free_len / all_len < segmentation_config.free_edge_ratio:
        print(f"回路{index}自由边长度在总轮廓长度中占比不超过{segmentation_config.free_edge_ratio*100}%")
        return None
    
    # 如果整体肘板轮廓接近于矩形，则不进行输出
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')
    points = []
    for seg in poly_refs:
        if isinstance(seg.ref, DArc):
            continue
        points.append(seg.start_point)
        points.append(seg.end_point)
        x_coords = [seg.start_point[0], seg.end_point[0]]
        y_coords = [seg.start_point[1], seg.end_point[1]]

        max_x = max(max_x, *x_coords)
        min_x = min(min_x, *x_coords)
        max_y = max(max_y, *y_coords)
        min_y = min(min_y, *y_coords)

    area = (max_x - min_x) * (max_y - min_y)
    if is_near_rectangle(points, area, segmentation_config.near_rectangle_tolerance):
        print("肘板轮廓接近矩形！")
        return None

    # step6: 绘制对边分类后的几何图像
    plot_info_poly(poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'),tis,ds,sfs,others)

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
                log_to_file(file_path, f"起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
            else:
                log_to_file(file_path, f"起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")

    # step8: 输出自由边信息
    log_to_file(file_path, "自由边轮廓：")
    for seg in free_edges[0]:
        if isinstance(seg.ref, DArc):
            log_to_file(file_path, f"起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
        else:
            log_to_file(file_path, f"起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")

    # step9: 输出角隅孔信息
    cornerhole_index = 1
    for edge in edges:
        # 圆弧角隅孔和直线角隅孔
        if edge[0].isCornerhole:
            log_to_file(file_path, f"角隅孔{cornerhole_index}轮廓：")
            for seg in edge:
                if isinstance(seg.ref, DArc):
                    log_to_file(file_path, f"起点：{seg.ref.start_point}、终点：{seg.ref.end_point}、圆心：{seg.ref.center}、半径：{seg.ref.radius}（圆弧）、句柄: {seg.ref.handle}")
                else:
                    log_to_file(file_path, f"起点：{seg.start_point}、终点{seg.end_point}（直线）、句柄: {seg.ref.handle}")
            cornerhole_index += 1
        # 星形角隅孔
        if seg.StarCornerhole is not None:
            log_to_file(file_path, f"角隅孔{cornerhole_index}轮廓：")
            log_to_file(file_path, f"坐标：{seg.StarCornerhole}（星形角隅孔）")
            cornerhole_index += 1

    # step10:输出周围标注信息
    k=0
    for i,t_t in enumerate(tis):
        t=t_t[0]
        pos=t_t[1]
        content=t.content
        log_to_file(file_path,f"标注{i+1}:")
        log_to_file(file_path,f"位置: {pos}、内容: {content}、颜色: {t.color}、句柄: {t.handle}")
        log_to_file(file_path,str(t_t[2]))
        log_to_file(file_path,t_t[3])
        k+=1
    for i,d_t in enumerate(ds):
        d=d_t[0]
        pos=d_t[1]
        content=d.text
        log_to_file(file_path,f"标注{i+1+k}:")
        log_to_file(file_path,f"位置: {pos}、内容: {content}、颜色: {d.color}、类型： {d.dimtype}、句柄: {d.handle}")

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

    # step12 对肘板进行分类：
    classification_res = poly_classifier(poly_refs, cornerhole_index - 1, free_edges, edges, 
                                         segmentation_config.type_path, segmentation_config.json_output_path, 
                                         f"{os.path.splitext(os.path.basename(segmentation_config.json_path))[0]}_infopoly{index}",
                                         is_output_json=True)
    
    log_to_file(file_path, f"肘板类别为{classification_res}")
    if classification_res == "Unclassified":
        return poly_refs

    return poly_refs
