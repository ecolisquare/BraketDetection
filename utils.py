import json 
from  element import *
import math
from SweepIntersectorLib.SweepIntersector import SweepIntersector
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import deque

def angleOfTwoVectors(A,B):
    lengthA = math.sqrt(A[0]**2 + A[1]**2)  
    lengthB = math.sqrt(B[0]**2 + B[1]**2)  
    dotProduct = A[0] * B[0] + A[1] * B[1]   
    angle = math.acos(dotProduct / (lengthA * lengthB))
    angle_degrees = angle * (180 / math.pi)  
    return angle_degrees

# Ramer-Douglas-Peucker algorithm for line simplification
def rdp(points, epsilon):
    if len(points) < 3:
        return points
    # Find the point with the maximum distance from the line segment (first to last point)
    start = points[0]
    end = points[-1]
    line_vec = np.array([end.x - start.x, end.y - start.y])
    line_len = np.linalg.norm(line_vec)

    max_dist = 0
    index = 0

    for i in range(1, len(points) - 1):
        p = points[i]
        vec = np.array([p.x - start.x, p.y - start.y])
        proj_len = np.dot(vec, line_vec) / line_len
        proj_point = np.array([start.x, start.y]) + proj_len * (line_vec / line_len)
        dist = np.linalg.norm([p.x, p.y] - proj_point)

        if dist > max_dist:
            max_dist = dist
            index = i

    # If the max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon:
        left = rdp(points[:index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]  

#json --> elements
def readJson(path):
    elements=[]
    segments=[]
    color = [3, 7, 4]
    linetype = ["BYLAYER", "Continuous"]


    try:  
        with open(path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)[0]  
        
        for ele in data_list:
            e=None
            # 颜色过滤
            if ele["color"] not in color:
                continue
            # 虚线过滤
            if ele.get("linetype") is None or ele["linetype"] not in linetype:
                continue
            if ele["type"]=="line":
                e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele["color"])
                segments.append(DSegment(e.start_point,e.end_point,e))
            elif ele["type"] == "arc":
                # 创建DArc对象
                e = DArc(DPoint(ele["center"][0], ele["center"][1]), ele["radius"], ele["startAngle"], ele["endAngle"])
                A = e.start_point.as_tuple()
                B = e.end_point.as_tuple()
                O = e.center.as_tuple()
                
                # 计算角度差
                start_angle = ele["startAngle"]
                end_angle = ele["endAngle"]
                if end_angle<start_angle:
                    end_angle=end_angle+360
                total_angle = end_angle - start_angle
                
                # 定义分段数量，可以根据角度总长动态决定（这里的分段数可以自行调整）
                num_segments = max(2, int(total_angle / 45))  # 每45度一个分段
                step_angle = total_angle / num_segments  # 每个分段的角度

                # 生成多段线段
                for i in range(num_segments):
                    # 计算起点和终点的角度
                    angle1 = start_angle + i * step_angle
                    angle2 = start_angle + (i + 1) * step_angle

                    # 计算每个角度对应的点
                    x1 = O[0] + e.radius * math.cos(math.radians(angle1))
                    y1 = O[1] + e.radius * math.sin(math.radians(angle1))
                    x2 = O[0] + e.radius * math.cos(math.radians(angle2))
                    y2 = O[1] + e.radius * math.sin(math.radians(angle2))

                    # 创建线段并加入segments列表
                    start_point = DPoint(x1, y1)
                    end_point = DPoint(x2, y2)
                    segments.append(DSegment(start_point, end_point, e))

                

            elif ele["type"]=="lwpolyline" or ele["type"]=="polyline":
                vs = ele["vertices"]
                ps = [DPoint(v[0], v[1]) for v in vs]

                # Apply line simplification
                simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level

                e = DLwpolyline(simplified_ps, ele["color"], ele["isClosed"])
                l = len(simplified_ps)
                for i in range(l - 1):
                    segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                if ele["isClosed"]:
                    segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
            else:
                pass
            if e is not None:
                elements.append(e)
        return elements,segments
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")

#expand lines by fixed length
def expandFixedLength(segList,dist):


    new_seglist=[] 
    for seg in segList:
        p1=seg[0]
        p2=seg[1]
        v=(p2[0]-p1[0],p2[1]-p1[1])
        l=math.sqrt(v[0]*v[0]+v[1]*v[1])
        if l<=dist:
            continue
        v=(v[0]/l*dist,v[1]/l*dist)
        new_seglist.append(DSegment(DPoint(p1[0]-v[0],p1[1]-v[1]),DPoint(p2[0]+v[0],p2[1]+v[1]),seg.ref))
    return new_seglist

#remove duplicate points on the same edge
def remove_duplicates(input_list):  
    seen = set()  
    result = []  
    for item in input_list:  
        if item not in seen:  
            seen.add(item)  
            result.append(item)  
    return result

# Helper function to compute intersection between two segments
def segment_intersection(p1, p2, q1, q2, epsilon=1e-9):
    """ Returns the intersection point between two line segments, or None if they don't intersect. """
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    r = (p2.x - p1.x, p2.y - p1.y)
    s = (q2.x - q1.x, q2.y - q1.y)

    denominator = cross_product(r, s)

    if abs(denominator) < epsilon:  # Parallel or collinear
        return None

    t = cross_product((q1.x - p1.x, q1.y - p1.y), s) / denominator
    u = cross_product((q1.x - p1.x, q1.y - p1.y), r) / denominator

    if 0 <= t <= 1 and 0 <= u <= 1:  # Intersection occurs within both segments
        intersect_x = p1.x + t * r[0]
        intersect_y = p1.y + t * r[1]
        return DPoint(intersect_x, intersect_y)
    
    return None

# Function to find all intersections
def find_all_intersections(segments, epsilon=1e-9):
    intersection_dict = {}
    for i, seg1 in enumerate(segments):
        for j, seg2 in enumerate(segments):
            if i >= j:
                continue  # Avoid duplicate checks and self-intersections
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2, epsilon)
            if intersection:
                if seg1 not in intersection_dict:
                    intersection_dict[seg1] = []
                if seg2 not in intersection_dict:
                    intersection_dict[seg2] = []
                
                # Append the intersection for both segments
                intersection_dict[seg1].append(intersection)
                intersection_dict[seg2].append(intersection)

    # Sort intersections along each segment by their distance from the start point
    for seg, isects in intersection_dict.items():
        isects.sort(key=lambda p: (p.x - seg.start_point.x)**2 + (p.y - seg.start_point.y)**2)

    return intersection_dict

# filter polys by area of polys's bounding boxes 
def filterPolys(polys,t=100):
    filtered_polys=[]
    for poly in polys:
        x_min,x_max=poly[0].start_point.x,poly[0].start_point.x
        y_min,y_max=poly[0].start_point.y,poly[0].start_point.y
        if x_min>poly[0].end_point.x:
            x_min=poly[0].end_point.x
        if x_max< poly[0].end_point.x:
            x_max=poly[0].end_point.x
        if y_min>poly[0].end_point.y:
            y_min=poly[0].end_point.y
        if y_max<poly[0].end_point.y:
            y_max=poly[0].end_point.y
        for i in range(len(poly)):
            if i==0:
                continue
            edge=poly[i]
            if x_min>edge.start_point.x:
                x_min=edge.start_point.x
            if x_max< edge.start_point.x:
                x_max=edge.start_point.x
            if y_min>edge.start_point.y:
                y_min=edge.start_point.y
            if y_max<edge.start_point.y:
                y_max=edge.start_point.y

            if x_min>edge.end_point.x:
                x_min=edge.end_point.x
            if x_max< edge.end_point.x:
                x_max=edge.end_point.x
            if y_min>edge.end_point.y:
                y_min=edge.end_point.y
            if y_max<edge.end_point.y:
                y_max=edge.end_point.y

        area=(y_max-y_min)*(x_max-x_min)
        if area>t:
            filtered_polys.append(poly)
    return filtered_polys


from collections import deque

def split_segments(segments, intersections): 
    """根据交点将线段分割并构建 edge_map"""
    new_segments = []
    edge_map = {}

    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        inter_points = sorted(inter_points, key=lambda p: (p.x, p.y))
        points = [seg.start_point] + inter_points + [seg.end_point]
        
        # 将线段分割为多个部分
        for i in range(len(points) - 1):
            # 比较两个点的大小，确保起点较小
            start_point, end_point = points[i], points[i+1]
            if (end_point.x, end_point.y) < (start_point.x, start_point.y):
                start_point, end_point = end_point, start_point
            
            # 创建新线段
            new_seg = DSegment(start_point, end_point, seg.ref)
            new_segments.append(new_seg)
            
            # 添加到 edge_map 中，包含正向和反向的线段
            edge_map[DSegment(start_point, end_point)] = new_seg
            edge_map[DSegment(end_point, start_point)] = new_seg  # 反向线段
    
    return new_segments, edge_map

def build_graph(segments):
    """根据分割后的线段构建图，保存每条边及其引用的ref"""
    graph = {}

    for seg in segments:
        p1, p2 = seg.start_point, seg.end_point
        
        if p1 not in graph:
            graph[p1] = []
        if p2 not in graph:
            graph[p2] = []
        
        # 保存连接的点以及线段引用信息（ref）
        graph[p1].append((p2, seg.ref))
        graph[p2].append((p1, seg.ref))

    return graph


def bfs_paths(graph, start_point, end_point):
    """基于广度优先搜索找到所有从start_point到end_point的路径，返回路径中的Dsegment"""
    queue = deque([(start_point, [start_point], [])])  # (当前点，路径中的点，路径中的线段)
    all_paths = []

    while queue:
        (current_point, point_path, seg_path) = queue.popleft()

        for neighbor, ref in graph.get(current_point, []):
            if neighbor not in point_path:
                new_point_path = point_path + [neighbor]
                new_seg = DSegment(current_point, neighbor, ref)  # 构建对应的Dsegment
                new_seg_path = seg_path + [new_seg]

                if neighbor == end_point:
                    if len(new_point_path) > 2:  # 避免 trivial paths
                        all_paths.append(new_seg_path)
                else:
                    queue.append((neighbor, new_point_path, new_seg_path))

    return all_paths

def compute_arc_replines(new_segments):
    """
    计算arc_replines，筛选出ref是弧线且半径在20~160之间的线段。
    :param new_segments: 分割后的线段列表
    :return: arc_replines 列表
    """
    arc_replines = []

    for segment in new_segments:
        # 检查线段的ref是否是弧线，且半径在 20 到 160 之间
        if isinstance(segment.ref, DArc):
            radius = segment.ref.radius
            if 20 <= radius <= 160:
                arc_replines.append(segment)

    return arc_replines

def convert_ref_to_tuple(ref):
    """
    Convert different ref objects (Line, Arc, etc.) to a unified tuple format for comparison.
    """
    if isinstance(ref, DLine):
        # Convert Line to a tuple of its start and end points
        return ('Line', (ref.start_point.x, ref.start_point.y), (ref.end_point.x, ref.end_point.y))
    elif isinstance(ref, DArc):
        # Convert Arc to a tuple of its defining properties: center, radius, start_angle, end_angle
        return ('Arc', (ref.center.x, ref.center.y), ref.radius, ref.start_angle, ref.end_angle)
    else:
        # Handle other types or return a generic string for unknown types
        return ('Unknown', str(ref))

def remove_duplicate_polygons(closed_polys):
    """
    Remove duplicate closed_polys based on the set of unique 'ref' values for each polygon.
    :param closed_polys: List of polygons, where each polygon is a list of Segment objects
    :return: Deduplicated list of polygons
    """
    unique_polys = []
    seen_refs = set()  # To store unique sets of refs for comparison

    for poly in closed_polys:
        # Convert all ref objects in the polygon to a unified comparable format
        refs = {convert_ref_to_tuple(seg.ref) for seg in poly}

        # If this set of refs is unique, add the polygon to the result
        refs_tuple = tuple(sorted(refs))  # Sorting to ensure consistent order for comparison

        if refs_tuple not in seen_refs:
            unique_polys.append(poly)
            seen_refs.add(refs_tuple)

    return unique_polys

# filter polys by area of polys's bounding boxes 
def filterPolys(polys,t=100):
    filtered_polys=[]
    for poly in polys:
        x_min,x_max=poly[0].start_point.x,poly[0].start_point.x
        y_min,y_max=poly[0].start_point.y,poly[0].start_point.y
        if x_min>poly[0].end_point.x:
            x_min=poly[0].end_point.x
        if x_max< poly[0].end_point.x:
            x_max=poly[0].end_point.x
        if y_min>poly[0].end_point.y:
            y_min=poly[0].end_point.y
        if y_max<poly[0].end_point.y:
            y_max=poly[0].end_point.y
        for i in range(len(poly)):
            if i==0:
                continue
            edge=poly[i]
            if x_min>edge.start_point.x:
                x_min=edge.start_point.x
            if x_max< edge.start_point.x:
                x_max=edge.start_point.x
            if y_min>edge.start_point.y:
                y_min=edge.start_point.y
            if y_max<edge.start_point.y:
                y_max=edge.start_point.y

            if x_min>edge.end_point.x:
                x_min=edge.end_point.x
            if x_max< edge.end_point.x:
                x_max=edge.end_point.x
            if y_min>edge.end_point.y:
                y_min=edge.end_point.y
            if y_max<edge.end_point.y:
                y_max=edge.end_point.y

        area=(y_max-y_min)*(x_max-x_min)
        if area>t:
            filtered_polys.append(poly)
    return filtered_polys

# 保留简单路径的多边形
def points_are_close(p1, p2, tolerance=1e-3):
    """
    判断两个点是否在一定误差范围内相等。
    :param p1: 第一个点 (DPoint)
    :param p2: 第二个点 (DPoint)
    :param tolerance: 容差（允许的误差范围）
    :return: 如果两个点的距离小于容差，则返回 True
    """
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5 < tolerance

def remove_complicated_polygons(polys, tolerance=1e-5):
    """
    去除复杂路径的多边形，确保每个点最多被访问两次（容差允许的范围内判断点相同）。
    :param polys: 多边形列表
    :param tolerance: 容差（允许的误差范围）
    :return: 保留简单路径的多边形列表
    """
    res = []

    for poly in polys:
        count = {}
        flag = True
        for seg in poly:
            # 判断start_point是否已存在于count中（基于误差范围判断）
            start_exists = False
            end_exists = False

            for point in count.keys():
                if points_are_close(seg.start_point, point, tolerance):
                    start_exists = True
                    count[point] += 1
                    if count[point] > 2:
                        flag = False
                    break

            if not start_exists:
                count[seg.start_point] = 1

            # 判断end_point是否已存在于count中（基于误差范围判断）
            for point in count.keys():
                if points_are_close(seg.end_point, point, tolerance):
                    end_exists = True
                    count[point] += 1
                    if count[point] > 2:
                        flag = False
                    break

            if not end_exists:
                count[seg.end_point] = 1

        if flag:
            res.append(poly)

    return res


def findClosedPolys_via_BFS(segments, drawIntersections=False, linePNGPath="./line.png", drawPolys=False, polyPNGPath="./poly.png"):
    # compute intersections using the improved method
    isecDic = find_all_intersections(segments)

    # filter and remove duplicates
    for seg, isects in isecDic.items():
        isecDic[seg] = remove_duplicates(isects)


    # Step 2: 根据交点分割线段
    new_segments, edge_map = split_segments(segments, isecDic)

    # Step 3: 构建基于分割后线段的图结构
    graph= build_graph(new_segments)

    closed_polys = []

    # 基于角隅孔计算参考边
    arc_replines = compute_arc_replines(new_segments)

    # Step 4: 对每个 arc_repline，使用 BFS 查找路径
    for arc_repline in arc_replines:
        start_point = arc_repline.start_point
        end_point = arc_repline.end_point

        # 使用 BFS 查找从 start_point 到 end_point 的所有路径
        paths = bfs_paths(graph, start_point, end_point)

        # 将找到的路径添加到 closed_polys
        closed_polys.extend(paths)

    if drawIntersections:
        #plot original segments
        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k:')

        # plot intersection points
        for seg, isects in isecDic.items():
            for p in isects:
                plt.plot(p.x, p.y, 'r.')

        plt.gca().axis('equal')
        plt.savefig(linePNGPath)

    # from plot_geo import plot_polys
    # plot_polys(new_segments, "./output/newpoly_lines")

    # 剔除重复路径
    polys = remove_duplicate_polygons(closed_polys)

    # 根据边框对多边形进行过滤
    polys = filterPolys(polys,t=3000)

    # 仅保留基本路径
    polys = remove_complicated_polygons(polys)
    
    print(len(polys))
    return polys