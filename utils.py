import json 
from  element import *
import math
from SweepIntersectorLib.SweepIntersector import SweepIntersector
from plot_geo import plot_geometry,plot_polys
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import deque
import os
from sklearn.cluster import DBSCAN


def is_point_in_polygon(point, polygon_edges):
    """
    判断一个点是否在一组线段围成的多边形内

    :param point: (x, y) 要判断的点
    :param polygon_edges: [(x1, y1, x2, y2), ...] 多边形边的列表，每条边由两个点的坐标表示
    :return: True 如果点在多边形内，否则 False
    """
    x, y = point.x,point.y
    n = len(polygon_edges)
    inside = False

    # 遍历多边形的每一条边
    for i in range(n):
        x1, y1, x2, y2 = polygon_edges[i].start_point.x,polygon_edges[i].start_point.y,polygon_edges[i].end_point.x,polygon_edges[i].end_point.y
        # 计算点与边两个端点的向量叉积
        cross_product = (x - x1) * (y2 - y1) - (x2 - x1) * (y - y1)
        # 判断点相对于边的位置
        if cross_product > 0:  # 点在边的左侧
            inside = not inside
        elif cross_product == 0 and (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1):
            # 点在边上（这里可以根据需要决定是否将边上的点视为在内或在外）
            # 本例中，我们假设边上的点不在多边形内
            return True

    # 如果经过所有边后，inside为True，则点在多边形内；否则在多边形外
    # 注意：这里假设多边形是封闭的，即最后一条边与第一条边相连
    # 如果多边形不是封闭的，需要额外处理
    return inside



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


def coordinatesmap(p:DPoint,insert,scales,rotation):
    # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
    x,y=(p[0]*scales[0])+insert[0],(p[1]*scales[1])+insert[1]
    return DPoint(x,y)
#json --> elements
def readJson(path):
    elements=[]
    segments=[]
    color = [3, 7, 4]
    linetype = ["BYLAYER", "Continuous"]
    elementtype=["line","arc","lwpolyline","polyline"]
    try:  
        with open(path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        for ele in data_list[0]:
            if ele["type"]=="line":
                # 颜色过滤
                if ele["color"] not in color:
                    continue
                # 虚线过滤
                if ele.get("linetype") is None or ele["linetype"] not in linetype:
                    continue
                e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele["color"])
                elements.append(e)
                segments.append(DSegment(e.start_point,e.end_point,e))
            elif ele["type"] == "arc":
                # 颜色过滤
                if ele["color"] not in color:
                    continue
                # 虚线过滤
                if ele.get("linetype") is None or ele["linetype"] not in linetype:
                    continue
                # 创建DArc对象
                e = DArc(DPoint(ele["center"][0], ele["center"][1]), ele["radius"], ele["startAngle"], ele["endAngle"])
                elements.append(e)
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
                    # if start_point.y>-48500 or end_point.y>-48500:
                    segments.append(DSegment(start_point, end_point, e))
            elif ele["type"]=="lwpolyline" or ele["type"]=="polyline":
                # 颜色过滤
                if ele["color"] not in color:
                    continue
                # 虚线过滤
                if ele.get("linetype") is None or ele["linetype"] not in linetype:
                    continue
                vs = ele["vertices"]
                ps = [DPoint(v[0], v[1]) for v in vs]

                # Apply line simplification
                simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level

                e = DLwpolyline(simplified_ps, ele["color"], ele["isClosed"])
                elements.append(e)
                l = len(simplified_ps)
                for i in range(l - 1):
                    # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                    segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                if ele["isClosed"]:
                    # if simplified_ps[-1].y>-48500 or simplified_ps[0].y>-48500:
                    segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
            elif ele["type"]=="insert":
                # 颜色过滤
                if ele["color"] not in color:
                    continue
                blockName=ele["blockName"]
                # x1,x2,y1,y2=ele["bound"]["x1"],ele["bound"]["x2"],ele["bound"]["y1"],ele["bound"]["y2"]
                scales=ele["scales"]
                rotation=ele["rotation"]
                insert=ele["insert"]
                block_data=data_list[1][blockName]
                for sube in block_data:
                    if sube["type"] not in elementtype or sube["color"] !=3  or sube.get("linetype") is None or sube["linetype"] not in linetype:
                        continue
                    if sube["type"]=="line":
                        e=DLine(coordinatesmap(DPoint(sube["start"][0],sube["start"][1]),insert,scales,rotation),
                        coordinatesmap(DPoint(sube["end"][0],sube["end"][1]),insert,scales,rotation)
                        ,sube["color"])
                        elements.append(e)
                        segments.append(DSegment(e.start_point,e.end_point,e))
                    elif sube["type"] == "arc":
                        # 创建DArc对象
                        e = DArc(coordinatesmap(DPoint(sube["center"][0], sube["center"][1]),insert,scales,rotation),
                         sube["radius"], sube["startAngle"], sube["endAngle"])
                        elements.append(e)
                        A = e.start_point.as_tuple()
                        B = e.end_point.as_tuple()
                        O = e.center.as_tuple()
                        
                        # 计算角度差
                        start_angle = sube["startAngle"]
                        end_angle = sube["endAngle"]
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
                    elif sube["type"]=="lwpolyline" or sube["type"]=="polyline":
                        vs = sube["vertices"]
                        ps = [coordinatesmap(DPoint(v[0], v[1]),insert,scales,rotation) for v in vs]

                        # Apply line simplification
                        simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level
                        e = DLwpolyline(simplified_ps, sube["color"], sube["isClosed"])
                        elements.append(e)
                        l = len(simplified_ps)
                        for i in range(l - 1):
                            segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
                        if sube["isClosed"]:
                            segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
            elif ele["type"]=="text":
                e=DText(ele["bound"],ele["insert"], ele["color"],ele["content"],ele["height"])
                elements.append(e)
            else:
                pass
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

def split_segments(segments, intersections,epsilon=0.25): 
    """根据交点将线段分割并构建 edge_map"""
    new_segments = []
    edge_map = {}
    point_map={}

    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        inter_points = sorted([seg.start_point] + inter_points + [seg.end_point], key=lambda p: (p.x, p.y))
        points=inter_points
        segList=[]
        
        # 将线段分割为多个部分
        for i in range(len(points) - 1):
            # 比较两个点的大小，确保起点较小
            start_point, end_point = points[i], points[i+1]
            if (end_point.x, end_point.y) < (start_point.x, start_point.y):
                start_point, end_point = end_point, start_point
            if ((start_point.x-end_point.x)*(start_point.x-end_point.x)+(start_point.y-end_point.y)*(start_point.y-end_point.y)) < epsilon:
                continue
            # 创建新线段:
            new_seg = DSegment(start_point, end_point, seg.ref)
            segList.append(new_seg)


        for s in segList:
            new_segments.append(s)
            # 添加到 edge_map 中，包含正向和反向的线段
            if DSegment(s.start_point, s.end_point) not in edge_map:
                edge_map[DSegment(s.start_point, s.end_point)] = s
            if DSegment(s.end_point, s.start_point) not in edge_map:
                edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段

    for s in new_segments:
        vs,ve=s.start_point,s.end_point
        if vs not in point_map:
            point_map[vs]=set()
        if ve not in point_map:
            point_map[ve]=set()
        point_map[vs].add(s)
        point_map[ve].add(s)

    return new_segments, edge_map,point_map


def filter_segments(segments,intersections,point_map,expansion_param=12,iters=3):
    new_segments=[]
    edge_map = {}
    new_point_map={}
    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        inter_points = sorted([seg.start_point] + inter_points + [seg.end_point], key=lambda p: (p.x, p.y))
        points=inter_points
        segList=[]
        
        # 将线段分割为多个部分
        for i in range(len(points) - 1):
            # 比较两个点的大小，确保起点较小
            start_point, end_point = points[i], points[i+1]
            if (end_point.x, end_point.y) < (start_point.x, start_point.y):
                start_point, end_point = end_point, start_point
            if ((start_point.x-end_point.x)*(start_point.x-end_point.x)+(start_point.y-end_point.y)*(start_point.y-end_point.y)) < expansion_param*expansion_param:
                continue
   
            new_seg = DSegment(start_point, end_point, seg.ref)
            segList.append(new_seg)


        #filter lines
        if len(segList)>1:
            for s in segList:
                new_segments.append(s)
                # 添加到 edge_map 中，包含正向和反向的线段
                if DSegment(s.start_point, s.end_point) not in edge_map:
                    edge_map[DSegment(s.start_point, s.end_point)] = s
                if DSegment(s.end_point, s.start_point) not in edge_map:
                    edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        elif len(segList)==1:
            if isinstance(segList[0].ref, DArc):
                #arc
                new_segments.append(segList[0])
                s=segList[0]
                if DSegment(s.start_point, s.end_point) not in edge_map:
                    edge_map[DSegment(s.start_point, s.end_point)] = s
                if DSegment(s.end_point, s.start_point) not in edge_map:
                    edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
            else:
                vs,ve=segList[0].start_point,segList[0].end_point
                s_deg,e_deg=len(point_map[vs]),len(point_map[ve])
                if s_deg==1 or e_deg==1:
                    continue
                else:
                    new_segments.append(segList[0])
                    s=segList[0]
                    if DSegment(s.start_point, s.end_point) not in edge_map:
                        edge_map[DSegment(s.start_point, s.end_point)] = s
                    if DSegment(s.end_point, s.start_point) not in edge_map:
                        edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段


    for s in new_segments:
        vs,ve=s.start_point,s.end_point
        if vs not in new_point_map:
            new_point_map[vs]=set()
        if ve not in new_point_map:
            new_point_map[ve]=set()
        new_point_map[vs].add(s)
        new_point_map[ve].add(s)
    for p,ss in new_point_map.items():
        new_point_map[p]=list(ss)
    

 
    for i in range(iters):
        filtered_segments=[]
        filtered_edge_map={}
        filtered_point_map={}

        for s in new_segments:
            vs,ve=s.start_point,s.end_point
            div_s,div_e=len(new_point_map[vs]),len(new_point_map[ve])
            if div_s==1 or div_e==1:
                continue
            filtered_segments.append(s)
            if DSegment(s.start_point, s.end_point) not in filtered_edge_map:
                filtered_edge_map[DSegment(s.start_point, s.end_point)] = s
            if DSegment(s.end_point, s.start_point) not in filtered_edge_map:
                filtered_edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        for s in filtered_segments:
            vs,ve=s.start_point,s.end_point
            if vs not in filtered_point_map:
                filtered_point_map[vs]=set()
            if ve not in filtered_point_map:
                filtered_point_map[ve]=set()
            filtered_point_map[vs].add(s)
            filtered_point_map[ve].add(s)
        for p,ss in filtered_point_map.items():
            filtered_point_map[p]=list(ss)
        new_segments=filtered_segments
        edge_map=filtered_edge_map
        new_point_map=filtered_point_map
    return new_segments,edge_map,new_point_map
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
                    if len(new_point_path) > 1:  # 避免 trivial paths
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
            if 20 <= radius and radius <= 160:
                arc_replines.append(segment)


    return arc_replines

def compute_star_replines(new_segments,elements):
    vertical_lines=[]
    star_replines=[]
    for e in elements:
        if isinstance(e,DText) and e.content=="*":
            x,y=e.insert[0],e.insert[1]
            vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
    for i, seg1 in enumerate(vertical_lines):
        y_min=None
        s=None
        for j, seg2 in enumerate(new_segments):
            if i >= j:
                continue  # Avoid duplicate checks and self-intersections
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_min is None:
                    y_min=intersection[1]
                    s=seg2
                else:
                    if y_min>intersection[1]:
                        y_min=intersection[1]
                        s=seg2
        if s is not None:
            star_replines.append(s)
    return star_replines

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


def computeCenterCoordinates(poly):

    w=0
    x=0
    y=0
    for edge in poly:
        # if not isinstance(edge.ref,DArc):
            # l=edge.ref.weight
            # w=w+l
            # x=x+l*edge.ref.bc.x
            # y=y+l*edge.ref.bc.y
        l=edge.length()
        w+=l
        x+=l*(edge.start_point.x+edge.end_point.x)/2
        y+=l*(edge.start_point.y+edge.end_point.y)/2
    return DPoint(x/w,y/w)


#count the number of replines in a poly
def countReplines(poly,replines_set):
    count=0
    for edge in poly:
        if edge in replines_set:
            count+=1
    return count


def remove_duplicate_polygons(closed_polys,replines_set,eps=25.0,min_samples=1):
    """
    Remove duplicate closed_polys based on the set of unique 'ref' values for each polygon.
    :param closed_polys: List of polygons, where each polygon is a list of Segment objects
    :return: Deduplicated list of polygons
    """
    if len(closed_polys)==0:
        return []
    unique_polys = []
    bcs=[]

    # seen_refs = set()  # To store unique sets of refs for comparison

    # for poly in closed_polys:
    #     # Convert all ref objects in the polygon to a unified comparable format
    #     refs = {convert_ref_to_tuple(seg.ref) for seg in poly}

    #     # If this set of refs is unique, add the polygon to the result
    #     refs_tuple = tuple(sorted(refs))  # Sorting to ensure consistent order for comparison

    #     if refs_tuple not in seen_refs:
    #         unique_polys.append(poly)
    #         seen_refs.add(refs_tuple)
    poly_map={}
    for poly in closed_polys:
        bc=computeCenterCoordinates(poly)
        #bc=DPoint(int(bc.x/span)*span,int(bc.y/span)*span)
        if bc not in poly_map:
            poly_map[bc]=poly
        else:
            if len(poly_map[bc])<len(poly):
                poly_map[bc]=poly
    tps=[]
    for bc,poly in poly_map.items():
        # unique_polys.append(poly)
        # bcs.append(bc)
        tps.append((bc,poly))
    # tps=sorted(tps,key=lambda item: item[0].as_tuple())
    
    #filter tps
    
    points =[]
    replines_count=[]
    for tp in tps:
        points.append([tp[0].x,tp[0].y])
        replines_count.append(countReplines(tp[1],replines_set))
    #print(replines_count)
    points=np.array(points)


    # eps: 两点之间的最大距离，可以认为是邻居的距离。这个值可以根据你的数据集进行调整。
    # min_samples: 一个点被认为是核心点所需的最小邻居数。
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)

    labels = db.labels_

    poly_map={}
    #print("聚类标签：", labels)
    for idx,label in enumerate(labels):
        if label ==-1:
            unique_polys.append(tps[idx][1])
        else:
            if label not in poly_map:
                poly_map[label]=idx
            else:
                if replines_count[poly_map[label]]<replines_count[idx]:
                    #print(len(poly_map[label]),len(tps[idx][1]))
                    poly_map[label]=idx
    
    for label,idx in poly_map.items():
        unique_polys.append(tps[idx][1])
    
    return unique_polys


# filter polys by area of polys's bounding boxes 
def filterPolys(polys,arc_replines_set,t=100,d=5):
    filtered_polys=[]
    for poly in polys:
        if len(poly)>15:
            continue
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
        yy=y_max-y_min
        xx=x_max-x_min
        if yy<20 or xx<20:
            continue
        if yy>xx:
            div=yy/xx
        else:
            div=xx/yy
        if area>t and div<d:
            filtered_polys.append(poly)
    for poly in filtered_polys:
        l=len(poly)
        for i in range(l-1):
            if poly[i].start_point==poly[i+1].start_point:
                p=poly[i].start_point
                poly[i].start_point=poly[i].end_point
                poly[i].end_point=p
            elif poly[i].start_point==poly[i+1].end_point:
                p=poly[i].start_point
                poly[i].start_point=poly[i].end_point
                poly[i].end_point=p
                p=poly[i+1].start_point
                poly[i+1].start_point=poly[i+1].end_point
                poly[i+1].end_point=p
            elif poly[i].end_point==poly[i+1].end_point:
                p=poly[i+1].start_point
                poly[i+1].start_point=poly[i+1].end_point
                poly[i+1].end_point=p
        if poly[-1].start_point==poly[0].start_point:
            p=poly[-1].start_point
            poly[-1].start_point=poly[-1].end_point
            poly[-1].end_point=p
        elif poly[-1].start_point==poly[0].end_point:
            p=poly[-1].start_point
            poly[-1].start_point=poly[-1].end_point
            poly[-1].end_point=p
            p=poly[0].start_point
            poly[0].start_point=poly[0].end_point
            poly[0].end_point=p
        elif poly[-1].end_point==poly[0].end_point:
            p=poly[0].start_point
            poly[0].start_point=poly[0].end_point
            poly[0].end_point=p
    print(filtered_polys[0])
    valid_polys=[]

    for poly in filtered_polys:
        valid=True
        for edge in poly:
            if edge in arc_replines_set:
                o=DPoint((edge.ref.start_point.x+edge.ref.end_point.x)/2,
                  (edge.ref.start_point.y+edge.ref.end_point.y)/2)
                if  is_point_in_polygon(o,poly):
                    valid=False
                    break
        if  valid:
            valid_polys.append(poly)

    return valid_polys

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

def outputLines(segments,point_map,linePNGPath,drawIntersections=False,drawLines=False):
    if drawLines:
        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k-')
    if drawIntersections:
        for p,ss in point_map.items():
            if len(ss)>1:
                # print(p.x,p.y)
                plt.plot(p.x, p.y, 'r.')

    plt.gca().axis('equal')
    plt.savefig(linePNGPath)
    print(f"直线图保存于:{linePNGPath}")

def outputPolysAndGeometry(polys,path,draw_polys=False,draw_geometry=False,n=10):
    if draw_geometry:
        for i,poly in enumerate(polys):
            if i>=n:
                break
            plot_geometry(poly,os.path.join(path,f"geometry{i}.png"))
    
    if draw_polys:
        for i,poly in enumerate(polys):
            if i>=n:
                break
            plot_polys(poly,os.path.join(path,f"poly{i}.png"))
    print(f"封闭多边形图像保存于:{path}")


def findClosedPolys_via_BFS(elements,segments,segmentation_config):
    verbose=segmentation_config.verbose
    # Step 1: 计算交点
    if verbose:
        print("计算交点")
    isecDic = find_all_intersections(segments,segmentation_config.intersection_epsilon)

    # Step 2: 根据交点分割线段
    if verbose:
        print("根据交点分割线段")
    new_segments, edge_map,point_map= split_segments(segments, isecDic,segmentation_config.segment_split_epsilon)
    #filter lines

    new_segments, edge_map,point_map= filter_segments(segments,isecDic,point_map,segmentation_config.segment_filter_length,segmentation_config.segment_filter_iters)


    outputLines(new_segments,point_map,segmentation_config.line_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments)


    # Step 3: 构建基于分割后线段的图结构
    if verbose:
        print("构建基于分割后线段的图结构")
        print(f"过滤后线段条数:{len(new_segments)}")
    graph= build_graph(new_segments)

    closed_polys = []

    # 基于角隅孔计算参考边
    arc_replines = compute_arc_replines(new_segments)
    star_replines=compute_star_replines(new_segments,elements)
    if verbose:
        print(f"圆弧角隅孔个数: {len(arc_replines)}")
        print(f"星形角隅孔个数: {len(star_replines)}")
    # for star_repline in star_replines:
    #     print(star_repline)
    replines=arc_replines+star_replines
    replines_set=set()
    arc_replines_set=set()
    for arc_rep in arc_replines:
        if arc_rep not in arc_replines_set:
            arc_replines_set.add(arc_rep)
            arc_replines_set.add(DSegment(arc_rep.end_point,arc_rep.start_point))
    for rep in replines:
        if rep not in replines_set:
            replines_set.add(rep)
            replines_set.add(DSegment(rep.end_point,rep.start_point))
    # Step 4: 对每个 repline，使用 BFS 查找路径
    if verbose:
        print("查找闭合路径")
    for repline in replines:
        start_point = repline.start_point
        end_point = repline.end_point

        # 使用 BFS 查找从 start_point 到 end_point 的所有路径
        paths = bfs_paths(graph, start_point, end_point)

        # 构成闭合路径
        for path in paths:
            path.append(repline)

        # 将找到的路径添加到 closed_polys
        closed_polys.extend(paths)
    if verbose:
        print("查找完毕")
    # for closed_poly in closed_polys:
    #     print(closed_poly)
    
    #poly simplify


    # 根据边框对多边形进行过滤
    #polys = filterPolys(polys,t=3000,d=5)
    polys = filterPolys(closed_polys,arc_replines_set,segmentation_config.bbox_area,segmentation_config.bbox_ratio)

    # 剔除重复路径
    polys = remove_duplicate_polygons(polys,replines_set,segmentation_config.eps,segmentation_config.min_samples)
    #print(bcs)
    # print(polys[0])
    # print(polys[1])
    # 仅保留基本路径
    polys = remove_complicated_polygons(polys,segmentation_config.remove_tolerance)
    if verbose:
        print(f"封闭多边形个数:{len(polys)}")
    outputPolysAndGeometry(polys,segmentation_config.poly_image_dir,segmentation_config.draw_polys,segmentation_config.draw_geometry,segmentation_config.draw_poly_nums)
    return polys