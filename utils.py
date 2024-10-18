import json 
from  element import *
import math
from SweepIntersectorLib.SweepIntersector import SweepIntersector
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
            if ele["color"] not in color:
                continue
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

# Updated function to find closed polygons
def findClosedPolys(segments, drawIntersections=False, linePNGPath="./line.png", drawPolys=False, polyPNGPath="./poly.png"):
    # compute intersections using the improved method
    isecDic = find_all_intersections(segments)

    # filter and remove duplicates
    for seg, isects in isecDic.items():
        isecDic[seg] = remove_duplicates(isects)

    # find all the edges
    edge_set = set()
    for seg, isects in isecDic.items():
        l = len(isects)
        for i in range(l - 1):
            edge_set.add(DSegment(isects[i], isects[i + 1], seg.ref))

    edge_list = list(edge_set)

    # map edges to their segment pairs
    edge_map = {}
    for e in edge_list:
        edge_map[DSegment(e.start_point, e.end_point)] = e
        edge_map[DSegment(e.end_point, e.start_point)] = e

    # Creating a graph to find cycles (closed paths)
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Find all basic cycles (closed polygons)
    cycles = nx.cycle_basis(G)
    
    polys = []
    print("before filtering:")
    print("Found cycles (closed paths):")
    for cycle in cycles:
        poly = []
        l = len(cycle)
        print(f"vertex count: {l}")
        for i in range(l - 1):
            poly.append(edge_map[DSegment(cycle[i], cycle[i + 1])])
        poly.append(edge_map[DSegment(cycle[-1], cycle[0])])
        polys.append(poly)
    print("after filtering:")
    filtered_polys=filterPolys(polys,t=3000)
    # print(filtered_polys[3])
    for poly in filtered_polys:
        l=len(poly)
        print(f"vertex count: {l}")
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

    if drawPolys:
        # for poly in polys:
        #     for e in poly:
        #         plt.plot([e[0][0],e[1][0]],[e[0][1],e[1][1]],'k:')
        pass

    return filtered_polys

