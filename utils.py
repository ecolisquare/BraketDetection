import json 
from  element import *
import math
from plot_geo import plot_geometry,plot_polys, plot_info_poly
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import os
from sklearn.cluster import DBSCAN
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed,TimeoutError
import time
from functools import partial
from itertools import combinations
from bracket_parameter_extraction import *

def numberInString(content):
    flag=False
    for i in range(10):
        if str(i) in content:
            flag=True
            break
    return flag
# def is_point_in_polygon(point, polygon_edges):
    
#     polygon_points = set()  # Concave polygon example
#     for edge in polygon_edges:
#         vs,ve=edge.start_point,edge.end_point
#         polygon_points.add((vs.x,vs.y))
#         polygon_points.add((ve.x,ve.y))

#     polygon_points = list(polygon_points)

#     polygon = Polygon(polygon_points)

#     point = Point(point.x, point.y)

#     # Check if the point is inside the polygon
#     is_inside = polygon.contains(point)

#     return is_inside

def computeAreaOfPoly(poly):
    if len(poly)<3:
        return 0.0
    points=[]
    for edge in poly:
        points.append(edge.start_point)
    area=0.0
    for i in range(len(poly)-1):
        p1=points[i]
        p2=points[i+1]
        triArea=(p1.x*p2.y-p1.y*p2.x)/2
        area+=triArea
    p1=points[-1]
    p2=points[0]
    area+=(p1.x*p2.y-p1.y*p2.x)/2
    return math.fabs(area)
    
def are_equal_with_tolerance(a, b, tolerance=1e-6):
    return abs(a - b) < tolerance

def conpute_angle_of_two_segments(seg1,seg2):
   
    dx1 = seg1.end_point.x - seg1.start_point.x
    dy1 = seg1.end_point.y - seg1.start_point.y
    dx2 = seg2.end_point.x - seg2.start_point.x
    dy2 = seg2.end_point.y - seg2.start_point.y

    # 计算两个方向向量的模长
    length1 = math.sqrt(dx1**2 + dy1**2)
    length2 = math.sqrt(dx2**2 + dy2**2)
    
    
    
    # 归一化叉积
    cross_product = math.fabs((dx1 * dy2 - dy1 * dx2) / (length1 * length2))
    if cross_product>1:
        cross_product=1

    return math.asin(cross_product)/math.pi*180
def is_parallel(seg1, seg2, tolerance=0.05):
    """判断两条线段是否平行"""
    dx1 = seg1.end_point.x - seg1.start_point.x
    dy1 = seg1.end_point.y - seg1.start_point.y
    dx2 = seg2.end_point.x - seg2.start_point.x
    dy2 = seg2.end_point.y - seg2.start_point.y

    # 计算两个方向向量的模长
    length1 = math.sqrt(dx1**2 + dy1**2)
    length2 = math.sqrt(dx2**2 + dy2**2)
    
    # 防止长度为0的线段
    if length1 == 0 or length2 == 0:
        return False
    
    # 归一化叉积
    cross_product = (dx1 * dy2 - dy1 * dx2) / (length1 * length2)
    
    # 返回是否接近0
    #print(cross_product)
    return are_equal_with_tolerance(cross_product, 0, tolerance)


def angleOfTwoVectors(A,B):
    lengthA = math.sqrt(A[0]**2 + A[1]**2)  
    lengthB = math.sqrt(B[0]**2 + B[1]**2)  
    dotProduct = A[0] * B[0] + A[1] * B[1]   
    cos_angle=dotProduct / (lengthA * lengthB)
    if math.fabs(cos_angle)>1:
        if cos_angle<0:
            cos_angle=-1
        else:
            cos_angle=1
    angle = math.acos(cos_angle)
    angle_degrees = angle * (180 / math.pi)  
    return angle_degrees

def angleOfTwoSegmentsWithCommonStarter(p,a,b):
    A=a.start_point if p==a.end_point else a.end_point
    B=b.start_point if p==b.end_point else b.end_point
    return angleOfTwoVectors([A.x-p.x,A.y-p.y],[B.x-p.x,B.y-p.y])

# def isParallel(a,b,eps=1.0):
#     dx1=a.end_point.x-a.start_point.x
#     dy1=a.end_point.y-a.start_point.y
#     dx2=b.end_point.x-b.start_point.x
#     dy2=b.end_point.y-b.start_point.y
#     return math.fabs(dx1*dy2-dx2*dy1)<eps
# Ramer-Douglas-Peucker algorithm for line simplification
def rdp(points, epsilon):
    p_set=set()
    ps=[]
    for p in points:
        if p not in p_set:
            p_set.add(p)
            ps.append(p)
    return ps
    # if len(points) < 3:
    #     return points
    # # Find the point with the maximum distance from the line segment (first to last point)
    # start = points[0]
    # end = points[-1]
    # line_vec = np.array([end.x - start.x, end.y - start.y])
    # line_len = np.linalg.norm(line_vec)

    # max_dist = 0
    # index = 0

    # for i in range(1, len(points) - 1):
    #     p = points[i]
    #     vec = np.array([p.x - start.x, p.y - start.y])
    #     proj_len = np.dot(vec, line_vec) / line_len
    #     proj_point = np.array([start.x, start.y]) + proj_len * (line_vec / line_len)
    #     dist = np.linalg.norm([p.x, p.y] - proj_point)

    #     if dist > max_dist:
    #         max_dist = dist
    #         index = i

    # # If the max distance is greater than epsilon, recursively simplify
    # if max_dist > epsilon:
    #     left = rdp(points[:index + 1], epsilon)
    #     right = rdp(points[index:], epsilon)
    #     return left[:-1] + right
    # else:
    #     return [start, end]  

def process_lwpoline(vs,vs_type,color,handle,meta,isClosed,hasArc):
    #print(len(vs_type),len(vs))
    new_vs=[]
    elements=[]
    arcs=[]
    segments=[]
    arc_set=set()
    for i,v in enumerate(vs):
        if len(v)==4:
            new_vs.append([v[0], v[1]])
            new_vs.append([v[2], v[3]])        

        elif len(v)==5:
            x,y,s_a,e_a,r=v[0],v[1],v[2],v[3],v[4]
            s_a=s_a/math.pi*180
            e_a=e_a/math.pi*180
            if s_a>e_a:
                e_a+=360
            x_new = x + r * math.cos(s_a * math.pi / 180)
            y_new = y + r * math.sin(s_a * math.pi / 180)
            start_point=DPoint(x_new,y_new)
            x_new = x + r * math.cos(e_a * math.pi / 180)
            y_new = y + r * math.sin(e_a * math.pi / 180)
            end_point=DPoint(x_new,y_new)
            if len(new_vs)>0:
                prev=DPoint(new_vs[-1][0],new_vs[-1][1])
                l1,l2=DSegment(prev,start_point),DSegment(end_point,prev)
                if l1.length()<l2.length():
                    next,other=start_point,end_point
                else:
                    next,other=end_point,start_point
                if next==prev:
                    if other==prev:
                        pass
                    else:
                        new_vs.append([other.x,other.y])
                else:
                    new_vs.append([next.x,next.y])
                    new_vs.append([other.x,other.y])
            else:
                new_vs.append([start_point.x,start_point.y])
                new_vs.append([end_point.x,end_point.y])
            if r>1000:
                # line=DLine(start_point,end_point,color,handle,meta)
                # elements.append(line)
                # segments.append(DSegment(line.start_point,line.end_point,line))
                pass
            else:
                arc=DArc(DPoint(x,y),r,s_a,e_a,color,handle,meta)
                elements.append(arc)
                arcs.append(arc)
                s1,s2=DSegment(start_point,end_point,None),DSegment(end_point,start_point,None)
                if s1 not in arc_set:
                    arc_set.add(s1)
                if s2 not in arc_set:
                    arc_set.add(s2)
            # total_angle = e_a - s_a
            # num_segments = max(2, int(math.fabs(total_angle) / 45))  # 每45度一个分段
            # if r>10000 or math.fabs(total_angle)<20:
            #     num_segments=1
            # angle_step = total_angle / num_segments
            # print(total_angle,num_segments,s_a,e_a)
            # for j in range(num_segments+1):
            #     angle = s_a + j * angle_step
            #     x_new = x + r * math.cos(angle * math.pi / 180)
            #     y_new = y + r * math.sin(angle * math.pi / 180)
            #     new_vs.append([x_new,y_new])
    for i in range(len(new_vs)-1):
        s,e=DPoint(new_vs[i][0],new_vs[i][1]),DPoint(new_vs[i+1][0],new_vs[i+1][1])
        seg=DSegment(s,e,None)
        if seg.length()>0 and seg not in arc_set:
            line=DLine(s,e,color,handle,meta)
            segments.append(DSegment(line.start_point,line.end_point,line))
            elements.append(line)
    
    if isClosed:
        line=DLine(DPoint(new_vs[-1][0],new_vs[-1][1]),DPoint(new_vs[0][0],new_vs[0][1]),color,handle,meta)
        elements.append(line)
        segments.append(DSegment(line.start_point,line.end_point,line))
    #print(len(elements),len(arcs),len(segments),handle)
    return elements,arcs,segments
            
def coordinatesmap(p:DPoint,insert,scales,rotation):
    rr=rotation/180*math.pi
    cosine=math.cos(rr)
    sine=math.sin(rr)

    # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
    x,y=((cosine*p[0]*scales[0]-sine*p[1]*scales[1]))+insert[0],((sine*p[0]*scales[0]+cosine*p[1]*scales[1]))+insert[1]
    return DPoint(x,y)
def transform_point(point,meta):
    return coordinatesmap(point,meta.insert,meta.scales,meta.rotation)
def transform_elements(elements):
    for e in elements:
        if e.meta.blockName!="TOP":
            e.transform()

def segment_arc_intersection(segment: DSegment, arc: DArc):
    # Parametrize the segment
    p1 = segment.start_point
    p2 = segment.end_point
    cx, cy = arc.center.x, arc.center.y
    r = arc.radius

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    a = dx**2 + dy**2
    b = 2 * (dx * (p1.x - cx) + dy * (p1.y - cy))
    c = (p1.x - cx)**2 + (p1.y - cy)**2 - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # No intersection

    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)

    points = []

    for t in [t1, t2]:
        if 0 <= t <= 1:
            ix = p1.x + t * dx
            iy = p1.y + t * dy
            intersection_point = DPoint(ix, iy)

            # Check if the intersection point is on the arc
            angle = math.degrees(math.atan2(intersection_point.y - cy, intersection_point.x - cx))
            angle = (angle + 360) % 360
            start_angle = (arc.start_angle + 360) % 360
            end_angle = (arc.end_angle + 360) % 360

            if start_angle > end_angle:
                if angle >= start_angle or angle <= end_angle:
                    points.append(intersection_point)
            else:
                if start_angle <= angle <= end_angle:
                    points.append(intersection_point)

    return points

def angle_from_center(point,start_point,arc):
    if point==start_point:
        return -10
    angle = math.degrees(math.atan2(point.y - arc.center.y, point.x - arc.center.x))
    # Normalize angle to [0, 360)
    angle = (angle + 360) % 360
    return angle
def angle_from_center_cross_zero(point,st_ag,start_point,arc):
    if point==start_point:
        return -10
    ag=angle_from_center(point,start_point,arc)
    if ag<st_ag:
        ag+=360
    return ag
# Function to sort points by angle on the arc
def sort_points_on_arc(points, arc):
    start_point, end_point = arc.points_on_arc()
    start_angle = (arc.start_angle + 360) % 360
    end_angle = (arc.end_angle+ 360) % 360

    if start_angle > end_angle:
        # If the arc crosses 0 degrees, handle the wrap-around
        return sorted(points, key=lambda pt: angle_from_center_cross_zero(pt,start_angle,start_point,arc))
    else:
        return sorted(points, key=lambda pt: angle_from_center(pt,start_point,arc))


#expand lines by fixed length
def expandFixedLength(segList,dist,both=True,verbose=True):


    new_seglist=[] 
    n=len(segList)
    # if verbose:
    #     pbar=tqdm(total=n,desc="Preprocess")
    for seg in segList:
        # if verbose:
        #     pbar.update()
        p1=seg[0]
        p2=seg[1]
        v=(p2[0]-p1[0],p2[1]-p1[1])
        l=math.sqrt(v[0]*v[0]+v[1]*v[1])
        if l<=0.25:
            continue
        v=(v[0]/l*dist,v[1]/l*dist)
        vs=DPoint(p1[0]-v[0],p1[1]-v[1]) if both else DPoint(p1[0],p1[1])
        ve=DPoint(p2[0]+v[0],p2[1]+v[1])
        new_seglist.append(DSegment(vs,ve,seg.ref))
    # if verbose:
    #     pbar.close()
    return new_seglist
# Function to split arcs using segments
def split_arcs(arcs, segments):
    split_segments = []
    pbar=tqdm(desc="split arcs",total=len(arcs)*len(segments))
    for arc in arcs:
        start_point, end_point = arc.points_on_arc()
        intersection_points = [start_point,end_point]

        for segment in segments:
            pbar.update()
            intersection_points.extend(segment_arc_intersection(segment, arc))

        # Remove duplicates and sort by angle
        intersection_points = list(set(intersection_points))
        intersection_points=sort_points_on_arc(intersection_points, arc)


         # Convert to segments

        for i in range(len(intersection_points) - 1):
            sp, ep = intersection_points[i], intersection_points[i + 1]

            # Calculate angles of the segment's points on the arc
            sp_angle = math.degrees(math.atan2(sp.y - arc.center.y, sp.x - arc.center.x))
            ep_angle = math.degrees(math.atan2(ep.y - arc.center.y, ep.x - arc.center.x))

            sp_angle = (sp_angle + 360) % 360
            ep_angle = (ep_angle + 360) % 360

            if sp_angle > ep_angle:
                ep_angle += 360
            delta_angle=ep_angle-sp_angle
            num_subdivisions = max(2, int((ep_angle - sp_angle) / 45))  # At least 2 segments, split every 10 degrees
            if delta_angle<30:
                num_subdivisions=1
            for j in range(num_subdivisions):
                t1 = j / num_subdivisions
                t2 = (j + 1) / num_subdivisions

                sub_sp_angle = sp_angle + t1 * (ep_angle - sp_angle)
                sub_ep_angle = sp_angle + t2 * (ep_angle - sp_angle)

                sub_sp = DPoint(arc.center.x + arc.radius * math.cos(math.radians(sub_sp_angle)),
                                arc.center.y + arc.radius * math.sin(math.radians(sub_sp_angle)))
                sub_ep = DPoint(arc.center.x + arc.radius * math.cos(math.radians(sub_ep_angle)),
                                arc.center.y + arc.radius * math.sin(math.radians(sub_ep_angle)))

                split_segments.append(DSegment(sub_sp, sub_ep,arc))
        
    pbar.close()
    return split_segments

def transform_segments(segments,scales,rotation,insert):
    new_segments=[]
    for segment in segments:
        vs,ve=segment.start_point,segment.end_point
        vs=coordinatesmap(vs,insert,scales,rotation)
        ve=coordinatesmap(ve,insert,scales,rotation)
        segment=DSegment(vs,ve,segment.ref)
        new_segments.append(segment)
    return new_segments
#json --> elements
def process_block(T_is_contained,block_datas,blockName,scales,rotation,insert,attribs,bound,block_elements,segmentation_config):
    print(f"正在处理块:{blockName}")
    # if blockName!="L125X75X7" or insert[1]>-36900:
    #     return [],[],[]
    filtered_block_elements=[]
    if segmentation_config.bracket_layer is  None:
        filtered_block_elements=block_elements
    else:
        for element in block_elements:
            if element["layerName"]==segmentation_config.bracket_layer:
                filtered_block_elements.append(element)
    block_elements=filtered_block_elements
    block_meta_data=DInsert(blockName,scales,rotation,insert,attribs,bound)
    #print(blockName,scales,rotation,insert,attribs,block_elements)
    elements=[]
    arcs=[]
    segments=[]
    stiffeners=[]
    arc_splits=[]
    ori_segments=[]
    for attrib in attribs:
        text=attrib["attribText"]
        if is_useful_text(text.strip()):
            elements.append(DText(bound={"x1":-50,"x2":50,"y1":-50,"y2":50},content=text,handle=attrib["attribHandle"],meta=block_meta_data))
    color = segmentation_config.color
    linetype =segmentation_config.line_type 
    elementtype=segmentation_config.element_type
    layname=segmentation_config.remove_layername
    for ele in block_elements:
        if  (T_is_contained and ele.get("layerName") is not None and ele["layerName"]!="T") or (T_is_contained and ele.get("layerName") is None) :
            continue
        if ele["type"]=="line":
           
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                if ele["layerName"] in segmentation_config.stiffener_name:
                    e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele["color"],ele["handle"],meta=block_meta_data)
                    stiffeners.append(DSegment(e.start_point,e.end_point,e))
                if ele["layerName"] in segmentation_config.remove_layername:
                    continue
             # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele["color"],ele["handle"],meta=block_meta_data)
    
            elements.append(e)
            segments.append(DSegment(e.start_point,e.end_point,e))
        elif ele["type"] == "arc":
            # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                continue
            # 创建DArc对象
            e = DArc(DPoint(ele["center"][0], ele["center"][1]), ele["radius"], ele["startAngle"], ele["endAngle"],ele["color"],ele["handle"],meta=block_meta_data)
            elements.append(e)
            arcs.append(e)
            continue
        elif ele["type"]=="spline":
            # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                continue
            vs = ele["vertices"]
            ps = [DPoint(v[0], v[1]) for v in vs]

            # Apply line simplification
            simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level

            e = DLwpolyline(simplified_ps, ele["color"], False,ele["handle"],meta=block_meta_data)
            elements.append(e)
            l = len(simplified_ps)
            for i in range(l - 1):
                # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
        elif ele["type"]=="lwpolyline" :
                # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                continue
            vs = ele["vertices"]
            vs_type=ele["verticesType"]
            lwe,lwa,lws=process_lwpoline(vs,vs_type,ele["color"],ele["handle"],block_meta_data,ele["isClosed"],ele["hasArc"])
            elements.extend(lwe)
            arcs.extend(lwa)
            segments.extend(lws)
            pass
            #ps = [DPoint(v[0], v[1]) for v in vs]

            # Apply line simplification
            #simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level

            #e = DLwpolyline(simplified_ps, ele["color"], ele["isClosed"],ele["handle"],True,ele["verticesType"],ele["vertices"],ele["hasArc"],meta=block_meta_data)
            #elements.append(e)
            #l = len(simplified_ps)
            #for i in range(l - 1):
                # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                #segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
            #if ele["isClosed"]:
                # if simplified_ps[-1].y>-48500 or simplified_ps[0].y>-48500:
                #segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
        elif  ele["type"]=="polyline":

            # 颜色过滤
            # if ele["color"] not in color:
            #     continue
            # 虚线过滤
            # if ele.get("linetype") is None or ele["linetype"] not in linetype:
            #     continue
            if ele.get("layerName") is not None and ele["layerName"] in layname:
                continue
            vs = ele["vertices"]
            ps = [DPoint(v[0], v[1]) for v in vs]

            # Apply line simplification
            simplified_ps = rdp(ps, epsilon=5.0)  # Adjust epsilon for simplification level
            #print(simplified_ps)
            e = DLwpolyline(simplified_ps, ele["color"], ele["isClosed"],ele["handle"],meta=block_meta_data)
            elements.append(e)
            l = len(simplified_ps)
            for i in range(l - 1):
                # if simplified_ps[i].y>-48500 or simplified_ps[i+1].y>-48500:
                segments.append(DSegment(simplified_ps[i], simplified_ps[i + 1], e))
            if ele["isClosed"]:
                # if simplified_ps[-1].y>-48500 or simplified_ps[0].y>-48500:
                segments.append(DSegment(simplified_ps[-1], simplified_ps[0], e))
        elif ele["type"]=="insert":
                sub_blockName=ele["blockName"]
                # x1,x2,y1,y2=ele["bound"]["x1"],ele["bound"]["x2"],ele["bound"]["y1"],ele["bound"]["y2"]
                sub_scales=ele["scales"]
                sub_rotation=ele["rotation"]
                sub_insert=ele["insert"]
                sub_attribs=ele['attribs']
                sub_block_data=block_datas[sub_blockName]
                sub_bound=ele["bound"]
                #pre-check
                sub_T_is_contained=False
                for sube in sub_block_data:
                    if sube.get("layerName") is not None and sube["layerName"]=="T":
                        sub_T_is_contained=True
                        break
                sub_elements,sub_segments,sub_arc_splits,sub_ori_segments,sub_stiffeners=process_block(sub_T_is_contained,block_datas,sub_blockName,sub_scales,sub_rotation,sub_insert,sub_attribs,sub_bound,sub_block_data,segmentation_config)
                for sube in sub_elements:
                    sube.meta=block_meta_data
                elements.extend(sub_elements)
                segments.extend(sub_segments)
                arc_splits.extend(sub_arc_splits)
                ori_segments.extend(sub_ori_segments)
                stiffeners.extend(sub_stiffeners)
        elif ele["type"]=="text" and blockName=="TOP":
                e=DText(ele["bound"],ele["insert"], ele["color"],ele["content"].strip(),ele["height"],ele["handle"],meta=block_meta_data)
                elements.append(e)
        elif  ele["type"]=="mtext" and blockName=="TOP":
            string = ele["text"].strip()
            cleaned_string = re.sub(r"^\\A1;", "", string)
            e=DText(ele["bound"],ele["insert"], ele["color"],cleaned_string,ele["width"],ele["handle"],meta=block_meta_data,is_mtext=True)
            elements.append(e)
        elif ele["type"]=="dimension" and blockName=="TOP":
            textpos=ele["textpos"]
            defpoints=[]
            for i in range(5):
                k="defpoint"+str(i+1)
                if k in ele:
                    defpoint=ele[k]
                    defpoints.append(DPoint(defpoint[0],defpoint[1]))
                else:
                    break
            e=DDimension(DPoint(textpos[0],textpos[1]),ele["color"],ele["text"].strip(),ele["measurement"],defpoints,ele["dimtype"],ele["handle"],meta=block_meta_data)
            elements.append(e)
        else:
            pass
    
    
    for s in segments:
        if s.length()>0:
            ori_segments.append(DSegment(s.start_point,s.end_point,s.ref))
        else:
            # print(s)
            pass
    expand_segments=expandFixedLength(segments,segmentation_config.line_expand_length)
    arc_splits.extend(split_arcs(arcs,expand_segments))
    for s in arc_splits:
        if s.length()>0:
            ori_segments.append(DSegment(s.start_point,s.end_point,s.ref))
        else:
            # print(s)
            pass
    # arc_splits=expandFixedLength(arc_splits,segmentation_config.arc_expand_length)
    # segments=segments+arc_splits

    segments=transform_segments(segments,scales,rotation,insert)
    arc_splits=transform_segments(arc_splits,scales,rotation,insert)
    ori_segments=transform_segments(ori_segments,scales,rotation,insert)
    stiffeners=transform_segments(stiffeners,scales,rotation,insert)
    transform_elements(elements)
    return elements,segments,arc_splits,ori_segments,stiffeners
# def process_blocks(block_sub_datas,segmentation_config):
#     elements=[]
#     segments=[]
#     stiffeners=[]
#     ori_segments=[]
#     for block_sub_data in block_sub_datas:
#         blockName,scales,rotation,insert,attribs,bound,block_elements=block_sub_data[0],block_sub_data[1],block_sub_data[2],block_sub_data[3],block_sub_data[4],block_sub_data[5],block_sub_data[6]
#         block_e,block_s,block_o,block_sf=process_block(blockName,scales,rotation,insert,attribs,bound,block_elements,segmentation_config)
#         elements.extend(block_e)
#         segments.extend(block_s)
#         ori_segments.extend(block_o)
#         stiffeners.extend(block_sf)
#     return elements,segments,ori_segments,stiffeners
def readJson(path,segmentation_config):
    # elements=[]
    block_sub_datas=[]
    # arcs=[]
    # segments=[]
    # color = [3, 7, 8, 4,2,140]
    # linetype = ["BYLAYER", "Continuous","Bylayer","CONTINUOUS","ByBlock","BYBLOCK"]
    # elementtype=["line","arc","lwpolyline","polyline","spline"]
    # layname=["Stiffener_Invisible"]
    try:  
        with open(path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_datas=data_list[1]
        elements,segments,arc_splits,ori_segments,stiffeners=process_block(False,block_datas,"TOP",[1.0,1.0],0,[0,0],[],None,data_list[0],segmentation_config)
        segments=expandFixedLength(segments,segmentation_config.line_expand_length)
        arc_splits=expandFixedLength(arc_splits,segmentation_config.arc_expand_length)
           
       
       
        return elements,segments+arc_splits,ori_segments,stiffeners
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")

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
# def find_all_intersections(segments, epsilon=1e-9):
#     intersection_dict = {}
#     n=len(segments)
#     pbar=tqdm(total=n*(n-1)/2,desc="计算交点")
#     for i, seg1 in enumerate(segments):
#         for j, seg2 in enumerate(segments):
#             if i >= j :
#                 continue  # Avoid duplicate checks and self-intersections

#             p1, p2 = seg1.start_point, seg1.end_point
#             q1, q2 = seg2.start_point, seg2.end_point
#             intersection = segment_intersection(p1, p2, q1, q2, epsilon)
#             if intersection:
#                 if seg1 not in intersection_dict:
#                     intersection_dict[seg1] = []
#                 if seg2 not in intersection_dict:
#                     intersection_dict[seg2] = []
                
#                 # Append the intersection for both segments
#                 intersection_dict[seg1].append(intersection)
#                 intersection_dict[seg2].append(intersection)
#             pbar.update()
#     # Sort intersections along each segment by their distance from the start point
#     for seg, isects in intersection_dict.items():
        
#         isects.sort(key=lambda p: (p.x - seg.start_point.x)**2 + (p.y - seg.start_point.y)**2)
#         intersection_dict[seg]=isects
#     pbar.close()
#     return intersection_dict

# Function to compute intersections for two subsets of segments
def compute_intersections(chunk1, chunk2, epsilon=1e-9):
    intersection_dict = {}
    n=len(chunk1)*(len(chunk2))
    pbar=tqdm(total=n,desc="计算交点")
    for seg1 in chunk1:
        for seg2 in chunk2:
            pbar.update()
            if seg1 == seg2 :
                continue  # Skip self-intersection
            
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2, epsilon)
            if intersection:
                if seg1 not in intersection_dict:
                    intersection_dict[seg1] = []
                # if seg2 not in intersection_dict:
                #     intersection_dict[seg2] = []
                
                # Append the intersection for both segments
                intersection_dict[seg1].append(intersection)
                # intersection_dict[seg2].append(intersection)
            
    pbar.close()
  
    return intersection_dict

# Function to merge results from multiple processes
def merge_intersections(results,segments):
    merged = {}
    for s in segments:
        if s not in merged:
            merged[s]=[s.start_point,s.end_point]
    for result in results:
        for seg, intersections in result.items():
            if seg not in merged:
                merged[seg] = []
            merged[seg].extend(intersections)
    for seg, isects in merged.items():
    
        isects.sort(key=lambda p: (p.x - seg.start_point.x)**2 + (p.y - seg.start_point.y)**2)
        merged[seg]=isects
    return merged

# Main function to find all intersections using multiprocessing
def find_all_intersections(segments, epsilon=0.1):
    n = len(segments)
    k= max((n+31)//32,1)
    # Divide segments into chunks of size k
    segment_chunks = [segments[i:i + k] for i in range(0, n, k)]
    s_l=[len(c) for c in segment_chunks]
    print(len(segment_chunks))
    print(s_l)
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=32) as executor:
        partial_intersections = partial(compute_intersections, chunk2=segments,epsilon=epsilon)
        results = list(executor.map(partial_intersections, segment_chunks))

    # Merge results from all processes
    intersection_dict = merge_intersections(results,segments)
    return intersection_dict
from collections import deque

def split_segments(segments, intersections,epsilon=0.25,expansion_param=0): 
    """根据交点将线段分割并构建 edge_map"""
    new_segments = []
    edge_map = {}
    point_map={}

    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        inter_points = sorted([seg.start_point] + inter_points + [seg.end_point], key=lambda p: (seg.start_point.x-p.x)*(seg.start_point.x-p.x)+(seg.start_point.y-p.y)*(seg.start_point.y-p.y))
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


def filter_segments(segments,intersections,point_map,expansion_param=12,iters=3,interval=1):
    new_segments=[]
    edge_map = {}
    new_point_map={}
    for seg, inter_points in intersections.items():
        # 按照坐标顺序排序交点
        
        inter_points = sorted([seg.start_point] + inter_points + [seg.end_point], key=lambda p: (seg.start_point.x-p.x)*(seg.start_point.x-p.x)+(seg.start_point.y-p.y)*(seg.start_point.y-p.y))
        points=inter_points
        segList=[]
        
        # 将线段分割为多个部分
        for i in range(len(points) - 1):
            # 比较两个点的大小，确保起点较小
            start_point, end_point = points[i], points[i+1]
            if (end_point.x, end_point.y) < (start_point.x, start_point.y):
                start_point, end_point = end_point, start_point
            if start_point == end_point:
                continue
            if ((start_point.x-end_point.x)*(start_point.x-end_point.x)+(start_point.y-end_point.y)*(start_point.y-end_point.y)) < expansion_param*expansion_param:
                continue
   
            new_seg = DSegment(start_point, end_point, seg.ref)
            segList.append(new_seg)


        # #filter lines
        # if len(segList)>1:
        for s in segList:
            new_segments.append(s)
            # 添加到 edge_map 中，包含正向和反向的线段
            if DSegment(s.start_point, s.end_point) not in edge_map:
                edge_map[DSegment(s.start_point, s.end_point)] = s
            if DSegment(s.end_point, s.start_point) not in edge_map:
                edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        # elif len(segList)==1:
        #     if isinstance(segList[0].ref, DArc):
        #         #arc
        #         new_segments.append(segList[0])
        #         s=segList[0]
        #         if DSegment(s.start_point, s.end_point) not in edge_map:
        #             edge_map[DSegment(s.start_point, s.end_point)] = s
        #         if DSegment(s.end_point, s.start_point) not in edge_map:
        #             edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段
        #     else:
        #         vs,ve=segList[0].start_point,segList[0].end_point
        #         s_deg,e_deg=len(point_map[vs]),len(point_map[ve])
        #         if s_deg==1 or e_deg==1:
        #             continue
        #         else:
        #             new_segments.append(segList[0])
        #             s=segList[0]
        #             if DSegment(s.start_point, s.end_point) not in edge_map:
        #                 edge_map[DSegment(s.start_point, s.end_point)] = s
        #             if DSegment(s.end_point, s.start_point) not in edge_map:
        #                 edge_map[DSegment(s.end_point, s.start_point)] = s  # 反向线段


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
    

    pbar=tqdm(total=iters,desc="过滤线段")
    for i in range(iters):
        filtered_segments=[]
        filtered_edge_map={}
        filtered_point_map={}

        for s in new_segments:
            vs,ve=s.start_point,s.end_point
            div_s,div_e=len(new_point_map[vs]),len(new_point_map[ve])
            # if vs==ve:
            #     continue
            if div_s==1 or div_e==1:
                continue
            # if div_s==3 and (i+1)%interval==0:
            #     ns=[ss for ss in new_point_map[vs] if ss !=s]
            #     if isinstance(ns[0].ref,DArc) and ns[0].ref == ns[1].ref:
            #         continue
            # if div_e==3 and (i+1)%interval==0:
            #     ns=[ss for ss in new_point_map[ve] if ss !=s]
            #     if isinstance(ns[0].ref,DArc) and ns[0].ref == ns[1].ref:
            #         continue
            
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
        pbar.update()
    pbar.close()
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


def bfs_paths(graph, start_point, end_point,max_length,timeout=5):
    """基于广度优先搜索找到所有从start_point到end_point的路径，返回路径中的Dsegment"""
    start_time=time.time()
    queue = deque([(start_point, [start_point], [])])  # (当前点，路径中的点，路径中的线段)
    all_paths = []

    while queue:
        current_time=time.time()
        if (current_time-start_time)>=timeout:
            print(f'{start_point}->{end_point}已超时')
            break
        (current_point, point_path, seg_path) = queue.popleft()
        if len(seg_path)>=max_length:
            continue
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

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) formed by three points p1, p2, and p3.
    The angle is measured from vector (p2 -> p1) to vector (p2 -> p3).
    """
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = math.atan2(det, dot_product)  # Angle in radians
    angle_deg = math.degrees(angle)

    # Ensure the angle is in the range [0, 360]
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def dfs_paths_with_repline(visited_edges,graph, repline, max_length, timeout=5):
    """
    Perform DFS to find a closed loop starting at repline.start_point,
    with the first segment being repline and ending back at repline.start_point.
    Prioritize edges with the largest counterclockwise angle change within [10, 170] degrees.
    """
    start_time = time.time()
    start_point = repline.start_point
    stack = [(repline.end_point, [start_point, repline.end_point], [repline])]  # Start with repline
    # visited_edges = {(repline.start_point, repline.end_point)}  # Mark the first edge as visited

    while stack:
        current_time = time.time()
        if (current_time - start_time) >= timeout:
            print(f'{start_point} search timed out')
            break
        
        current_point, point_path, seg_path = stack.pop()

        if len(seg_path) >= max_length:
            continue
  
        if current_point == start_point and len(seg_path) > 1:
            return seg_path  # Return the first valid closed loop
    
        # Get neighbors and sort them by counterclockwise angle change
        neighbors = []
        for neighbor, ref in graph.get(current_point, []):
            edge = (current_point, neighbor)
            prev_point = point_path[-2]
            if edge not in visited_edges and prev_point!=neighbor:
                neighbors.append((neighbor, ref))

        # Sort neighbors by the counterclockwise angle change
        if len(point_path) > 1:
            prev_point = point_path[-2]
            neighbors.sort(key=lambda nbr: calculate_angle(prev_point, current_point, nbr[0]), reverse=True)
        flag=False
        for neighbor, ref in neighbors:
           
            # angle = calculate_angle(point_path[-2], current_point, neighbor) if len(point_path) > 1 else 0
            
            # if 10 <= angle <= 170:  
            edge = (current_point, neighbor)
            visited_edges.add(edge)

            new_point_path = point_path + [neighbor]
            new_seg = DSegment(current_point, neighbor, ref)
            new_seg_path = seg_path + [new_seg]

            stack.append((neighbor, new_point_path, new_seg_path))
            flag=True
            break
        if flag==False:
            return []

    return []  # Return an empty path if no closed loop is found

def process_repline_with_repline_dfs(visited_edges,repline, graph, segmentation_config):
    """
    Process a single repline using the DFS algorithm to find a closed loop
    starting and ending at repline.start_point.
    """
    path = dfs_paths_with_repline(visited_edges,graph, repline, segmentation_config.path_max_length, segmentation_config.timeout)
    return path


# def compute_arc_replines(new_segments,point_map):
#     """
#     计算arc_replines，筛选出ref是弧线且半径在20~160之间的线段。
#     :param new_segments: 分割后的线段列表
#     :return: arc_replines 列表
#     """
#     arc_replines_map={}
#     arc_replines = []

#     for segment in new_segments:
#         # 检查线段的ref是否是弧线，且半径在 20 到 160 之间
#         if isinstance(segment.ref, DArc):
#             radius = segment.ref.radius
#             if 20 <= radius and radius <= 160:
#                 arc_tuple=(segment.ref.center.x,segment.ref.center.y,segment.ref.radius,segment.ref.start_angle,segment.ref.end_angle)
#                 if  arc_tuple not in arc_replines_map:
#                     arc_replines_map[arc_tuple]=[]
#                 arc_replines_map[arc_tuple].append(segment)
#     for arc_tuple,segments in arc_replines_map.items():
#         # arc_replines_map[arc_tuple]=sorted(segments,key= lambda s: DSegment(s.ref.center,s.mid_point()).slope())
#         for s in segments:
#             if len(point_map[s.start_point])>2 or len(point_map[s.end_point])>2:
#                 arc_replines.append(s)
        
#     return arc_replines

def compute_star_replines(new_segments,elements):
    vertical_lines=[]
    star_replines=[]
    star_pos_map={}
    star_pos_set=set()
    for e in elements:
        if isinstance(e,DText) and is_star_text(e.content):
            x,y=(e.bound["x1"]+e.bound["x2"])/2,(e.bound["y1"]+e.bound["y2"])/2
            vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y+5000)))
    for i, seg1 in enumerate(vertical_lines):
        y_min=None
        s=None

        for j, seg2 in enumerate(new_segments):
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
            rs=DSegment(s.end_point,s.start_point,s.ref)
            if s not in star_pos_map:
                star_pos_map[s]=set()
            if rs not in star_pos_map:
                star_pos_map[rs]=set()
            star_pos_map[s].add(seg1.start_point)
            star_pos_map[rs].add(seg1.start_point)
            star_pos_set.add(seg1.start_point)
    for pos,ss in star_pos_map.items():
        star_pos_map[pos]=list(ss)
    return star_replines,star_pos_map,list(star_pos_set)


def compute_line_replines(new_segments,point_map):
    line_replines=[]
    for s in new_segments:
        if s.length()<13 or s.length()>30:
            continue
        vs,ve=s.start_point,s.end_point
        div_s,div_e=len(point_map[vs]),len(point_map[ve])
        if div_s==3 and div_e==3:
            sn=[neighbor for neighbor in point_map[vs] if neighbor!=s]
            if sn[0].length()<sn[1].length():
                a=sn[0]
            else:
                a=sn[1]
            en=[neighbor for neighbor in point_map[ve] if neighbor!=s]
            if en[0].length()<en[1].length():
                b=en[0]
            else:
                b=en[1]
            if a.start_point==vs:
                p=a.end_point
            else:
                p=a.start_point
            if p in [b.start_point,b.end_point]:
                line_replines.append(s)
    return line_replines

def is_repline(s,segmentation_config):
    #print( s.ref.meta.scales[0])
    if isinstance(s.ref,DArc) and s.ref.radius<=segmentation_config.arc_repline_max_length and s.ref.radius>=segmentation_config.arc_repline_min_length:
        return True
    elif(not isinstance(s.ref,DArc)) and s.length()>=segmentation_config.line_repline_min_length and s.length()<=segmentation_config.line_repline_max_length:
        return True
    return False


def checkRefAndSlope(p,segments,tolerance,point_map,segmentation_config):
   
    #print(slopes)
    flag=False
    idx=None
    lines=[]
    for i,s1 in enumerate(segments):
        for j,s2 in enumerate(segments):
            if i>=j:
                continue
            if is_parallel(s1,s2,tolerance):
                flag=True
                l1,l2=segments[i].length(),segments[j].length()
                # if p==DPoint(777,1321) or p==DPoint():
                #     print(l1,l2)
                #     print(s1,s2)
                if l1<segmentation_config.repline_neighbor_min_length:
                    l=l1
                    q=s1.start_point if s1.end_point==p else s1.end_point
                    nq=[s for s in point_map[q] if s!=s1 and is_parallel(s,s1,tolerance)]
                    if len(nq)>0:
                        l+=nq[0].length()
                    #print(l)
                    if l<segmentation_config.repline_neighbor_min_length:
                        flag=False
                if flag and l2<segmentation_config.repline_neighbor_min_length:
                    l=l2
                    q=s2.start_point if s2.end_point==p else s2.end_point
                    nq=[s for s in point_map[q] if s!=s2 and is_parallel(s,s2,tolerance)]
                    if len(nq)>0:
                        l+=nq[0].length()
                    #print(l)
                    if l<segmentation_config.repline_neighbor_min_length:
                        flag=False
                
                if flag:
                    lines.append(s1)
                    lines.append(s2)
                    break
        if flag:
            break
    return [flag,lines]

def checkValid(repline,segments,tolerance,segmentation_config):
    flag=True
    for i,segment in enumerate(segments):
        if segment.length()>segmentation_config.check_valid_min_length and is_parallel(segment,repline,tolerance) and segment.ref==repline.ref:
            flag=False
            break
        
    return flag
                    
def checkTwoEndLines(ls,le,segmentation_config):
    flag=True
    if len(ls)!=2 or len(le)!=2:
        return False
    for i,s1 in enumerate(ls):
        for j,s2 in enumerate(le):
            if s1==s2 or s1==DSegment(s2.end_point,s2.start_point,s2.ref):
                continue
            if is_parallel(s1,s2,segmentation_config.is_parallel_tolerance_neighobor):
                flag=False
            if flag==False:
                break
        if flag==False:
            break
    return flag
def compute_cornor_holes(filtered_segments,filtered_point_map,segmentation_config):
    cornor_holes=[]
    segment_is_visited=set()
    for s in filtered_segments:
        vs,ve=s.start_point,s.end_point
        degs,dege=len(filtered_point_map[vs]),len(filtered_point_map[ve])
        if s not in segment_is_visited and (degs>2 or dege>2) and is_repline(s,segmentation_config):
            if degs>2 and dege>2:
                ns=[ss for ss in filtered_point_map[vs] if ss!=s ]
                ne=[ss for ss in filtered_point_map[ve] if ss!=s ]
                #print(222)
                if checkRefAndSlope(vs,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkRefAndSlope(ve,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkValid(s,ns,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config) and checkValid(s,ne,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config):
                    # a,b=checkRefAndSlope(ns)[1],checkRefAndSlope(ne)[1]
                    # pa=a.start_point if a.start_point!=vs else a.end_point
                    # pb=b.start_point if b.start_point!=ve else b.end_point
                    # sss=expandFixedLength([DSegment(vs,pa),DSegment(ve,pb)],150,False)
                    # if len(sss) <2:
                    #     break
                    # inter=segment_intersection(sss[0].start_point,sss[0].end_point,sss[1].start_point,sss[1].end_point)
                    # if inter is not None:
                    
                    ls=checkRefAndSlope(vs,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                    le=checkRefAndSlope(ve,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                    if checkTwoEndLines(ls,le,segmentation_config):
                        segment_is_visited.add(s)
                        cornor_holes.append(DCornorHole([s]))
            else:
                other=None
                segments=[]
                if degs>2:
                    start=vs
                    other=ve
                else:
                    start=ve
                    other=vs
                dego=len(filtered_point_map[other])
                segments.append(s)
                current=s
                flag=True
                while dego==2 and flag:
                    
                    cs=[ss for ss in filtered_point_map[other] if ss!=current]
                    if  len(cs)!=1:
                        flag=False
                    
                        break
                    else:
                        current=cs[0]
               
                    os=[p for p in [current.start_point,current.end_point] if p!=other]
                    if len(os)!=1 :
                        flag=False
                     
                        break
                    else:
                        other=os[0]
                 
                    dego=len(filtered_point_map[other])
                    if  current not in segment_is_visited and is_repline(current,segmentation_config):
                            segments.append(current)
                    else:
                        flag=False
                        break
                if flag:
                    ns=[ss for ss in filtered_point_map[start] if ss!=s ]
                    ne=[ss for ss in filtered_point_map[other] if ss!=current ]
                    #print(111)
                    if checkRefAndSlope(start,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkRefAndSlope(other,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[0] and checkValid(s,ns,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config) and checkValid(current,ne,segmentation_config.is_parallel_tolerance_neighobor,segmentation_config):
                        ls=checkRefAndSlope(start,ns,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                        le=checkRefAndSlope(other,ne,segmentation_config.is_parallel_tolerance_neighobor,filtered_point_map,segmentation_config)[1]
                        if checkTwoEndLines(ls,le,segmentation_config):
                            for ss in segments:
                                segment_is_visited.add(ss)
                            cornor_holes.append(DCornorHole(segments))
    for cornor_hole in cornor_holes:
        for s in cornor_hole.segments:
            s.isCornerhole=True
    #print(cornor_holes)
    return cornor_holes


def filter_cornor_holes(cornor_holes,filtered_point_map,segmentation_config):
    filtered_cornor_holes=[]
    for cornor_hole in cornor_holes:
        total=0
        n=0
        for s in cornor_hole.segments:
            total+=s.length()
            n+=1

        if n>0 and total>=segmentation_config.cornor_hole_total_length and total/n>=segmentation_config.cornor_hole_average_length:
            filtered_cornor_holes.append(cornor_hole)
    return filtered_cornor_holes

def isBraketHints(e):
    if isinstance(e,DText) and bool(re.match(r".*(F?B\d+X\d+).*", e.content)):
        return  True
    return False
def findBraketByHints(filtered_segments,text_map):
    vertical_lines=[]
    for pos,ts in text_map.items():
        for t in ts:
            result=t[1]
            if result["Type"]=="B" or result["Type"]=="BK":
                vertical_lines.append(DSegment(pos,DPoint(pos.x,pos.y+5000)))
                break
    braket_start_lines=[]

    #print(vertical_lines)
    for i, seg1 in enumerate(vertical_lines):
        y_min=None
        s=None
        for j, seg2 in enumerate(filtered_segments):
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
            braket_start_lines.append(s)
           
    

    return braket_start_lines

def findBracketByPoints(pos,filtered_segments):
    braket_start_lines=[]
    vertical_lines=[]
    for p in pos:
        x,y=p.x,p.y
        vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y-5000)))
    for i, seg1 in enumerate(vertical_lines):
        y_max=None
        s=None
        for j, seg2 in enumerate(filtered_segments):
          
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_max is None:
                    y_max=intersection[1]
                    s=seg2
                else:
                    if y_max<intersection[1]:
                        y_max=intersection[1]
                        s=seg2
        if s is not None:
            braket_start_lines.append(s)
    return braket_start_lines

def removeReferenceLines(elements,texts,initial_segments,all_segments,point_map,segmentation_config):
    vertical_lines=[]
    vl2=[]
    v1e=[]
    v2e=[]
    text_pos=[]
    for e in texts:
        mid_point=DPoint((e.bound["x1"]+e.bound["x2"])/2,(e.bound["y1"]+e.bound["y2"])/2)
        x,y=mid_point.x,mid_point.y
        x_1=(x+e.bound["x1"])/2
        x_2=(x+e.bound["x2"])/2
        vertical_lines.append(DSegment(DPoint(x_1,y),DPoint(x_1,y-500)))
        vertical_lines.append(DSegment(DPoint(x_2,y),DPoint(x_2,y-500)))
        vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y-500)))
        v1e.append(e)
        v1e.append(e)
        v1e.append(e)
        v2e.append(e)
        v2e.append(e)
        v2e.append(e)
        vl2.append(DSegment(DPoint(x,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_1,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_2,y),DPoint(x,y+500)))
    #print(len(vertical_lines))
    print(len(vertical_lines)*len(all_segments))
    print(len(vl2)*len(all_segments))
    horizontal_line=[]
    hl2=[]
    h1e=[]
    h2e=[]
            
    for i, seg1 in enumerate(vertical_lines):
        y_max=None
        s=None
        for j, seg2 in enumerate(all_segments):
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_max is None:
                    y_max=intersection[1]
                    s=seg2
                else:
                    if y_max<intersection[1]:
                        y_max=intersection[1]
                        s=seg2
        if s is not None:
            text_pos=seg1.start_point
            if (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance and s.ref.color==7 and (len(point_map[s.start_point])==1 or len(point_map[s.end_point])==1):
                horizontal_line.append(s)
                h1e.append(v1e[i])
    for i, seg1 in enumerate(vl2):
        y_min=None
        s=None
        for j, seg2 in enumerate(all_segments):
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
            text_pos=seg1.start_point
            if (y_min-text_pos.y)<=segmentation_config.reference_text_max_distance and s.ref.color==7 and (len(point_map[s.start_point])==1 or len(point_map[s.end_point])==1):
                hl2.append(s)
                h2e.append(v2e[i])
    # print(len(horizontal_line))
    reference_lines=[]
    text_pos_map={}
    text_set=set()
    for i,line in enumerate(horizontal_line):
        if len(point_map[line.start_point])==1:
            p=line.end_point
        else:
            p=line.start_point
        sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>90]
        ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>90]
        for j,s in enumerate(ss):
            reference_lines.append(sr[j])
            start_point=p
            current_point=s.start_point if s.start_point!=start_point else s.end_point
            current_line=s
            while True:
                if len(point_map[current_point])<=1:
                    break
                current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance)]
                if len(current_nedge)==0:
                    break
                current_line=current_nedge[0]
                current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
           
            if current_point not in text_pos_map:
                text_pos_map[current_point]=set()
                text_pos_map[current_point].add(h1e[i])
                if h1e[i] not in text_set:
                    text_set.add(h1e[i])
                h1e[i].textpos=True
            else:
                text_pos_map[current_point].add(h1e[i])
                if h1e[i] not in text_set:
                    text_set.add(h1e[i])
                h1e[i].textpos=True
        
        reference_lines.append(line.ref)
    for i,line in enumerate(hl2):
        if len(point_map[line.start_point])==1:
            p=line.end_point
        else:
            p=line.start_point
        sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle]
        ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle]
        for j,s in enumerate(ss):
            reference_lines.append(sr[j])
            start_point=p
            current_point=s.start_point if s.start_point!=start_point else s.end_point
            current_line=s
            while True:
                if len(point_map[current_point])<=1:
                    break
                current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance)]
                if len(current_nedge)==0:
                    break
                current_line=current_nedge[0]
                current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
            if current_point not in text_pos_map:
                text_pos_map[current_point]=set()
                text_pos_map[current_point].add(h2e[i])
                if h2e[i] not in text_set:
                    text_set.add(h2e[i])
                h2e[i].textpos=True
            else:
                text_pos_map[current_point].add(h2e[i])
                if h2e[i] not in text_set:
                    text_set.add(h2e[i])
                h2e[i].textpos=True
        reference_lines.append(line.ref)
    print(len(reference_lines)*len(initial_segments))
    new_segments=[]
    for s in initial_segments:
        
        if s.ref not in reference_lines:
            new_segments.append(s)
            #print(s.ref)
    return new_segments,reference_lines,text_pos_map,text_set

import itertools

def process_intersections(chunck,segments,point_map,segmentation_config):
    """Compute the intersection for a single vertical line and all segments."""
    vertical_lines=chunck[0]
    v1e=chunck[1]
    horizontal_line=[]
    h1e=[]
    pbar=tqdm(desc="find reference line",total=len(vertical_lines)*len(segments))
    for i, seg1 in enumerate(vertical_lines):
        y_max=None
        s=None
        for j, seg2 in enumerate(segments):
            pbar.update()
            if len(point_map[seg2.start_point])==1 and len(point_map[seg2.end_point])==1:
                continue
            p1, p2 = seg1.start_point, seg1.end_point
            q1, q2 = seg2.start_point, seg2.end_point
            intersection = segment_intersection(p1, p2, q1, q2)
            if intersection:
                if y_max is None:
                    y_max=intersection[1]
                    s=seg2
                else:
                    if y_max<intersection[1]:
                        y_max=intersection[1]
                        s=seg2
        if s is not None:
            text_pos=seg1.start_point
            if (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance and s.ref.color==7 and (len(point_map[s.start_point])==1 or len(point_map[s.end_point])==1):
                horizontal_line.append(s)
                h1e.append(v1e[i])
    pbar.close()
    return [horizontal_line,h1e]

def process_intersections2(chunck,segments,point_map,segmentation_config):
    vl2=chunck[0]
    v2e=chunck[1]
    hl2=[]
    h2e=[]
    pbar=tqdm(desc="find reference lines",total=len(vl2)*len(segments))
    for i, seg1 in enumerate(vl2):
        y_min=None
        s=None
        for j, seg2 in enumerate(segments):
            pbar.update()
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
            text_pos=seg1.start_point
            if (y_min-text_pos.y)<=segmentation_config.reference_text_max_distance and s.ref.color==7 and (len(point_map[s.start_point])==1 or len(point_map[s.end_point])==1):
                hl2.append(s)
                h2e.append(v2e[i])
    pbar.close()
    return [hl2,h2e]
def checkReferenceLine(p,ns,ss,segmentation_config):
    if len(ns)==1:
        if len(ss)==1:
            return True
        return False
    if len(ns)>2:
        return False
    if len(ss)!=2:
        return False
    return True
def removeReferenceLines(elements,texts,initial_segments,all_segments,point_map,segmentation_config):
    vertical_lines=[]
    vl2=[]
    v1e=[]
    v2e=[]
    text_pos=[]
    for e in texts:
        mid_point=DPoint((e.bound["x1"]+e.bound["x2"])/2,(e.bound["y1"]+e.bound["y2"])/2)
        x,y=mid_point.x,mid_point.y
        x_1=(x+e.bound["x1"])/2
        x_2=(x+e.bound["x2"])/2
        vertical_lines.append(DSegment(DPoint(x_1,y),DPoint(x_1,y-500)))
        vertical_lines.append(DSegment(DPoint(x_2,y),DPoint(x_2,y-500)))
        vertical_lines.append(DSegment(DPoint(x,y),DPoint(x,y-500)))
        v1e.append(e)
        v1e.append(e)
        v1e.append(e)
        v2e.append(e)
        v2e.append(e)
        v2e.append(e)
        vl2.append(DSegment(DPoint(x,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_1,y),DPoint(x,y+500)))
        vl2.append(DSegment(DPoint(x_2,y),DPoint(x,y+500)))
    #print(len(vertical_lines))
    print(len(vertical_lines)*len(all_segments))
    print(len(vl2)*len(all_segments))
    horizontal_line=[]
    hl2=[]
    h1e=[]
    h2e=[]
    n = len(vertical_lines)
    k= max((n+31)//32,1)
    # Divide segments into chunks of size k
    segment_chunks = [vertical_lines[i:i + k] for i in range(0, n, k)]
    element_chunks=  [v1e[i:i + k] for i in range(0, n, k)]
    chuncks=[]
    for i in range(len(segment_chunks)):
        chuncks.append([segment_chunks[i],element_chunks[i]])
    s_l=[len(c) for c in segment_chunks]
    print(len(segment_chunks))
    print(s_l)
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=32) as executor:
        partial_intersections = partial(process_intersections, segments=all_segments,point_map=point_map,segmentation_config=segmentation_config)
        results = list(executor.map(partial_intersections, chuncks))
    for result in results:
        horizontal_line.extend(result[0])
        h1e.extend(result[1])
    n = len(vl2)
    k= max((n+31)//32,1)
    # Divide segments into chunks of size k
    segment_chunks = [vl2[i:i + k] for i in range(0, n, k)]
    element_chunks=  [v2e[i:i + k] for i in range(0, n, k)]
    chuncks=[]
    for i in range(len(segment_chunks)):
        chuncks.append([segment_chunks[i],element_chunks[i]])
    s_l=[len(c) for c in segment_chunks]
    print(len(segment_chunks))
    print(s_l)
    # Use ProcessPoolExecutor for parallel computation
    with ProcessPoolExecutor(max_workers=32) as executor:
        partial_intersections = partial(process_intersections2, segments=all_segments,point_map=point_map,segmentation_config=segmentation_config)
        results = list(executor.map(partial_intersections, chuncks))
    for result in results:
        hl2.extend(result[0])
        h2e.extend(result[1])
  
    reference_lines=[]
    text_pos_map={}
    text_set=set()
    text_pos_map2={}
    text_set2=set()
    for i,line in enumerate(horizontal_line):
        if len(point_map[line.start_point])==1:
            p=line.end_point
        else:
            p=line.start_point
        
        ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
        sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
        ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
        flag=checkReferenceLine(p,ns,ss,segmentation_config)
        if flag:
            refs=[]
            for j,s in enumerate(ss):
                refs.append(sr[j])
                start_point=p
                current_point=s.start_point if s.start_point!=start_point else s.end_point
                current_line=s
                k=0
                while True:
                    # print(k)
                    k+=1
                    if k>10:
                        flag=False
                        break
                    if len(point_map[current_point])<=1:
                        break
                    current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance)]
                    if len(current_nedge)==0:
                        break
                    current_line=current_nedge[0]
                    current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
            
                if flag and current_point not in text_pos_map:
                    text_pos_map[current_point]=set()
                    text_pos_map[current_point].add(h1e[i])
                    if h1e[i] not in text_set:
                        text_set.add(h1e[i])
                    h1e[i].textpos=True
                elif flag:
                    text_pos_map[current_point].add(h1e[i])
                    if h1e[i] not in text_set:
                        text_set.add(h1e[i])
                    h1e[i].textpos=True
                if flag==False:
                    break
            if flag:
                reference_lines.extend(refs)
                reference_lines.append(line.ref)
    for i,line in enumerate(hl2):
        if len(point_map[line.start_point])==1:
            p=line.end_point
        else:
            p=line.start_point
        ns=[s for s in point_map[p] if s!=line and s.length()>segmentation_config.reference_line_min_length] 
        sr=[s.ref for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
        ss=[s for s in point_map[p] if s.length()>segmentation_config.reference_line_min_length  and (isinstance(s.ref, DLine) or isinstance(s.ref,DLwpolyline)) and s!=line and angleOfTwoSegmentsWithCommonStarter(p,s,line)>segmentation_config.reference_min_angle and angleOfTwoSegmentsWithCommonStarter(p,s,line)<segmentation_config.reference_max_angle]
        flag=checkReferenceLine(p,ns,ss,segmentation_config)
        if flag:
            refs=[]
            for j,s in enumerate(ss):
                refs.append(sr[j])
                start_point=p
                current_point=s.start_point if s.start_point!=start_point else s.end_point
                current_line=s
                k=0
                while True:
                    k+=1
                    if k>10:
                        flag=False
                        break
                    if len(point_map[current_point])<=1:
                        break
                    current_nedge=[sss for sss in point_map[current_point] if sss!=current_line and is_parallel(current_line,sss,segmentation_config.is_parallel_tolerance)]
                    if len(current_nedge)==0:
                        break
                    current_line=current_nedge[0]
                    current_point=current_line.start_point if current_line.start_point!=current_point else current_line.end_point
                if flag and current_point not in text_pos_map2:
                    text_pos_map2[current_point]=set()
                    text_pos_map2[current_point].add(h2e[i])
                    if h2e[i] not in text_set2:
                        text_set2.add(h2e[i])
                    h2e[i].textpos=True
                elif flag:
                    text_pos_map2[current_point].add(h2e[i])
                    if h2e[i] not in text_set2:
                        text_set2.add(h2e[i])
                    h2e[i].textpos=True
                if flag==False:
                    break
            if flag:
                reference_lines.extend(refs)
                reference_lines.append(line.ref)
    print(len(reference_lines)*len(initial_segments))
    new_segments=[]
    removed_segments=[]
    for s in initial_segments:
        
        if s.ref not in reference_lines:
            new_segments.append(s)
            #print(s.ref)
        else:
            removed_segments.append(s)
    print("==============")
    print(len(removed_segments))
    return new_segments,reference_lines,text_pos_map,text_set,text_pos_map2,text_set2,removed_segments
    
# def convert_ref_to_tuple(ref):
#     """
#     Convert different ref objects (Line, Arc, etc.) to a unified tuple format for comparison.
#     """
#     if isinstance(ref, DLine):
#         # Convert Line to a tuple of its start and end points
#         return ('Line', (ref.start_point.x, ref.start_point.y), (ref.end_point.x, ref.end_point.y))
#     elif isinstance(ref, DArc):
#         # Convert Arc to a tuple of its defining properties: center, radius, start_angle, end_angle
#         return ('Arc', (ref.center.x, ref.center.y), ref.radius, ref.start_angle, ref.end_angle)
#     else:
#         # Handle other types or return a generic string for unknown types
#         return ('Unknown', str(ref))


def computeCenterCoordinates(poly):

    # w=0
    # x=0
    # y=0
    # for edge in poly:
    #     # if not isinstance(edge.ref,DArc):
    #         # l=edge.ref.weight
    #         # w=w+l
    #         # x=x+l*edge.ref.bc.x
    #         # y=y+l*edge.ref.bc.y
    #     l=edge.length()
    #     w+=l
    #     x+=l*(edge.start_point.x+edge.end_point.x)/2
    #     y+=l*(edge.start_point.y+edge.end_point.y)/2
    points = []
    for segment in poly:
        points.append(segment.start_point)
        points.append(segment.end_point)
        x = sum(point.x for point in points) / len(points)
        y = sum(point.y for point in points) / len(points)
   
    return DPoint(x,y)


#count the number of replines in a poly
def countReplines(poly,replines_set):
    count=0
    for edge in poly:
        if edge in replines_set:
            count+=1
    return count


def remove_duplicate_polygons(closed_polys,eps=25.0,min_samples=1):
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
    # poly_map={}
    # for poly in closed_polys:
    #     bc=computeCenterCoordinates(poly)
    #     #bc=DPoint(int(bc.x/span)*span,int(bc.y/span)*span)
    #     if bc not in poly_map:
    #         poly_map[bc]=poly
    #     else:
    #         if len(poly_map[bc])<len(poly):
    #             poly_map[bc]=poly
    tps=[]
    for poly in closed_polys:
        bc=computeCenterCoordinates(poly)
        tps.append((bc,poly))
    # for bc,poly in poly_map.items():
        # unique_polys.append(poly)
        # bcs.append(bc)
        # tps.append((bc,poly))
    # tps=sorted(tps,key=lambda item: item[0].as_tuple())
    
    #filter tps
    
    points =[]
    areas=[]
    for tp in tps:
        points.append([tp[0].x,tp[0].y])
        # replines_count.append(countReplines(tp[1],replines_set))
        areas.append(computeAreaOfPoly(tp[1]))
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
                if areas[poly_map[label]]>areas[idx]:
                    #print(len(poly_map[label]),len(tps[idx][1]))
                    poly_map[label]=idx
    
    for label,idx in poly_map.items():
        unique_polys.append(tps[idx][1])
        #print(tps[idx][0])
    
    return unique_polys

def computeBoundingBox(poly):
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
    return x_min,x_max,y_min,y_max


# filter polys by area of polys's bounding boxes 
def filterPolys(polys,max_length=15,min_length=3,t_min=5000,t_max=1000*1500,d=5):
    
    for poly in polys:
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
    filtered_polys=[]
    for poly in polys:

        if len(poly)>max_length or len(poly)<=min_length:
            continue
       
        #area and box
        x_min,x_max,y_min,y_max=computeBoundingBox(poly)
        area=computeAreaOfPoly(poly)
        # print(area)
        yy=y_max-y_min
        xx=x_max-x_min
        if yy<20 or xx<20:
            continue
        if yy>xx:
            div=yy/xx
        else:
            div=xx/yy
        if area<t_min or area>t_max or div>=d:
            continue
        #parellel line
        flag=False
        for i,segment in enumerate(poly):
            dx_1 = segment.end_point.x - segment.start_point.x
            dy_1 = segment.end_point.y - segment.start_point.y
            l=(dx_1**2+dy_1**2)**0.5
            v_1=(dy_1/l*50.0,-dx_1/l*50.0)
            mid_p=DPoint((segment.start_point.x+segment.end_point.x)/2,(segment.start_point.y+segment.end_point.y)/2)

            for j,other in  enumerate(poly):
                if i==j:
                    continue
                dx_2 = other.end_point.x - other.start_point.x
                dy_2 = other.end_point.y - other.start_point.y
                
                # 计算斜率
                k_1 = math.pi/2 if dx_1 == 0 else math.atan(dy_1 / dx_1)
                k_2 = math.pi/2 if dx_2 == 0 else math.atan(dy_2 / dx_2)
                if (math.fabs(dx_1) <=0.025 and math.fabs(dx_2) <=0.025) or (math.fabs(k_1-k_2)<=0.1):
                    s1=DSegment(DPoint(segment.start_point.x+v_1[0],segment.start_point.y+v_1[1]),DPoint(segment.start_point.x-v_1[0],segment.start_point.y-v_1[1]))
                    s2=DSegment(DPoint(segment.end_point.x+v_1[0],segment.end_point.y+v_1[1]),DPoint(segment.end_point.x-v_1[0],segment.end_point.y-v_1[1]))                                
                    s3=DSegment(DPoint(mid_p.x+v_1[0],mid_p.y+v_1[1]),DPoint(mid_p.x-v_1[0],mid_p.y-v_1[1]))
                    i1=segment_intersection(s1.start_point,s1.end_point,other.start_point,other.end_point)
                    if i1==other.end_point or i1==other.start_point:
                        i1=None
                    i2=segment_intersection(s2.start_point,s2.end_point,other.start_point,other.end_point)

                    if i2==other.end_point or i2==other.start_point:
                        i2=None
                    i3=segment_intersection(s3.start_point,s3.end_point,other.start_point,other.end_point)
                    
                    if i3==other.end_point or i3==other.start_point:
                        i3=None
                    if (i1 is not None) or (i2 is not None) or (i3 is not None):
                        flag=True
                        break
            if flag:
                break
        if not flag:
            filtered_polys.append(poly)
    
    #print(filtered_polys[0])
    valid_polys=filtered_polys

    # for poly in filtered_polys:
    #     valid=True
    #     for edge in poly:
    #         if edge in arc_replines_set:
    #             o=DPoint((edge.ref.start_point.x+edge.ref.end_point.x)/2,
    #               (edge.ref.start_point.y+edge.ref.end_point.y)/2)
    #             if  is_point_in_polygon(o,poly):
    #                 valid=False
    #                 break
    #     if  True:
    #         valid_polys.append(poly)

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

def outputLines(segmentation_config,segments,point_map,polys,cornor_holes,star_pos,texts,text_map,dimensions,replines,linePNGPath,drawIntersections=False,drawLines=False,drawPolys=False):
    def p_minus(a,b):
        return DPoint(a.x-b.x,a.y-b.y)
    def p_add(a,b):
        return DPoint(a.x+b.x,a.y+b.y)
    def p_mul(a,k):
        return DPoint(a.x*k,a.y*k)
    if drawLines:
        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k-')
    if drawIntersections:
        for p,ss in point_map.items():
            if len(ss)>1:
                # print(p.x,p.y)
                plt.plot(p.x, p.y, 'r.')
    if drawPolys:
        for poly in polys:
            for edge in poly:
                x_values = [edge.start_point.x, edge.end_point.x]
                y_values = [edge.start_point.y, edge.end_point.y]
                plt.plot(x_values, y_values, 'b-', lw=2)
    for cornor_hole in cornor_holes:
        for edge in cornor_hole.segments:
            x_values = [edge.start_point.x, edge.end_point.x]
            y_values = [edge.start_point.y, edge.end_point.y]
            plt.plot(x_values, y_values, 'g-', lw=2)
    for p in star_pos:
        plt.plot(p.x, p.y, 'b.')
    # for p in braket_pos:
    #     plt.plot(p.x, p.y, 'g.')
    for edge in replines:
        x_values = [edge.start_point.x, edge.end_point.x]
        y_values = [edge.start_point.y, edge.end_point.y]
        plt.plot(x_values, y_values, 'r-', lw=2)
    if segmentation_config.draw_texts:
        for p,t in text_map.items():
            plt.plot(p.x, p.y, 'g.')
            plt.text(p.x, p.y, [(tt[0].content,tt[1]["Type"],tt[2]) for tt in t], fontsize=10)
        for d_t in dimensions:
            p=d_t[1]
            d=d_t[0]
            if d.dimtype==32 or d.dimtype==33 or d.dimtype==161 or d.dimtype==160:
                l0=p_minus(d.defpoints[0],d.defpoints[2])
                l1=p_minus(d.defpoints[1],d.defpoints[2])
                d10=l0.x*l1.x+l0.y*l1.y
                d00=l0.x*l0.x+l0.y*l0.y
                if d00 <1e-4:
                    x=d.defpoints[1]
                else:
                    x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
                d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]
                ss=[DSegment(d1,d4),DSegment(d4,d3),DSegment(d3,d2)]
                sss=[DSegment(d2,d1)]
                ss=expandFixedLength(ss,25,True,False)
                sss=expandFixedLength(sss,100,True,False)
                q=sss[0].end_point
                sss=expandFixedLength(sss,100,False,False)
                for s in ss:
                    plt.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                for s in sss:
                    plt.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
                plt.arrow(sss[0].end_point.x, sss[0].end_point.y, d1.x-sss[0].end_point.x, d1.y-sss[0].end_point.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(sss[0].start_point.x, sss[0].start_point.y, d2.x-sss[0].start_point.x, d2.y-sss[0].start_point.y, head_width=20, head_length=20, fc='red', ec='red')
                perp_vx, perp_vy = sss[0].start_point.x - sss[0].end_point.x, sss[0].start_point.y-sss[0].end_point.y
                rotation_angle = np.arctan2(-perp_vy, -perp_vx) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            elif d.dimtype==37:
                a,b,o=d.defpoints[1],d.defpoints[2],d.defpoints[3]
                r=DSegment(d.defpoints[0],o).length()
                r0=DSegment(a,o).length()
                oa_=p_mul(p_minus(a,o),r/r0)
                ob_=p_mul(p_minus(b,o),r/r0)
                ao_=p_mul(oa_,-1)
                bo_=p_mul(ob_,-1)
                a_= p_add(o, oa_)
                b_ = p_add(o, ob_)
                ia_=p_add(o,ao_)
                ib_=p_add(o,bo_)
                delta=p_mul(DPoint(oa_.y,-oa_.x),3)
                sp=p_add(delta,a_)
                plt.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
                q=p_mul(p_add(a_,sp),0.5)
                rotation_angle = np.arctan2(-delta.y, delta.x) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            elif d.dimtype==163:
                a,b=d.defpoints[0],d.defpoints[3]
                o=p_mul(p_add(a,b),0.5)
                ss=[DSegment(a,b)]
                ss=expandFixedLength(ss,100,True,False)
                ss=expandFixedLength(ss,100,False,False)
                a_,b_=ss[0].start_point,ss[0].end_point
                plt.arrow(a_.x, a_.y, a.x-a_.x,a.y-a_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(b_.x, b_.y, b.x-b_.x,b.y-b_.y, head_width=20, head_length=20, fc='red', ec='red')
                q=p_add(p_mul(b_,0.7),p_mul(b,0.3))
                ab=p_minus(b,a)
                rotation_angle = np.arctan2(ab.y, -ab.x) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            elif d.dimtype==34:
                a,b_,b,o=d.defpoints[0],d.defpoints[1],d.defpoints[2],d.defpoints[3]
                r=DSegment(d.defpoints[4],o).length()
                ra=DSegment(a,o).length()
                rb=DSegment(b,o).length()
                oa_=p_mul(p_minus(a,o),r/ra)
                ob_=p_mul(p_minus(b,o),r/rb)
                ao_=p_mul(oa_,-1)
                bo_=p_mul(ob_,-1)
                a_= p_add(o, oa_)
                b_ = p_add(o, ob_)
                ia_=p_add(o,ao_)
                ib_=p_add(o,bo_)
                delta=p_mul(DPoint(oa_.y,-oa_.x),3)
                sp=p_add(delta,a_)
                plt.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red')
                plt.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
                q=p_mul(p_add(a_,sp),0.5)
                rotation_angle = np.arctan2(delta.y, -delta.x) * (180 / np.pi)
                plt.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
        #plt.plot(p.x, p.y, marker='o', markerfacecolor="#E38C35",markersize=10)
        
    # for cornor_hole in cornor_holes:
    #     print(cornor_hole.segments)
    plt.gca().axis('equal')
    plt.savefig(linePNGPath)
    print(f"直线图保存于:{linePNGPath}")

def outputPolysAndGeometry(point_map,polys,path,draw_polys=False,draw_geometry=False,n=10):
    t=len(polys) if len(polys)<n else n
    
    if draw_geometry:
        pbar=tqdm(total=t,desc="输出几何")
        for i,poly in enumerate(polys):
            if i>=n:
                break
            plot_geometry(poly,os.path.join(path,f"geometry{i}.png"))
            pbar.update()
        pbar.close()
   
    if draw_polys:
        pbar=tqdm(total=t,desc="输出多边形")
        for i,poly in enumerate(polys):
            if i>=n:
                break
            plot_polys(point_map,poly,os.path.join(path,f"poly{i}.png"))
            pbar.update()
        pbar.close()
    print(f"封闭多边形图像保存于:{path}")

def removeOddPoints(filtered_segments,filtered_point_map,segmentation_config):
    edge_set=set()
    for p,ne in filtered_point_map.items():
        for e in ne:
            edge_set.add(e)
    for p,ne in filtered_point_map.items():
        degp=len(ne)
        if p==2 and checkRefAndSlope(ne,segmentation_config.is_parallel_tolerance_neighobor)[0]:
            a=ne[0].start_point if ne[0].start_point!=p else ne[0].end_point
            b=ne[1].start_point if ne[1].start_point!=p else ne[1].end_point
            new_edge=DSegment(a,b,ne[0].ref)
            edge_set.remove(ne[0])
            edge_set.remove(ne[1])
            edge_set.add(new_edge)
    new_segments=list(edge_set)
    new_point_map={}
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
    return new_segments,new_point_map
    
def process_repline(repline, graph, segmentation_config):
    """
    处理单个 repline 的逻辑，使用 BFS 查找路径并构成闭合路径。
    """
    start_point = repline.start_point
    end_point = repline.end_point
    # 使用 BFS 查找从 start_point 到 end_point 的所有路径
    paths = bfs_paths(graph, start_point, end_point, segmentation_config.path_max_length,segmentation_config.timeout)

    # 构成闭合路径
    for path in paths:
        path.append(repline)
    return paths

def process_text_map(text_map,removed_segments,segmentation_config):
    new_text_map={}
    text_pos_map1,text_pos_map2,text_pos_map3=text_map["top"][1],text_map["bottom"][1],text_map["other"][1]
    for p,texts in text_pos_map1.items():
        for t in texts:
            result=parse_elbow_plate(t.content, annotation_position="top")
            if result is None:
                result={
                    "Type":"None"
                }
            mid_point=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2)
            x,y=mid_point.x,mid_point.y
            x_1=(x+t.bound["x1"])/2
            x_2=(x+t.bound["x2"])/2
            l1=DSegment(DPoint(x_1,y),DPoint(x_1,y-500))
            l2=DSegment(DPoint(x_2,y),DPoint(x_2,y-500))
            l3=DSegment(DPoint(x,y),DPoint(x,y-500))
            d_min=None
            for i, seg1 in enumerate([l1,l2,l3]):
                y_max=None
                s=None
                for j, seg2 in enumerate(removed_segments):
                    p1, p2 = seg1.start_point, seg1.end_point
                    q1, q2 = seg2.start_point, seg2.end_point
                    intersection = segment_intersection(p1, p2, q1, q2)
                    if intersection:
                        if y_max is None:
                            y_max=intersection[1]
                            s=seg2
                        else:
                            if y_max<intersection[1]:
                                y_max=intersection[1]
                                s=seg2
                if s is not None:
                    text_pos=seg1.start_point
                    if (text_pos.y-y_max)<=segmentation_config.reference_text_max_distance :
                        d=math.fabs(text_pos.y-y_max)
                        if d_min is None:
                            d_min=d
                        else:
                            if d_min>d:
                                d_min=d




                       
            if p not in new_text_map:
                new_text_map[p]=[]
                new_text_map[p].append([t,result,"top",d_min])
            else:
                new_text_map[p].append([t,result,"top",d_min])

    for p,texts in text_pos_map2.items():
        for t in texts:
            result=parse_elbow_plate(t.content, annotation_position="bottom")
            if result is None:
                result={
                    "Type":"None"
                }
            mid_point=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2)
            x,y=mid_point.x,mid_point.y
            x_1=(x+t.bound["x1"])/2
            x_2=(x+t.bound["x2"])/2
            l1=DSegment(DPoint(x_1,y),DPoint(x_1,y+500))
            l2=DSegment(DPoint(x_2,y),DPoint(x_2,y+500))
            l3=DSegment(DPoint(x,y),DPoint(x,y+500))
            d_min=None
            for i, seg1 in enumerate([l1,l2,l3]):
                y_min=None
                s=None
                for j, seg2 in enumerate(removed_segments):
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
                    text_pos=seg1.start_point
                    if (text_pos.y-y_min)<=segmentation_config.reference_text_max_distance :
                        d=math.fabs(text_pos.y-y_min)
                        if d_min is None:
                            d_min=d
                        else:
                            if d_min>d:
                                d_min=d
            if p not in new_text_map:
                new_text_map[p]=[]
                new_text_map[p].append([t,result,"bottom",-d_min])
            else:
                new_text_map[p].append([t,result,"bottom",-d_min])
    for p,texts in text_pos_map3.items():
        for t in texts:
            result=parse_elbow_plate(t.content, annotation_position="other")
            if result is None:
                result={
                    "Type":"None"
                }
            
            if p not in new_text_map:
                new_text_map[p]=[]
                new_text_map[p].append([t,result,"other",None])
            else:
                new_text_map[p].append([t,result,"other",None])
    for p ,texts in new_text_map.items():
        text_wo_d=[]
        text_w_d=[]
        text_map={}
        for t in texts:
            if t[3] is None:
                text_wo_d.append(t)
            else:
                if t[1]["Type"]=="R":
                    text_wo_d.append(t)
                else:
                    content=t[0].content.strip()
                    if content not in text_map:
                        text_map[content]=(t[3],t)
                    else:
                        if text_map[content][0]<t[3]:
                            text_map[content]=(t[3],t)
        for content,t_t in text_map.items():
            text_w_d.append(t_t[1])

        text_w_d=sorted(text_w_d,key=lambda t:t[3],reverse=True)
        new_text_w_d=[]
        if len(text_w_d)>1:
            for i in range(len(text_w_d)):
                if i==0:
                    #top
                    t=text_w_d[i][0]
                    result=parse_elbow_plate(t.content, annotation_position="top")
                    if result is None:
                        result={
                            "Type":"None"
                    }
                    new_text_w_d.append([t,result,"top",text_w_d[i][3]])
                if i==1:
                    #bottom
                    t=text_w_d[i][0]
                    result=parse_elbow_plate(t.content, annotation_position="bottom")
                    if result is None:
                        result={
                            "Type":"None"
                    }
                    new_text_w_d.append([t,result,"bottom",text_w_d[i][3]])
                if i>1:
                    new_text_w_d.append(text_w_d[i])
        else:
            new_text_w_d=text_w_d
        new_text_map[p]=[]
        new_text_map[p].extend(new_text_w_d)
        new_text_map[p].extend(text_wo_d)

    return new_text_map
def findClosedPolys_via_BFS(elements,texts,dimensions,segments,segmentation_config):
    verbose=segmentation_config.verbose
    # Step 1: 计算交点
    # if verbose:
    #     print("计算交点")
    isecDic = find_all_intersections(segments,segmentation_config.intersection_epsilon)

    # Step 2: 根据交点分割线段
    # if verbose:
    #     print("根据交点分割线段")
    new_segments, edge_map,point_map= split_segments(segments, isecDic,segmentation_config.segment_filter_length)
    filtered_segments, filtered_edge_map,filtered_point_map= filter_segments(segments,isecDic,point_map,segmentation_config.segment_filter_length,segmentation_config.segment_filter_iters,segmentation_config.segment_remove_interval)
    
    
    
    #remove rfernce lines
    initial_segments,reference_lines,text_pos_map1,text_set1,text_pos_map2,text_set2,removed_segments=removeReferenceLines(elements,texts,segments,new_segments,point_map,segmentation_config)
    text_set3=set()
    text_pos_map3={}
    for t in texts:
        if t not in text_set1 and t not in text_set2:
            pos=DPoint((t.bound['x1']+t.bound['x2'])/2,(t.bound['y1']+t.bound['y2'])/2)
            if pos not in text_pos_map3:
                text_pos_map3[pos]=set()
                text_pos_map3[pos].add(t)
            else:
                text_pos_map3[pos].add(t)
    text_map={}
    for pos,ts in text_pos_map3.items():
        text_pos_map3[pos]=list(ts)
    text_map["top"]=(text_set1,text_pos_map1)
    text_map["bottom"]=(text_set2,text_pos_map2)
    text_map["other"]=(text_set3,text_pos_map3)
    text_map=process_text_map(text_map,removed_segments,segmentation_config)
    isecDic = find_all_intersections(initial_segments,segmentation_config.intersection_epsilon)
    new_segments, edge_map,point_map= split_segments(initial_segments, isecDic,segmentation_config.segment_filter_length)
    #filter lines

    filtered_segments, filtered_edge_map,filtered_point_map= filter_segments(initial_segments,isecDic,point_map,segmentation_config.segment_filter_length,segmentation_config.segment_filter_iters,segmentation_config.segment_remove_interval)
    braket_start_lines=findBraketByHints(filtered_segments,text_map)
    # polys=[]
    # outputLines(new_segments,point_map,polys,segmentation_config.line_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)

    #filtered_segments,filtered_point_map=removeOddPoints(filtered_segments,filtered_point_map,segmentation_config)

    # Step 3: 构建基于分割后线段的图结构
    if verbose:
        print("构建基于分割后线段的图结构")
        print(f"过滤后线段条数:{len(filtered_segments)}")
    graph= build_graph(filtered_segments)

    closed_polys = []

    # 基于角隅孔计算参考边
    #arc_replines = compute_arc_replines(filtered_segments,point_map)
    star_replines,star_pos_map,star_pos=compute_star_replines(filtered_segments,elements)
    #line_replines=compute_line_replines(filtered_segments,filtered_point_map)
    cornor_holes=compute_cornor_holes(filtered_segments,filtered_point_map,segmentation_config)
    cornor_holes=filter_cornor_holes(cornor_holes,filtered_point_map,segmentation_config)
    # for cornor_hole in cornor_holes:
    #     for s in cornor_hole.segments:
    #         s.isCornerhole=True

    if verbose:
        #print(f"圆弧角隅孔个数: {len(arc_replines)}")
        print(f"星形角隅孔个数: {len(star_replines)}")
        print(f"非星形角隅孔个数: {len(cornor_holes)}")
        #print(f"直线形角隅孔个数: {len(line_replines)}")
        print(f"根据标注找到的肘板个数: {len(braket_start_lines)}")
    # for star_repline in star_replines:
    #     print(star_repline)
    #replines=arc_replines+star_replines+line_replines+braket_start_lines
    replines=[]
    for cornor_hole in cornor_holes:
        replines.append(cornor_hole.segments[0])
    star_replines=findBracketByPoints(star_pos,filtered_segments)
    # braket_start_lines=findBracketByPoints(braket_pos,filtered_segments)
    replines=replines+braket_start_lines+star_replines
    replines_set=set()
    #arc_replines_set=set()
    #line_replines_set=set()
    # for arc_rep in arc_replines:
    #     if arc_rep not in arc_replines_set:
    #         arc_replines_set.add(arc_rep)
    #         arc_replines_set.add(DSegment(arc_rep.end_point,arc_rep.start_point))
    # for line_rep in line_replines:
    #     if line_rep not in line_replines_set:
    #         line_replines_set.add(line_rep)
    #         line_replines_set.add(DSegment(line_rep.end_point,line_rep.start_point))
    for rep in replines:
        if rep not in replines_set:
            replines_set.add(rep)
            replines_set.add(DSegment(rep.end_point,rep.start_point))
    # Step 4: 对每个 repline，使用 BFS 查找路径
    # 多线程加速部分
    if verbose:
        pbar = tqdm(total=len(replines), desc="查找闭合路径")

    # 使用 ThreadPoolExecutor
    closed_polys = []
    multi_thread_start_time = time.time()
    errors=[]
    if verbose:
        print(min(32, os.cpu_count())if segmentation_config.max_workers <=0 else segmentation_config.max_workers)
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count()  )if segmentation_config.max_workers <=0 else segmentation_config.max_workers) as executor:
        # 提交任务到线程池
        future_to_repline = {executor.submit(process_repline, repline, graph, segmentation_config): repline for repline in replines}
        
        for future in as_completed(future_to_repline):
            repline = future_to_repline[future]
            try:
                # 获取结果，设置超时时间
                paths = future.result()
                closed_polys.extend(paths)
                # print(f"任务 {repline} 完成")
            except Exception as exc:
                errors.append(f"处理 {repline} 时发生错误: {exc}")
            if verbose:
                pbar.update()
    
    # print("============")
    
    # print("=========")
    # closed_polys=[]
    # for repline in replines:
    #     paths=bfs_paths(graph,repline.start_point,repline.end_point,segmentation_config.path_max_length,timeout=1000)
    #     closed_polys.extend(paths)
    #     pbar.update()
    if verbose:
        pbar.close()
        print(errors)
    multi_thread_end_time = time.time()
    if verbose:
        print(f"回路探测执行部分耗时: {multi_thread_end_time - multi_thread_start_time:.2f} 秒")
        print("查找完毕")
    # for closed_poly in closed_polys:
    #     print(closed_poly)
    #HIDDENs
    if segmentation_config.dfs_optional:
        sampled_lines=[]
        for s in filtered_segments:
            sampled_lines.append(DSegment(s.start_point,s.end_point,s.ref))
            sampled_lines.append(DSegment(s.end_point,s.start_point,s.ref))
        visited_edges=set()
        if verbose:
            pbar=tqdm(desc="sampled_lines",total=len(sampled_lines))
        for sampled_line in sampled_lines:
            if verbose:
                pbar.update()
            visited_edges.add((sampled_line[0],sampled_line[1]))
            path=process_repline_with_repline_dfs(visited_edges,sampled_line,graph,segmentation_config)
            if(len(path)>=segmentation_config.path_min_length):
                closed_polys.append(path)
        if verbose:
            pbar.close()
    #poly simplify
    print(len(closed_polys))

    # 根据边框对多边形进行过滤
    #polys = filterPolys(polys,t=3000,d=5)
    if segmentation_config.bracket_layer is None:
        polys = filterPolys(closed_polys,segmentation_config.path_max_length,segmentation_config.path_min_length,segmentation_config.bbox_min_area,segmentation_config.bbox_max_area,segmentation_config.bbox_ratio)
    print(len(polys))
    # 剔除重复路径
    polys = remove_duplicate_polygons(polys,segmentation_config.eps,segmentation_config.min_samples)
    #print(bcs)
    # print(polys[0])
    # print(polys[1])
    # 仅保留基本路径
    polys = remove_complicated_polygons(polys,segmentation_config.remove_tolerance)
    if verbose:
        print(f"封闭多边形个数:{len(polys)}")
    #outputPolysAndGeometry(filtered_point_map,polys,segmentation_config.poly_image_dir,segmentation_config.draw_polys,segmentation_config.draw_geometry,segmentation_config.draw_poly_nums)
    if segmentation_config.draw_line_image:
        outputLines(segmentation_config,filtered_segments,filtered_point_map,polys,cornor_holes,star_pos ,texts,text_map,dimensions,replines,segmentation_config.line_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys,)
    
    return polys, new_segments, point_map,star_pos_map,cornor_holes,text_map


 
def is_numeric(s):
    return bool(re.match(r'^[0-9]+$', s))
 
def is_r_numeric(s):
    return bool(re.match(r'^R[0-9]+$', s))

def is_material(s):
    return bool(re.match(r"^[A-Z]{2}$",s)) or bool(re.match(r"^~[A-Z]{2}$",s)) or bool(re.match(r"^[0-9]{2}[A-Z]{2}$",s))  or bool(re.match(r"^~[0-9]{2}[A-Z]{2}$",s))
# def isUsefulHints(e):
#     if (not isinstance(e,DText)) and (not isinstance(e,DDimension)):
#         return False
#     if  isBraketHints(e):
#         return False
#     content=e.content if isinstance(e,DText) else e.text
#     return is_numeric(content) or is_r_numeric(content)


# def findAllTextAndDimensions(elements):
#     text_and_dimensions=[e for e in elements if isUsefulHints(e)]
#     return text_and_dimensions


def isUsefulHints(e):
    # flag=True

    #return is_numeric(e.content) or is_r_numeric(e.content) or isBraketHints(e) or is_material(e.content)
    return is_useful_text(e.content)
def findAllTextsAndDimensions(elements):
    texts=[]
    dimensions=[]
    for e in elements:
        if isinstance(e,DDimension):
            dimensions.append(e)
        elif isinstance(e,DText):
            texts.append(e)
    return texts,dimensions
def processTexts(texts):
    ts=[]
    for t in texts:
        if isUsefulHints(t):
        # if True:
            ts.append(t)
    return ts
def processDimensions(dimensions):
    ds=[]
    for d in dimensions:
        type=d.dimtype
        if type==32 or type==33 or type==161 or type==160:
            #转角标注& 对齐标注
            dim_pos=DPoint((d.defpoints[1].x+d.defpoints[2].x)/2,(d.defpoints[1].y+d.defpoints[2].y)/2)
            ds.append([d,dim_pos]) 
        elif type==37 or type==34:
            #三点角度标注
            dim_pos=DPoint(d.defpoints[3].x,d.defpoints[3].y)
            ds.append([d,dim_pos]) 
        elif type==38:
            #坐标标注
            pass
        elif type==163:
            #角度标注
            dim_pos=DPoint((d.defpoints[0].x+d.defpoints[3].x)/2,(d.defpoints[0].y+d.defpoints[3].y)/2)
            ds.append([d,dim_pos]) 
    return ds
