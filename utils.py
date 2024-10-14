import json 
from  element import *
import math
from SweepIntersectorLib.SweepIntersector import SweepIntersector
import matplotlib.pyplot as plt
import networkx as nx

def angleOfTwoVectors(A,B):
    lengthA = math.sqrt(A[0]**2 + A[1]**2)  
    lengthB = math.sqrt(B[0]**2 + B[1]**2)  
    dotProduct = A[0] * B[0] + A[1] * B[1]   
    angle = math.acos(dotProduct / (lengthA * lengthB))
    angle_degrees = angle * (180 / math.pi)  
    return angle_degrees  
#json --> elements
def readJson(path):
    elements=[]
    segments=[]
    color = [3, 7, 4]
    try:  
        with open(path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)[0]  
        
        for ele in data_list:
            e=None
            if ele["color"] not in color:
                continue
            if ele["type"]=="line":
                e=DLine(DPoint(ele["start"][0],ele["start"][1]),DPoint(ele["end"][0],ele["end"][1]),ele["color"])
                segments.append(DSegment(e.start_point,e.end_point,e))
            # elif ele["type"]=="arc":
            #     e=DArc(DPoint(ele["center"][0],ele["center"][1]),ele["radius"],ele["startAngle"],ele["endAngle"])
            #     A=e.start_point.as_tuple()
            #     B=e.end_point.as_tuple()
            #     O=e.center.as_tuple()
            #     OA=(A[0]-O[0],A[1]-O[1])
            #     OB=(B[0]-O[0],B[1]-O[1])
            #     angle=angleOfTwoVectors(OA,OB)
            #     edges=[]
            #     if angle<=45:
            #         #AB
            #         segments.append(DSegment(e.start_point,e.end_point,e))
            #     elif angle<=90:
            #         OC=(OA[0]+OB[0],OA[1]+OB[1])
            #         l=math.sqrt(OC[0]**2 + OC[1]**2)
            #         OC=(OC[0]/l*e.radius,OC[1]/l*e.radius)
            #         C=DPoint(OC[0]+O[0],OC[1]+O[1]) 
            #         #AC CB
            #         segments.append(DSegment(e.start_point,DPoint(C[0],C[1]),e))
            #         segments.append(DSegment(DPoint(C[0],C[1]),e.end_point,e))
            #     else:
            #         OC=(OA[0]+OB[0],OA[1]+OB[1])
            #         l=math.sqrt(OC[0]**2 + OC[1]**2)
            #         OC=(OC[0]/l*e.radius,OC[1]/l*e.radius)
            #         C=DPoint(OC[0]+O[0],OC[1]+O[1]) 

            #         OD=(OA[0]+OC[0],OA[1]+OC[1])
            #         l=math.sqrt(OD[0]**2 + OD[1]**2)
            #         OD=(OD[0]/l*e.radius,OD[1]/l*e.radius)
            #         D=DPoint(OD[0]+O[0],OD[1]+O[1]) 

            #         OE=(OC[0]+OB[0],OC[1]+OB[1])
            #         l=math.sqrt(OE[0]**2 + OE[1]**2)
            #         OE=(OE[0]/l*e.radius,OE[1]/l*e.radius)
            #         E=DPoint(OE[0]+O[0],OE[1]+O[1]) 
            #         #AD DC CE EB

            #         segments.append(DSegment(e.start_point,DPoint(D[0],D[1]),e))
            #         segments.append(DSegment(DPoint(D[0],D[1]),DPoint(C[0],C[1]),e))
            #         segments.append(DSegment(DPoint(C[0],C[1]),DPoint(E[0],E[1]),e))
            #         segments.append(DSegment(DPoint(E[0],E[1]),e.end_point,e))
            elif ele["type"] == "arc":
                # 创建DArc对象
                e = DArc(DPoint(ele["center"][0], ele["center"][1]), ele["radius"], ele["startAngle"], ele["endAngle"])
                A = e.start_point.as_tuple()
                B = e.end_point.as_tuple()
                O = e.center.as_tuple()
                
                # 计算角度差
                start_angle = ele["startAngle"]
                end_angle = ele["endAngle"]
                total_angle = end_angle - start_angle
                
                # 定义分段数量，可以根据角度总长动态决定（这里的分段数可以自行调整）
                num_segments = max(2, int(total_angle / 45))  # 每10度一个分段
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
                vs=ele["vertices"]
                ps=[]
                for v in vs:
                    ps.append(DPoint(v[0],v[1]))
                e=DLwpolyline(ps,ele["color"],ele["isClosed"])
                l =len(ps)
                for i in range(l-1):
                    segments.append(DSegment(ps[i],ps[i+1],e))
                if ele["isClosed"]:
                    segments.append(DSegment(ps[-1],ps[0],e))
            else:
                pass
            if e is not None:
                elements.append(e)
        return elements,segments
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")


def expandFixedLength(segList,dist):


    new_seglist=[] 
    for seg in segList:
        p1=seg[0]
        p2=seg[1]
        v=(p2[0]-p1[0],p2[1]-p1[1])
        l=math.sqrt(v[0]*v[0]+v[1]*v[1])
        v=(v[0]/l*dist,v[1]/l*dist)
        new_seglist.append(DSegment(DPoint(p1[0]-v[0],p1[1]-v[1]),DPoint(p2[0]+v[0],p2[1]+v[1]),seg.ref))
    return new_seglist

def remove_duplicates(input_list):  
    seen = set()  
    result = []  
    for item in input_list:  
        if item not in seen:  
            seen.add(item)  
            result.append(item)  
    return result  
  
def findClosedPolys(segments,drawIntersections=False,linePNGPath="./line.png",drawPolys=False,polyPNGPath="./poly.png"):
    lines_dict={}
    points_dict={}
    # compute intersections
    isector = SweepIntersector()
    isecDic = isector.findIntersections(segments,lines_dict,points_dict)
    #isecDic filter
    for seg,isects in isecDic.items():
        isecDic[seg]=remove_duplicates(isects)

    #find all the edges
    edge_set=set()
    for seg,isects in isecDic.items():
        l=len(isects)
        for i in range(l-1):
            # print(isects[i],"  to  ",isects[i+1])
            edge_set.add(DSegment(isects[i],isects[i+1],seg.ref))


    edge_list=list(edge_set)


    edge_map={}
    for e in edge_list:
        edge_map[DSegment(e.start_point,e.end_point)]=e
        edge_map[DSegment(e.end_point,e.start_point)]=e

    # print("========================")
    # print(edges)
    # 创建无向图并添加节点和边
    G = nx.Graph()
    # 添加一些边，形成多个闭合路径
    G.add_edges_from(edge_list)
    # 查找所有的基本环
    cycles = nx.cycle_basis(G)

    polys=[]
    # 打印出所有的闭合路径
    print("Found cycles (closed paths):")
    for cycle in cycles:
        poly=[]
        l=len(cycle)
        print(f"vertex count:{l}")
        for i in range(l-1):
            poly.append(edge_map[DSegment(cycle[i],cycle[i+1])])
        poly.append(edge_map[DSegment(cycle[-1],cycle[0])])
        polys.append(poly)
        # print(f"poly:{poly}")
    if drawIntersections:
        # plot original segments
        for seg in segments:
            vs,ve = seg
            plt.plot([vs[0],ve[0]],[vs[1],ve[1]],'k:')

        # plot intersection points
        for seg,isects in isecDic.items():
            for p in isects[1:-1]:      
                plt.plot(p[0],p[1],'r.')
                
        plt.gca().axis('equal')
        plt.savefig(linePNGPath)
    if drawPolys:
        pass
    print(len(polys))

    return polys