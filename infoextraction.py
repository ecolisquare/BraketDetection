from  element import *
from plot_geo import plot_geometry,plot_polys, plot_info_poly
import os
from utils import segment_intersection,computeBoundingBox
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
                k_1 = math.pi/2 if dx_1 == 0 else math.atan(dy_1 / dx_1)
                dx_2 = refs[-1].end_point.x - refs[-1].start_point.x
                dy_2 = refs[-1].end_point.y - refs[-1].start_point.y
                k_2 = math.pi/2 if dx_2 == 0 else math.atan(dy_2 / dx_2)

                # 判断斜率是否相等
                if are_equal_with_tolerance(k_1, k_2) or (math.fabs(dx_1) <0.025 and math.fabs(dx_2) <0.025):
                    if refs[-1].start_point==segment.end_point:
                        new_segment = DSegment(
                            refs[-1].end_point,     # 保留原本的起点
                            segment.start_point,        # 当前segment的终点
                            refs[-1].ref              # 保留原本的ref
                        )
                    elif refs[-1].start_point==segment.start_point:
                        new_segment = DSegment(
                            refs[-1].end_point,     # 保留原本的起点
                            segment.end_point,        # 当前segment的终点
                            refs[-1].ref              # 保留原本的ref
                        )
                    elif refs[-1].end_point==segment.end_point:
                        new_segment = DSegment(
                            refs[-1].start_point,     # 保留原本的起点
                            segment.start_point,        # 当前segment的终点
                            refs[-1].ref              # 保留原本的ref
                        )
                    else:
                            
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
    # if isinstance(refs[0].ref, DArc)  and isinstance(refs[-1].ref, DArc):
    #     if(refs[0].ref.start_angle, refs[0].ref.end_angle, refs[0].ref.center, refs[0].ref.radius) == (refs[-1].ref.start_angle, refs[-1].ref.end_angle, refs[-1].ref.center, refs[-1].ref.radius):
    #         del refs[-1]
    # elif isinstance(refs[0].ref, DArc):
    #     pass
    # else:
        
    #     if len(refs) != 0:
    #         dx_1 = segment.end_point.x - segment.start_point.x
    #         dy_1 = segment.end_point.y - segment.start_point.y
    #         k_1 = math.pi/2 if dx_1 == 0 else math.atan(dy_1 / dx_1)
    #         dx_2 = refs[-1].end_point.x - refs[-1].start_point.x
    #         dy_2 = refs[-1].end_point.y - refs[-1].start_point.y
    #         k_2 = math.pi/2 if dx_2 == 0 else math.atan(dy_2 / dx_2)

    #         # 判断斜率是否相等
    #         if are_equal_with_tolerance(k_1, k_2) or (math.fabs(dx_1) <0.025 and math.fabs(dx_2) <0.025):
    #             if refs[-1].start_point==refs[0].end_point:
    #                 new_segment = DSegment(
    #                     refs[-1].end_point,     # 保留原本的起点
    #                     refs[0].start_point,        # 当前segment的终点
    #                     refs[-1].ref              # 保留原本的ref
    #                 )
    #             elif refs[-1].start_point==refs[0].start_point:
    #                 new_segment = DSegment(
    #                     refs[-1].end_point,     # 保留原本的起点
    #                     refs[0].end_point,        # 当前segment的终点
    #                     refs[-1].ref              # 保留原本的ref
    #                 )
    #             elif refs[-1].end_point==refs[0].end_point:
    #                 new_segment = DSegment(
    #                     refs[-1].start_point,     # 保留原本的起点
    #                     refs[0].start_point,        # 当前segment的终点
    #                     refs[-1].ref              # 保留原本的ref
    #                 )
    #             else:
                        
    #                 new_segment = DSegment(
    #                     refs[-1].start_point,     # 保留原本的起点
    #                     refs[0].end_point,        # 当前segment的终点
    #                     refs[-1].ref              # 保留原本的ref
    #                 )
                
    #             # 替换refs中的第一个Segment
    #             refs[0] = new_segment
    #             del refs[-1]
    return refs

def textsInPoly(text_and_dimensions,poly):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    xx=(x_max-x_min)*0.25
    yy=(y_max-y_min)*0.25
    x_min=x_min-xx
    x_max=x_max+xx
    y_max=y_max+yy
    y_min=y_min-yy
    ts=[]
    for t in text_and_dimensions:
        pos=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2) if isinstance(t,DText) else t.textpos
        if x_min <= pos.x and pos.x <=x_max and y_min <=pos.y and y_max>=pos.y:
            ts.append(t)
    return ts

def braketTextInPoly(braket_texts,braket_pos,poly):
    x_min,x_max,y_min,y_max=computeBoundingBox(poly)
    xx=(x_max-x_min)*0.25
    yy=(y_max-y_min)*0.25
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

def outputPolyInfo(poly, segments, segmentation_config, point_map, index,star_pos_map,cornor_holes,text_and_dimensions,braket_texts,braket_pos):
    # step1: 计算几何中心坐标
    poly_centroid = calculate_poly_centroid(poly)

    # step2: 合并边界线
    poly_refs = calculate_poly_refs(poly)

    # step3: 标记角隅孔
    # for corner_hole in cornor_holes:
    #     for seg in corner_hole.segments:
    #         seg.isCornerHole=True
    # print(len(cornor_holes))
    cornor_holes_map={}
    for cornor_hole in cornor_holes:
        for s in cornor_hole.segments:
            cornor_holes_map[s]=cornor_hole.ID
            cornor_holes_map[DSegment(s.end_point,s.start_point,s.ref)]=cornor_hole.ID
    for seg in poly_refs:
        if seg in cornor_holes_map:
            seg.isCornerhole=True
            

    
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
        if segment.ref.color == 3:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 角隅孔确定
        elif poly_refs[(i - 1) % len(poly_refs)].isCornerhole or poly_refs[(i + 1) % len(poly_refs)].isCornerhole:
            segment.isConstraint = True
            poly_refs[i].isConstraint = True
        # 平行线确定
        else:
            dx_1 = segment.end_point.x - segment.start_point.x
            dy_1 = segment.end_point.y - segment.start_point.y
            l=(dx_1**2+dy_1**2)**0.5
            v_1=(dy_1/l*50.0,-dx_1/l*50.0)
            for j,other in  enumerate(segments):
                if segment==other:
                    continue
                dx_2 = other.end_point.x - other.start_point.x
                dy_2 = other.end_point.y - other.start_point.y
                
                # 计算斜率
                k_1 = math.pi/2 if dx_1 == 0 else math.atan(dy_1 / dx_1)
                k_2 = math.pi/2 if dx_2 == 0 else math.atan(dy_2 / dx_2)
                if (math.fabs(dx_1) <=0.025 and math.fabs(dx_2) <=0.025) or (math.fabs(k_1-k_2)<=0.1):
                    s1=DSegment(DPoint(segment.start_point.x+v_1[0],segment.start_point.y+v_1[1]),DPoint(segment.start_point.x-v_1[0],segment.start_point.y-v_1[1]))
                    s2=DSegment(DPoint(segment.end_point.x+v_1[0],segment.end_point.y+v_1[1]),DPoint(segment.end_point.x-v_1[0],segment.end_point.y-v_1[1]))                                
                    i1=segment_intersection(s1.start_point,s1.end_point,other.start_point,other.end_point)
                    if i1==other.end_point or i1==other.start_point:
                        i1=None
                    i2=segment_intersection(s2.start_point,s2.end_point,other.start_point,other.end_point)
                    if i2==other.end_point or i2==other.start_point:
                        i2=None
                    if (i1 is not None) or (i2 is not None):
                        segment.isConstraint = True
                        poly_refs[i].isConstraint = True
                        break
                       

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
            if poly_refs[0].isConstraint and not poly_refs[0].isCornerhole:
                constraint_edges[0] = current_edge + constraint_edges[0]  # 合并到第一条固定边
                edges[0] = current_edge + edges[0]
            else:
                constraint_edges.append(current_edge)
                edges.append(current_edge)
        elif current_type == 'free':
            if not poly_refs[0].isConstraint and not poly_refs[0].isCornerhole and len(free_edges)>0:
                free_edges[0] = current_edge + free_edges[0]  # 合并到第一条自由边
                edges[0] = current_edge + edges[0]
            else:
                free_edges.append(current_edge)
                edges.append(current_edge)
        elif current_type == 'cornerhole':
            if poly_refs[0].isCornerhole:
                cornerhole_edges[0] = current_edge + cornerhole_edges[0]
            else:
                cornerhole_edges.append(current_edge)
                edges.append(current_edge)


  
    # step 5.5：找到所有的标注
    ts=textsInPoly(text_and_dimensions,poly)
    bs,bps=braketTextInPoly(braket_texts,braket_pos,poly)
    # step6: 绘制对边分类后的几何图像
    plot_info_poly(poly_refs, os.path.join(segmentation_config.poly_info_dir, f'infopoly{index}.png'),ts,bs,bps)
    # 如果有多于两条的自由边则判定为不是肘板，不进行输出
    if len(free_edges) > 1:
        print(f"回路{index}超过两条自由边！")
        #return poly_refs
        return None
    if len(free_edges) == 0:
        print(f"回路{index}没有自由边！")
        #return poly_refs
        return  None

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

    # step10: TODO：输出周围标注信息
    k=0
    for i,t in enumerate(ts):
        content=t.content if isinstance(t,DText) else t.text
        pos=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2) if isinstance(t,DText) else t.textpos
        log_to_file(file_path,f"标注{i+1}:")
        log_to_file(file_path,f"位置: {pos}、内容: {content}、颜色: {t.color}、句柄: {t.handle}")
        k+=1
    for i,t in enumerate(bs):
        content=t.content if isinstance(t,DText) else t.text
        pos=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2) if isinstance(t,DText) else t.textpos
        log_to_file(file_path,f"标注{i+1+k}:")
        log_to_file(file_path,f"位置: {pos}、内容: {content}、颜色: {t.color}、句柄: {t.handle}")
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

    return poly_refs
