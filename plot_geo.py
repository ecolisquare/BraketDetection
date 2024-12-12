import matplotlib.pyplot as plt
import numpy as np
from element import *
import os

# 输出多边形所在原始图形
def plot_geometry(segments, path):
    fig, ax = plt.subplots()
    for segment in segments:
        if isinstance(segment.ref, DLine):
            x_values = [segment.ref.start_point.x, segment.ref.end_point.x]
            y_values = [segment.ref.start_point.y, segment.ref.end_point.y]
            ax.plot(x_values, y_values, 'b-', lw=2)
        
        elif isinstance(segment.ref, DLwpolyline):
            x_values = [p.x for p in segment.ref.points]
            y_values = [p.y for p in segment.ref.points]
            if segment.ref.isClosed:
                x_values.append(segment.ref.points[0].x)
                y_values.append(segment.ref.points[0].y)
            ax.plot(x_values, y_values, 'g-', lw=2)
        
        elif isinstance(segment.ref, DArc):
            if segment.ref.end_angle < segment.ref.start_angle:
                end_angle = segment.ref.end_angle + 360
            else:
                end_angle = segment.ref.end_angle
            theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
            x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
            y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
            ax.plot(x_arc, y_arc, 'r-', lw=2)

    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()


#输出封闭多边形
def plot_polys(point_map,segments,path):
    fig, ax = plt.subplots()
    # for segment in segments:
    #     vs,ve=segment.start_point,segment.end_point
    #     if len(point_map[vs])>2:
    #         # print(p.x,p.y)
    #         ax.plot(vs.x, vs.y, 'g.')
    #         # ax.text(vs.x, vs.y,str([s.length() for s in point_map[vs]]), fontsize=12, color='blue', rotation=90)
    #     if len(point_map[ve])>2:
    #         ax.plot(ve.x, ve.y, 'g.')
    #         # ax.text(ve.x, ve.y, str([s.length() for s in point_map[ve]]), fontsize=12, color='blue', rotation=90)
    for i, segment in enumerate(segments):
        x_values = [segment.start_point.x, segment.end_point.x]
        y_values = [segment.start_point.y, segment.end_point.y]
        color = 'red' if i % 2 == 0 else 'blue'
        ax.plot(x_values, y_values, color=color, lw=2)  # 使用不同的颜色
    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()


def expandFixedLengthGeo(segList,dist,both=True):


    new_seglist=[] 
    n=len(segList)

    for seg in segList:
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
    return new_seglist

def plot_info_poly(segments, path,texts,dimensions):
    fig, ax = plt.subplots()

    for t_t in texts:
        t=t_t[0]
        pos=t_t[1]
        content=t.content
        ax.text(pos.x, pos.y, content, fontsize=12, color='blue', rotation=0)
    for d_t in dimensions:
        d=d_t[0]
        pos=d_t[1]
        content=d.text
        if d.dimtype==32 or d.dimtype==33 or d.dimtype==161:
            d1,d2,d3,d4=d.defpoints[0], DPoint(d.defpoints[0].x+d.defpoints[1].x-d.defpoints[2].x,d.defpoints[0].y+d.defpoints[1].y-d.defpoints[2].y),d.defpoints[1],d.defpoints[2]
            ss=[DSegment(d1,d4),DSegment(d4,d3),DSegment(d3,d2)]
            sss=[DSegment(d2,d1)]
            ss=expandFixedLengthGeo(ss,25)
            sss=expandFixedLengthGeo(sss,100)
            q=sss[0].end_point
            sss=expandFixedLengthGeo(sss,100,False)
            for s in ss:
                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2)
            for s in sss:
                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2)
            perp_vx, perp_vy = sss[0].start_point.x - sss[0].end_point.x, sss[0].start_point.y-sss[0].end_point.y
            rotation_angle = np.arctan2(-perp_vy, -perp_vx) * (180 / np.pi)
            ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#00FFFF", fontsize=10)
    for i, segment in enumerate(segments):
        if segment.isCornerhole:
            color = "red"
        elif segment.isConstraint:
            color = "green"
        else:
            color = "blue"
        if isinstance(segment.ref, DArc):
            if segment.ref.end_angle < segment.ref.start_angle:
                end_angle = segment.ref.end_angle + 360
            else:
                end_angle = segment.ref.end_angle
            theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
            x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
            y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
            ax.plot(x_arc, y_arc, color, lw=2)
        else:
            x_values = [segment.start_point.x, segment.end_point.x]
            y_values = [segment.start_point.y, segment.end_point.y]
            ax.plot(x_values, y_values, color, lw=2)
    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()

def outputRes(segments,point_map,polys,resPNGPath,drawIntersections=False,drawLines=False,drawPolys=False):
    fig, ax = plt.subplots()
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
            for i, segment in enumerate(poly):
                if segment.isCornerhole:
                    color = "red"
                elif segment.isConstraint:
                    color = "green"
                else:
                    color = "blue"
                if isinstance(segment.ref, DArc):
                    if segment.ref.end_angle < segment.ref.start_angle:
                        end_angle = segment.ref.end_angle + 360
                    else:
                        end_angle = segment.ref.end_angle
                    theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
                    x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
                    y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
                    ax.plot(x_arc, y_arc, color, lw=2)
                else:
                    x_values = [segment.start_point.x, segment.end_point.x]
                    y_values = [segment.start_point.y, segment.end_point.y]
                    ax.plot(x_values, y_values, color, lw=2)

    plt.gca().axis('equal')
    plt.savefig(resPNGPath)
    print(f"结果图保存于:{resPNGPath}")
    fig.clf()

def output_training_img(polys, segments, training_img_output_folder, name):
    # 如果输出文件夹不存在，则创建
    os.makedirs(training_img_output_folder, exist_ok=True)

    for i, poly in enumerate(polys):
        fig, ax = plt.subplots()

        output_file = os.path.join(training_img_output_folder, f"{name}_{i}.png")

        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k-')

        for segment in poly:
            color = "red"

            if isinstance(segment.ref, DArc):
                if segment.ref.end_angle < segment.ref.start_angle:
                    end_angle = segment.ref.end_angle + 360
                else:
                    end_angle = segment.ref.end_angle
                theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
                x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
                y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
                ax.plot(x_arc, y_arc, color, lw=2)
            else:
                x_values = [segment.start_point.x, segment.end_point.x]
                y_values = [segment.start_point.y, segment.end_point.y]
                ax.plot(x_values, y_values, color, lw=2)
            
        plt.gca().axis('equal')
        plt.savefig(output_file)
        plt.close()