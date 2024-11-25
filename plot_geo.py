import matplotlib.pyplot as plt
import numpy as np
from element import *

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
def plot_polys(segments,path):
    fig, ax = plt.subplots()
    for i, segment in enumerate(segments):
        x_values = [segment.start_point.x, segment.end_point.x]
        y_values = [segment.start_point.y, segment.end_point.y]
        color = 'red' if i % 2 == 0 else 'blue'
        ax.plot(x_values, y_values, color=color, lw=2)  # 使用不同的颜色
    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()

def plot_info_poly(segments, path,texts,braket_texts,braket_pos):
    fig, ax = plt.subplots()
    for p in braket_pos:
        ax.plot(p.x,p.y,'g.')
    for t in texts:
        content=t.content if isinstance(t,DText) else t.text
        pos=DPoint((t.bound["x1"]+t.bound["x2"])/2,(t.bound["y1"]+t.bound["y2"])/2) if isinstance(t,DText) else t.textpos
        ax.text(pos.x, pos.y, content, fontsize=12, color='blue', rotation=0)
    for i,t in enumerate(braket_texts):
        content=t.content if isinstance(t,DText) else t.text
        ax.text(braket_pos[i].x, braket_pos[i].y+10, content, fontsize=12, color='green', rotation=0)
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