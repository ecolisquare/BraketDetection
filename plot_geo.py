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
    
    for segment in segments:
        x_values = [segment.start_point.x, segment.end_point.x]
        y_values = [segment.start_point.y, segment.end_point.y]
        ax.plot(x_values, y_values, 'b-', lw=2)
    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()