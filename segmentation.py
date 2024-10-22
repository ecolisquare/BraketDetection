from element import *
import random
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from plot_geo import plot_geometry,plot_polys

#文件中线段元素的读取和根据颜色过滤
elements,ori_segments=readJson("/home/user4/jndata/FR19.json")

#将线进行适当扩张
segments=expandFixedLength(ori_segments,15)

#找出所有包含角隅孔圆弧的基本环
polys=findClosedPolys_via_BFS(segments,drawIntersections=False,linePNGPath="./output/line.png")


n=10
for i,poly in enumerate(polys):
    if i>=n:
        break
    plot_geometry(poly,f"./output/geometry{i}.png")
   
for i,poly in enumerate(polys):
    if i>=n:
        break
    plot_polys(poly,f"./output/poly{i}.png")
 
