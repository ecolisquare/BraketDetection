from element import *
import random
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from plot_geo import plot_geometry,plot_polys

print("reading json---")
#文件中线段元素的读取和根据颜色过滤
elements,ori_segments=readJson("/home/user10/code/DXFStruct/测试数据/split/FR22-1.json")
print("json read!")
#将线进行适当扩张
segments=expandFixedLength(ori_segments,10)

#找出所有包含角隅孔圆弧的基本环
polys=findClosedPolys_via_BFS(segments,drawIntersections=True,linePNGPath="/home/user10/code/BraketDetection/output/line.png")

n=1000

for i,poly in enumerate(polys):
    if i>=n:
        break
    plot_geometry(poly,f"/home/user10/code/BraketDetection/output/geometry{i}.png")
   
for i,poly in enumerate(polys):
    if i>=n:
        break
    plot_polys(poly,f"/home/user10/code/BraketDetection/output/poly{i}.png")
 
