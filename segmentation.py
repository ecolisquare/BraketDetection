from element import *
import random
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from plot_geo import plot_geometry,plot_polys

#文件中线段元素的读取和根据颜色过滤
elements,ori_segments=readJson("/home/user10/code/DXFStruct/测试数据/图纸测试.json")

#将线进行适当扩张
segments=expandFixedLength(ori_segments,15)

#找出所有基本环
polys=findClosedPolys(segments,drawIntersections=True,linePNGPath="/home/user10/code/BraketDetection/output/line.png")


n=10
for i,poly in enumerate(polys):
    if i>=n:
        break
    plot_geometry(poly,f"/home/user10/code/BraketDetection/output/geometry{i}.png")
   
for i,poly in enumerate(polys):
    if i>=n:
        break
    plot_polys(poly,f"/home/user10/code/BraketDetection/output/poly{i}.png")
 
