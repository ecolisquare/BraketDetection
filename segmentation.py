from element import *
import random
import matplotlib.pyplot as plt
from utils import *
import numpy as np
from plot_geo import plot_geometry

#文件中线段元素的读取和根据颜色过滤
elements,ori_segments,arc_repline=readJson("/home/user4/jndata/FR18-3.json")

#将线进行适当扩张
segments=expandFixedLength(ori_segments,5)

#找出所有包含角隅孔圆弧的基本环
polys=findClosedPolys(segments,arc_repline,drawIntersections=True,linePNGPath="./output/line.png")


for i,poly in enumerate(polys):
    plot_geometry(poly,f"./output/{i}.png")
