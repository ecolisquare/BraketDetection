from element import *
import random
import matplotlib.pyplot as plt
from utils import *

#文件中线段元素的读取和根据颜色过滤
elements,ori_segments=readJson("/home/user4/jndata/all.json")

#将线进行适当扩张
segments=expandFixedLength(ori_segments,15)

#找出所有基本环
polys=findClosedPolys(segments,drawIntersections=True,linePNGPath="./output/line.png")

