from element import *
import random
import matplotlib.pyplot as plt
from utils import *

#read file
elements,ori_segments=readJson("./data/split/A-1.json")
#expand segments
segments=expandFixedLength(ori_segments,15)
#find all the closed polys
polys=findClosedPolys(segments,drawIntersections=True,linePNGPath="./output/line.png")