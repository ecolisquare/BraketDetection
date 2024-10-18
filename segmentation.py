from element import *
import random
import matplotlib.pyplot as plt
from utils import *

#read file
elements,ori_segments=readJson("../data/split/FR18.json")
#expand segments
segments=expandFixedLength(ori_segments,0)
#find all the closed polys
polys=findClosedPolys(segments,drawIntersections=True,linePNGPath="../output/inverse_line3.png",drawPolys=True)
