from element import *
import random
import matplotlib.pyplot as plt
from utils import *
import numpy as np

from config import *

segmentation_config=SegmentationConfig()

json_path = input("请输入路径")
if segmentation_config.verbose:
    print("读取json文件")
#文件中线段元素的读取和根据颜色过滤
elements,ori_segments=readJson(json_path)
if segmentation_config.verbose:
    print("json文件读取完毕")
#将线进行适当扩张
segments=expandFixedLength(ori_segments,segmentation_config.line_expand_length)

#找出所有包含角隅孔圆弧的基本环
polys=findClosedPolys_via_BFS(elements,segments,segmentation_config)

