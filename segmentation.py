from element import *
import random
import matplotlib.pyplot as plt
from utils import *
from infoextraction import *
import numpy as np
from plot_geo import *
from config import *

segmentation_config=SegmentationConfig()

json_path = input("请输入路径: ")
if segmentation_config.verbose:
    print("读取json文件")
#文件中线段元素的读取和根据颜色过滤
elements,ori_segments=readJson(json_path)
# braket_texts=findBraketByHints(elements)
# for e in elements:
#     if isinstance(e,DText):
#         texts.append(e)
#         print(e)
# print(len(texts))
# for t in braket_texts:
#     print(t)
# print(len(braket_texts))
if segmentation_config.verbose:
    print("json文件读取完毕")
#将线进行适当扩张
segments=expandFixedLength(ori_segments,segmentation_config.line_expand_length)

#找出所有包含角隅孔圆弧的基本环
polys, new_segments, point_map,star_pos_map,cornor_holes=findClosedPolys_via_BFS(elements,segments,segmentation_config)

#结构化输出每个肘板信息
polys_info = []
print("正在输出结构化信息...")
for i, poly in enumerate(polys):
    res = outputPolyInfo(poly, new_segments, segmentation_config, point_map, i, star_pos_map, cornor_holes)
    if res is not None:
        polys_info.append(res)

print("结构化信息输出完毕，保存于:", segmentation_config.poly_info_dir)

outputRes(new_segments, point_map, polys_info, segmentation_config.res_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)