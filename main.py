from element import *
import random
import matplotlib.pyplot as plt
from utils import *
from infoextraction import *
import numpy as np
from plot_geo import *
from config import *
# from DGCNN_model import *
from tqdm import tqdm
from classifier import *

if __name__ == '__main__':
    segmentation_config=SegmentationConfig()

    json_path = input("请输入路径: ")
    if segmentation_config.verbose:
        print("读取json文件")
    #文件中线段元素的读取和根据颜色过滤
    elements,ori_segments=readJson(json_path)
    elements,ori_segments=readJson(json_path)
    texts ,dimensions=findAllTextsAndDimensions(elements)
    ori_dimensions=dimensions
    dimensions=processDimensions(dimensions)
    texts=processTexts(texts)
    if segmentation_config.verbose:
        print("json文件读取完毕")
    #将线进行适当扩张
    segments=expandFixedLength(ori_segments,segmentation_config.line_expand_length)

    #找出所有包含角隅孔圆弧的基本环
    polys, new_segments, point_map,star_pos_map,cornor_holes,text_pos_map=findClosedPolys_via_BFS(elements,texts,dimensions,segments,segmentation_config)

    # #预训练的几何分类模型筛选肘板
    # model_path = "/home/user4/BraketDetection/DGCNN/cpkt/geometry_classifier.pth"
    # polys = filter_by_pretrained_DGCNN_Model(polys, model_path)

    # print("DGCNN筛选后剩余回路: ", len(polys))
    # outputPolysAndGeometry(polys,segmentation_config.poly_image_dir,segmentation_config.draw_polys,segmentation_config.draw_geometry,segmentation_config.draw_poly_nums)



    #结构化输出每个肘板信息
    polys_info = []
    pbar=tqdm(total=len(polys),desc="正在输出结构化信息")
    for i, poly in enumerate(polys):
        res = outputPolyInfo(poly, new_segments, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_pos_map)
        pbar.update()
        if res is not None:
            polys_info.append(res)
    pbar.close()
    print("结构化信息输出完毕，保存于:", segmentation_config.poly_info_dir)

    outputRes(new_segments, point_map, polys_info, segmentation_config.res_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)