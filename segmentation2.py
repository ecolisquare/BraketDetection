from element import *
import random
import matplotlib.pyplot as plt
from utils import *
from infoextraction2 import *
import numpy as np
from plot_geo import *
from config import *
# from DGCNN_model import *
from tqdm import tqdm
from classifier import *
from draw_dxf import *

if __name__ == '__main__':
    segmentation_config=SegmentationConfig()
    verbose=segmentation_config.verbose
    json_path = input("请输入路径: ")
    segmentation_config.json_path = json_path
    if segmentation_config.verbose:
        print("读取json文件")
    #文件中线段元素的读取和根据颜色过滤
    elements,segments,ori_segments,stiffeners=readJson(json_path,segmentation_config)
   
    ori_block=build_initial_block(ori_segments,segmentation_config)
    # grid,meta=segments_in_blocks(ori_segments,segmentation_config)
    # for row in grid:
    #     rows=[]
    #     for col in row:
    #         rows.append(len(col))
    #     print(rows)


    texts ,dimensions=findAllTextsAndDimensions(elements)
    
    ori_dimensions=dimensions
    dimensions=processDimensions(dimensions)
    texts=processTexts(texts)
    bk_code_pos=find_bkcode(texts)
    if segmentation_config.verbose:
        print("json文件读取完毕")
    

    #找出所有包含角隅孔圆弧的基本环
    polys, new_segments, point_map,star_pos_map,cornor_holes,text_map=findClosedPolys_via_BFS(elements,texts,dimensions,segments,segmentation_config)

    # #预训练的几何分类模型筛选肘板
    # model_path = "/home/user4/BraketDetection/DGCNN/cpkt/geometry_classifier.pth"
    # polys = filter_by_pretrained_DGCNN_Model(polys, model_path)

    # print("DGCNN筛选后剩余回路: ", len(polys))
    # outputPolysAndGeometry(polys,segmentation_config.poly_image_dir,segmentation_config.draw_polys,segmentation_config.draw_geometry,segmentation_config.draw_poly_nums)

    #结构化输出每个肘板信息
    edges_infos,poly_centroids,hint_infos,meta_infos=[],[],[],[]
    indices=[]
    pbar=tqdm(total=len(polys),desc="正在输出结构化信息")
    for i, poly in enumerate(polys):
        # try:
        #     res = outputPolyInfo(poly, ori_segments, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_pos_map)
        # except Exception as e:
        #     res=None

        #     print(e)
        # segments_nearby,blocks=segments_near_poly(poly,grid,meta)
        
        # visualize_grid_and_segment(segments_nearby, poly,meta[0],meta[1],meta[2], blocks)
        # print(len(segments_nearby))
        # try:
        #     segments_nearby=ori_block.segments_near_poly(poly)
        #     res = outputPolyInfo(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners)
        # except Exception as e:
        #     res=None
        segments_nearby=ori_block.segments_near_poly(poly)
        res = calculate_poly_features(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners)
        pbar.update()
        if res is not None:
            # print(res)
            edges_info,poly_centroid,hint_info,meta_info=res
            edges_infos.append(edges_info)
            poly_centroids.append(poly_centroid)
            hint_infos.append(hint_info)
            meta_infos.append(meta_info)
            indices.append(i)
    pbar.close()
    
    code_map=calculate_codemap(edges_infos,poly_centroids,hint_infos,meta_infos,bk_code_pos)

    edges_infos,poly_centroids,hint_infos,meta_infos=hint_search_step(edges_infos,poly_centroids,hint_infos,meta_infos,code_map)
  
    edges_infos,poly_centroids,hint_infos,meta_infos=diffusion_step(edges_infos,poly_centroids,hint_infos,meta_infos)

    polys_info,classi_res=classificationAndOutputStep(indices,edges_infos,poly_centroids,hint_infos,meta_infos,segmentation_config)

    free_edge_handles = []
    for idx,(poly_refs,cls) in enumerate(zip(polys_info,classi_res)):
        if cls=='Unclassified' or cls=='Unstandard':
            continue
        else:
            for seg in poly_refs:
                if seg.isConstraint == False and seg.isCornerhole == False:
                    free_edge_handles.append(seg.ref.handle)
    
    bboxs = []
    for poly_refs in polys_info:
        max_x = float('-inf')
        min_x = float('inf')
        max_y = float('-inf')
        min_y = float('inf')
        for seg in poly_refs:
            # 提取起点和终点的横纵坐标
            x_coords = [seg.start_point[0], seg.end_point[0]]
            y_coords = [seg.start_point[1], seg.end_point[1]]

            # 更新最大最小值
            max_x = max(max_x, *x_coords)
            min_x = min(min_x, *x_coords)
            max_y = max(max_y, *y_coords)
            min_y = min(min_y, *y_coords)

        bbox = [[min_x, min_y], [max_x, max_y]]
        bboxs.append(bbox)
    
    dxf_path = os.path.splitext(segmentation_config.json_path)[0] + '.dxf'
    dxf_output_folder = segmentation_config.dxf_output_folder
    draw_rectangle_in_dxf(dxf_path, dxf_output_folder, bboxs, classi_res,indices, free_edge_handles)