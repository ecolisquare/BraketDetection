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
import argparse
from load import dxf2json

def main():
    segmentation_config=SegmentationConfig()
    verbose=segmentation_config.verbose
    parser = argparse.ArgumentParser(description="肘板检测工具")
    parser.add_argument("-i", "--input", required=True, help="DXF 文件夹路径")
    args = parser.parse_args()
    dxf_path = args.input
    print("loading...")
    dxf2json(os.path.dirname(dxf_path),os.path.basename(dxf_path),os.path.dirname(dxf_path))
    json_path = os.path.join(os.path.dirname(dxf_path), (os.path.basename(dxf_path).split('.')[0] + ".json"))
    base, ext = os.path.splitext(json_path)
    segmentation_config.multi_json_path = f"{base}_multi.json"
    print("complete loading!")
    segmentation_config.json_path = json_path
    create_folder_safe(f"{segmentation_config.poly_info_dir}")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/标准肘板详细信息参考图")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/所有肘板图像(仅限开发模式)")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/所有有效回路图像")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/非标准肘板")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/标准肘板")
    create_folder_safe(f"{segmentation_config.poly_info_dir}/标准肘板(无分类)")

    if segmentation_config.verbose:
        print("读取json文件")
    #文件中线段元素的读取和根据颜色过滤
    elements,segments,ori_segments,stiffeners,sign_handles,polyline_handles, hatch_polys,jg_s=readJson(json_path,segmentation_config)
    hole_polys = get_hole_text_coor(json_path, segmentation_config.hole_layer)
    # print(sign_handles)
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
    polys, new_segments, point_map,star_pos_map,cornor_holes,text_map,removed_handles=findClosedPolys_via_BFS(elements,texts,dimensions,segments,sign_handles,segmentation_config)

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
        res = calculate_poly_features(poly, segments_nearby, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_map,stiffeners, hatch_polys, hole_polys,jg_s)
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
    polys_info,classi_res,flags=classificationAndOutputStep(indices,edges_infos,poly_centroids,hint_infos,meta_infos,segmentation_config,polys,polyline_handles)
    
    # 获得需要去重肘板的id
    delete_bracket_ids = find_dump_bracket_ids(polys_info, classi_res, indices)
    
    free_edge_handles = []
    all_handles=[]
    not_all_handles=[]
    non_free_edge_handles = []
    for idx,(poly_refs,cls,flag) in enumerate(zip(polys_info,classi_res,flags)):
        if cls=='Unclassified' or cls=='Unstandard':
            continue
        else:
            for seg in poly_refs:
                if seg.isConstraint == False and seg.isCornerhole == False:
                    free_edge_handles.append(seg.ref.handle)
                else:
                    non_free_edge_handles.append(seg.ref.handle)
            if len(cls.split(','))==1 and flag:
                for seg in poly_refs:
                    all_handles.append(seg.ref.handle)
            else:
                for seg in poly_refs:
                    not_all_handles.append(seg.ref.handle)
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
    draw_rectangle_in_dxf(dxf_path, dxf_output_folder, bboxs, classi_res,indices, free_edge_handles,non_free_edge_handles,all_handles,not_all_handles,removed_handles,delete_bracket_ids)

if __name__ == "__main__":
    print("肘板检测工具启动！")
    from multiprocessing import freeze_support
    freeze_support()
    main()
