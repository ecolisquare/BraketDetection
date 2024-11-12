from element import *
import matplotlib.pyplot as plt
from utils import *
from infoextraction import *
import numpy as np
import os
from plot_geo import *

from config import *


def output_training_data(polys, training_data_output_path):
    # 如果输出文件夹不存在，则创建
    os.makedirs(training_data_output_path, exist_ok=True)
    
    for i, poly in enumerate(polys):
        # 计算多边形中心坐标
        num_points = 0
        center_x, center_y = 0.0, 0.0
        
        for seg in poly:
            (start_x, start_y), (end_x, end_y), length = seg.start_point, seg.end_point, seg.length()
            center_x += start_x + end_x
            center_y += start_y + end_y
            num_points += 2  # 每个seg有两个点
        
        # 计算中心坐标
        center_x /= num_points
        center_y /= num_points

        # 创建文件并输出每个线段信息
        output_file = os.path.join(training_data_output_path, f"{i}.txt")
        with open(output_file, "w") as f:
            for seg in poly:
                (start_x, start_y), (end_x, end_y), length = seg.start_point, seg.end_point, seg.length()
                
                # 平移线段的起点和终点
                shifted_start_x = start_x - center_x
                shifted_start_y = start_y - center_y
                shifted_end_x = end_x - center_x
                shifted_end_y = end_y - center_y
                
                # 将平移后的线段信息写入文件，保留两位小数
                f.write(f"{shifted_start_x:.2f} {shifted_start_y:.2f} {shifted_end_x:.2f} {shifted_end_y:.2f} {length:.2f}\n")
            

def process_json_files(folder_path, output_foler, training_data_output_folder):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"路径 {folder_path} 不存在或不是一个文件夹。")
        return
    
    # 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_foler, name)
            training_data_output_path = os.path.join(training_data_output_folder, name)
            print(f"正在处理文件: {file_path}")
            
            # 打开并读取JSON文件内容
            try:
                process_json_data(file_path, output_path, training_data_output_path)  # 对数据进行操作
            except json.JSONDecodeError as e:
                print(f"解析JSON文件 {file_path} 时出错: {e}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

def process_json_data(json_path, output_path, training_data_output_path):
    segmentation_config=SegmentationConfig()
    segmentation_config.line_image_path = os.path.join(output_path, "line.png")
    segmentation_config.poly_image_dir = os.path.join(output_path, "poly_image")
    segmentation_config.poly_info_dir = os.path.join(output_path, "poly_info")
    segmentation_config.res_image_path = os.path.join(output_path, "res.png")

    try:
        os.makedirs(segmentation_config.poly_image_dir, exist_ok=True)
        os.makedirs(segmentation_config.poly_info_dir, exist_ok=True)
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

    if segmentation_config.verbose:
        print("读取json文件")
    #文件中线段元素的读取和根据颜色过滤
    elements,ori_segments=readJson(json_path)
    if segmentation_config.verbose:
        print("json文件读取完毕")
    #将线进行适当扩张
    segments=expandFixedLength(ori_segments,segmentation_config.line_expand_length)

    #找出所有包含角隅孔圆弧的基本环
    ppolys, new_segments, point_map,star_pos_map,cornor_holes=findClosedPolys_via_BFS(elements,segments,segmentation_config)

    output_training_data(ppolys, training_data_output_path)

    #结构化输出每个肘板信息
    print("正在输出结构化信息...")
    polys_info = []
    print("正在输出结构化信息...")
    for i, poly in enumerate(ppolys):
        res = outputPolyInfo(poly, new_segments, segmentation_config, point_map, i, star_pos_map, cornor_holes)
        if res is not None:
            polys_info.append(res)

    print("结构化信息输出完毕，保存于:", segmentation_config.poly_info_dir)

    outputRes(new_segments, point_map, polys_info, segmentation_config.res_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)



folder_path = "../jndata2"
output_folder = "./output"
training_data_output_folder = "./DGCNN/data_folder"
process_json_files(folder_path, output_folder, training_data_output_folder)