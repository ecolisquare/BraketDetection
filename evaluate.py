import os
import sys
import io
from load import dxf2json
import json
from element import *
from config import *
from utils import readJson, findAllTextsAndDimensions, processDimensions, processTexts, findClosedPolys_via_BFS,process_lwpoline
from shapely.geometry import Polygon, Point

def read_json(json_path, bracket_layer):
    texts=[]
    polys=[]
    try:  
        with open(json_path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_elements=data_list[0]
        for ele in block_elements:
            if ele["layerName"]!=bracket_layer:
                continue
            if ele["type"]=="text":
                e=DText(ele["bound"],ele["insert"], ele["color"],ele["content"].strip(),ele["height"],ele["handle"],meta=None)
                texts.append(e)
            elif ele["type"]=="lwpolyline":
                vs = ele["vertices"]
                new_vs=[]
                poly=[]
                for i,v in enumerate(vs):
                    if len(v)==4:
                        new_vs.append([v[0], v[1]])
                        new_vs.append([v[2], v[3]])        
                for i in range(len(new_vs)-1):
                    s,e=DPoint(new_vs[i][0],new_vs[i][1]),DPoint(new_vs[i+1][0],new_vs[i+1][1])
                    seg=DSegment(s,e,None)
                    if seg.length()>0:
                        poly.append(seg)
                
                if ele["isClosed"]:
                    s,e=DPoint(new_vs[-1][0],new_vs[-1][1]),DPoint(new_vs[0][0],new_vs[0][1])
                    seg=DSegment(s,e,None)
                    if seg.length()>0:
                        poly.append(seg)
                polys.append(poly)
        
    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")
    
    return texts, polys


def calculate_total_covered_area(gt_poly, test_polys):
    """
    计算 gt_poly 在 test_polys 中的总覆盖面积比例
    """
    gt_polygon = Polygon(gt_poly)
    if not gt_polygon.is_valid:
        print("gt invalid")
        return 0.0

    total_inter_area = 0.0
    for test_poly in test_polys:
        test_polygon = Polygon(test_poly)
        if not test_polygon.is_valid:
            print("test invalid")
            continue
        # 累加交集面积
        inter_area = gt_polygon.intersection(test_polygon).area
        total_inter_area += inter_area
    # 计算总覆盖比例
    gt_area = gt_polygon.area
    print(total_inter_area / gt_area)
    if gt_area > 0:

        return total_inter_area / gt_area
    else:
        return 0.0
    

def calculate_search_radius(polygon):
    """
    根据多边形的包围盒对角线计算搜索半径（对角线的一半）
    """
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x1, y1 = min(x_coords), min(y_coords)
    x2, y2 = max(x_coords), max(y_coords)
    diagonal = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return diagonal / 2

def get_text_center(text):
    """
    根据 text.bound 计算文本框的中心位置
    """
    x1, y1 = text.bound["x1"], text.bound["y1"]
    x2, y2 = text.bound["x2"], text.bound["y2"]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return Point(center_x, center_y)

def find_nearest_text(poly, texts, extra_range = 700):
    """
    在包围盒对角线的一半范围内寻找最近的文本框
    """
    polygon = Polygon(poly)
    center = polygon.centroid
    search_radius = calculate_search_radius(poly)
    nearest_text = None
    min_distance = float('inf')
    
    for text in texts:
        text_center = get_text_center(text)
        distance = center.distance(text_center)
        
        # 判断标注是否在框内或自适应搜索范围内
        if distance <= search_radius + extra_range:
            if distance < min_distance:
                nearest_text = text
                min_distance = distance

    return nearest_text

if __name__ == '__main__':
    test_dxf_path = "./output/Large8_braket.dxf"
    gt_dxf_path = "./gt/Large8gt.dxf"
    test_bracket_layer = "Braket"
    gt_bracket_layer = "分段总段划分"

    # test_dxf_path = input("请输入待评估图纸路径：")
    # test_bracket_layer = input("请输入待评估图纸中肘板标记所在图层名：")
    # gt_dxf_path = input("请输入人工标记图纸路径：")
    # gt_bracket_layer = input("请输人工标记图纸中肘板标记所在图层名：")

    print("----------------测试开始---------------")
    # sys.stdout = io.StringIO()

    # 将两个dxf文件转为json
    dxf2json(os.path.dirname(test_dxf_path),os.path.basename(test_dxf_path),os.path.dirname(test_dxf_path))
    dxf2json(os.path.dirname(gt_dxf_path),os.path.basename(gt_dxf_path),os.path.dirname(gt_dxf_path))

    # 获得两个json路径
    test_json_path = os.path.join(os.path.dirname(test_dxf_path), (os.path.basename(test_dxf_path).split('.')[0] + ".json"))
    gt_json_path = os.path.join(os.path.dirname(gt_dxf_path), (os.path.basename(gt_dxf_path).split('.')[0] + ".json"))

    # 解析两个json文件
    test_texts, test_polys_seg = read_json(test_json_path, test_bracket_layer)
    gt_texts, gt_polys_seg = read_json(gt_json_path, gt_bracket_layer)



    test_polys = []
    gt_polys = []

    for poly_seg in test_polys_seg:
        poly = []
        for seg in poly_seg:
            poly.append([seg.start_point.x, seg.start_point.y])
        test_polys.append(poly)
    for poly_seg in gt_polys_seg:
        poly = []
        for seg in poly_seg:
            poly.append([seg.start_point.x, seg.start_point.y])
        gt_polys.append(poly)
    
    # 评估肘板检出率
    coverage_threshold = 0.15
    detect_count = 0
    for gt_poly in gt_polys:
        if calculate_total_covered_area(gt_poly, test_polys) > coverage_threshold:
            detect_count += 1
    
    detection_precison = detect_count / len(gt_polys) if len(gt_polys) > 0 else 1

    # 评估肘板分类正确率
    gt_total_with_labels = 0
    successful_classifications = 0
    for gt_poly in gt_polys:
        nearest_gt_text = find_nearest_text(gt_poly, gt_texts)
        if nearest_gt_text is None:
            continue
        gt_total_with_labels += 1
        gt_polygon = Polygon(gt_poly)
        for test_poly in test_polys:
            test_polygon = Polygon(test_poly)
            if test_polygon.intersects(gt_polygon):
                nearest_test_text = find_nearest_text(test_poly, test_texts)
                if nearest_test_text and nearest_gt_text.content in nearest_test_text.content:
                    # print(f"{nearest_gt_text.content}, {nearest_test_text.content}")
                    successful_classifications += 1
                    break
    classification_precision = successful_classifications / gt_total_with_labels if gt_total_with_labels > 0 else 1

    # 输出评估结果
    # sys.stdout = sys.__stdout__
    print("----------------测试完毕---------------")
    print(f"肘板检出率: {detection_precison:.2f}")
    print(f"肘板分类正确率: {classification_precision:.2f}")
    print(gt_total_with_labels , len(gt_polys) , len(gt_texts),len(test_polys),len(test_texts))
    print("-------------测试结果输出完毕----------")
    # print([ len(s) for s in test_polys_seg ])
    
