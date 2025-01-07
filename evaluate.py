import os
import ezdxf
from load import dxf2json
import json

def read_json(json_path, bracket_layer):
    try:  
        with open(json_path, 'r', encoding='utf-8') as file:  
            data_list = json.load(file)
        block_datas=data_list[1]

    except FileNotFoundError:  
        print("The file does not exist.")
    except json.JSONDecodeError:  
        print("Error decoding JSON.")

if __name__ == '__main__':
    test_dxf_path = ""
    gt_dxf_path = ""
    test_bracket_layer = ""
    gt_bracket_layer = ""

    # test_dxf_path = input("请输入待评估图纸路径：")
    # test_bracket_layer = input("请输入待评估图纸中肘板标记所在图层名：")
    # gt_dxf_path = input("请输入人工标记图纸路径：")
    # gt_bracket_layer = input("请输人工标记图纸中肘板标记所在图层名：")

    # 将两个dxf文件转为json
    dxf2json(os.path.dirname(test_dxf_path),os.path.basename(test_dxf_path),os.path.dirname(test_dxf_path))
    dxf2json(os.path.dirname(gt_dxf_path),os.path.basename(gt_dxf_path),os.path.dirname(gt_dxf_path))

    # 获得两个json路径
    test_json_path = os.path.join(os.path.dirname(test_dxf_path), (os.path.basename(test_dxf_path).split('.')[0] + ".json"))
    gt_json_path = os.path.join(os.path.dirname(gt_dxf_path), (os.path.basename(gt_dxf_path).split('.')[0] + ".json"))

    # 解析两个json文件
    
