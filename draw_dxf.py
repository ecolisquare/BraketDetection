import os 
import numpy as np 
import ezdxf
import json


# def draw_rectangle_in_dxf(file_path, folder, bbox_list, classi_res):
        
        
#         folder = os.path.normpath(os.path.abspath(folder))
#         os.makedirs(folder, exist_ok=True)


#         doc = ezdxf.readfile(file_path)
#         msp = doc.modelspace()
        
#         if "Braket" not in doc.layers:
#             doc.layers.add("Braket", color=30) 


#         for idx, (bbox, classification) in enumerate(zip(bbox_list, classi_res)):

#             x1 = bbox[0][0] - 20
#             y1 = bbox[0][1] - 20
#             x2 = bbox[1][0] + 20
#             y2 = bbox[1][1] + 20

#             top_left = (x1, y1)
#             top_right = (x2, y1)
#             bottom_right = (x2, y2)
#             bottom_left = (x1, y2)

#             # 通过连接四条线来绘制矩形
#             msp.add_line(top_left, top_right, dxfattribs={"layer": "Braket"})  # 红色线条
#             msp.add_line(top_right, bottom_right, dxfattribs={"layer": "Braket"})
#             msp.add_line(bottom_right, bottom_left, dxfattribs={"layer": "Braket"})
#             msp.add_line(bottom_left, top_left, dxfattribs={"layer": "Braket"})

#             text = msp.add_text(classification, dxfattribs={"layer": "Braket", "height":50})
#             text.dxf.insert = ((x1+x2)/2, y2)
#             # msp.add_text("NO.{}".format(idx), dxfattribs={"layer": "Split", "height": 100}).set_dxf_attrib("insert",(x1, y1-20))

#             # 保存修改后的 DXF 文件

#         file_name = os.path.basename(file_path)[:-4]

#         doc.saveas(os.path.join(folder, "{}_braket.dxf".format(file_name)))

#         print(f"braket detect result drawn in {file_name}_braket")
#         print("Done....")




def draw_rectangle_in_dxf(file_path, folder, bbox_list, classi_res,idxs, free_edge_handles,non_free_edge_handles,all_handles,not_all_handles,removed_handles):
    folder = os.path.normpath(os.path.abspath(folder))
    os.makedirs(folder, exist_ok=True)

    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    if "Braket" not in doc.layers:
        doc.layers.add("Braket", color=30)

    for idx, (bbox, classification) in enumerate(zip(bbox_list, classi_res)):
        if classification=='Unclassified':
            continue
        if ',' in classification:
            continue
        x1 = bbox[0][0] - 20
        y1 = bbox[0][1] - 20
        x2 = bbox[1][0] + 20
        y2 = bbox[1][1] + 20

        # 使用多段线绘制矩形
        rectangle_points = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2),  # Bottom-left
        ]
        msp.add_lwpolyline(rectangle_points, close=True, dxfattribs={"layer": "Braket"})

        # 添加文本
        if classification != "Unclassified":
            text = msp.add_text(classification, dxfattribs={"layer": "Braket", "height": 50})
            text.dxf.insert = ((x1 + x2) / 2, y2)
        text2 = msp.add_text(f"poly_id {idxs[idx]}", dxfattribs={"layer": "Braket", "height": 50})
        text2.dxf.insert = ((x1 + x2) / 2, y1)


    all_free_layer="精确匹配_自由边"
    all_non_free_layer="精确匹配_非自由边"
    not_all_free_layer="模糊匹配_自由边"
    not_all_non_free_layer="模糊匹配_非自由边"
    if all_free_layer not in doc.layers:
        doc.layers.add(all_free_layer, color=7)
    if all_non_free_layer not in doc.layers:
        doc.layers.add(all_non_free_layer, color=7)
    if not_all_free_layer not in doc.layers:
        doc.layers.add(not_all_free_layer, color=7)
    if not_all_non_free_layer not in doc.layers:
        doc.layers.add(not_all_non_free_layer, color=7)
    for e in msp:
        if e.dxf.handle in free_edge_handles and e.dxf.handle in all_handles:
            e.dxf.layer = all_free_layer
        if e.dxf.handle in free_edge_handles and e.dxf.handle in not_all_handles:
            e.dxf.layer = not_all_free_layer
        if e.dxf.handle in non_free_edge_handles and e.dxf.handle in all_handles:
            e.dxf.layer = all_non_free_layer
        if e.dxf.handle in non_free_edge_handles and e.dxf.handle in not_all_handles:
            e.dxf.layer = not_all_non_free_layer
    ref_line_layer="引线"
    if ref_line_layer not in doc.layers:
        doc.layers.add(ref_line_layer,color=7)
    
    for e in msp:
        if e.dxf.handle in removed_handles:
            e.dxf.layer=ref_line_layer


    file_name = os.path.basename(file_path)[:-4]
    doc.saveas(os.path.join(folder, f"{file_name}_braket.dxf"))

    print(f"Braket detect result drawn in {file_name}_braket")
    print("Done....")
