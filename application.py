import sys
import os
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLineEdit, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

# 假设您已有的所有import保留，例如 element, utils 等
from element import *
from utils import *
from infoextraction import *
from plot_geo import *
from draw_dxf import *
from config import *


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JSON 文件处理工具")
        self.setGeometry(300, 300, 600, 200)

        # 创建界面组件
        self.folder_path_input = QLineEdit(self)
        self.folder_path_input.setPlaceholderText("输入 JSON 文件夹路径")
        self.folder_path_browse_button = QPushButton("浏览")
        self.folder_path_browse_button.clicked.connect(self.browse_folder)

        self.output_folder_input = QLineEdit(self)
        self.output_folder_input.setPlaceholderText("输入输出文件夹路径")
        self.output_folder_browse_button = QPushButton("浏览")
        self.output_folder_browse_button.clicked.connect(self.browse_output_folder)

        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.start_processing)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(QLabel("JSON 文件夹路径:"))
        layout.addWidget(self.folder_path_input)
        layout.addWidget(self.folder_path_browse_button)

        layout.addWidget(QLabel("输出文件夹路径:"))
        layout.addWidget(self.output_folder_input)
        layout.addWidget(self.output_folder_browse_button)

        layout.addWidget(self.start_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择 JSON 文件夹")
        if folder:
            self.folder_path_input.setText(folder)

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if folder:
            self.output_folder_input.setText(folder)

    def start_processing(self):
        folder_path = self.folder_path_input.text().strip()
        output_folder = self.output_folder_input.text().strip()

        if not folder_path or not os.path.isdir(folder_path):
            QMessageBox.critical(self, "错误", "请输入有效的 JSON 文件夹路径")
            return

        if not output_folder:
            QMessageBox.critical(self, "错误", "请输入有效的输出文件夹路径")
            return

        training_data_output_folder = os.path.join(output_folder, "DGCNN/data_folder")
        training_img_output_folder = os.path.join(output_folder, "training_img")

        try:
            process_json_files(folder_path, output_folder, training_data_output_folder, training_img_output_folder)
            QMessageBox.information(self, "完成", "处理完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理文件时出错: {e}")


def process_json_files(folder_path, output_folder, training_data_output_folder, training_img_output_folder):
    # 保留您原有的逻辑
    if not os.path.isdir(folder_path):
        print(f"路径 {folder_path} 不存在或不是一个文件夹。")
        return

    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, name)
            try:
                process_json_data(file_path, output_path, training_data_output_folder, training_img_output_folder, name)
            except json.JSONDecodeError as e:
                print(f"解析JSON文件 {file_path} 时出错: {e}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")


def process_json_data(json_path, output_path, training_data_output_folder, training_img_output_folder, name):
    segmentation_config=SegmentationConfig()
    segmentation_config.json_path = json_path
    segmentation_config.line_image_path = os.path.join(output_path, "line.png")
    segmentation_config.poly_image_dir = os.path.join(output_path, "poly_image")
    segmentation_config.poly_info_dir = os.path.join(output_path, "poly_info")
    segmentation_config.res_image_path = os.path.join(output_path, "res.png")
    segmentation_config.dxf_output_folder = os.path.join(output_path)

    try:
        # os.makedirs(segmentation_config.poly_image_dir, exist_ok=True)
        os.makedirs(segmentation_config.poly_info_dir, exist_ok=True)
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

    if segmentation_config.verbose:
        print("读取json文件")
    #文件中线段元素的读取和根据颜色过滤
    elements,segments=readJson(json_path,segmentation_config)
    #将线进行适当扩张
    
    texts ,dimensions=findAllTextsAndDimensions(elements)
    ori_dimensions=dimensions
    dimensions=processDimensions(dimensions)
    texts=processTexts(texts)
    if segmentation_config.verbose:
        print("json文件读取完毕")
    

    #找出所有包含角隅孔圆弧的基本环
    ppolys, new_segments, point_map,star_pos_map,cornor_holes,text_pos_map=findClosedPolys_via_BFS(elements,texts,dimensions,segments,segmentation_config)
    # output_training_data(ppolys, training_data_output_folder, name)

    # output_training_img(ppolys, new_segments, training_img_output_folder, name)
    #结构化输出每个肘板信息
    polys_info = []
    print("正在输出结构化信息...")
    for i, poly in enumerate(ppolys):
        res = outputPolyInfo(poly, new_segments, segmentation_config, point_map, i, star_pos_map, cornor_holes,texts,dimensions,text_pos_map)
        if res is not None:
            polys_info.append(res)

    print("结构化信息输出完毕，保存于:", segmentation_config.poly_info_dir)

    outputRes(new_segments, point_map, polys_info, segmentation_config.res_image_path,segmentation_config.draw_intersections,segmentation_config.draw_segments,segmentation_config.line_image_drawPolys)

    #将检测到的肘板标注在原本的dxf文件中
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
    draw_rectangle_in_dxf(dxf_path, dxf_output_folder, bboxs)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
