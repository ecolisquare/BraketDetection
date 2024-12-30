import os
import sys
import argparse
from test import process_json_files  # 假设已有该模块

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="肘板检测工具")
    parser.add_argument("-i", "--input", required=True, help="JSON 文件夹路径")
    parser.add_argument("-o", "--output", required=True, help="输出文件夹路径")

    args = parser.parse_args()
    folder_path = args.input
    output_folder = args.output

    # 验证输入路径
    if not os.path.isdir(folder_path):
        print(f"错误: 输入路径 '{folder_path}' 不是有效文件夹。")
        sys.exit(1)

    # 验证输出路径
    if not os.path.isdir(output_folder):
        print(f"错误: 输出路径 '{output_folder}' 不是有效文件夹。")
        sys.exit(1)

    # 执行 JSON 处理逻辑
    try:
        print(f"开始处理 JSON 文件夹: {folder_path}")
        training_data_output_folder = os.path.join(output_folder, "DGCNN/data_folder")
        training_img_output_folder = os.path.join(output_folder, "training_img")

        process_json_files(folder_path, output_folder, training_data_output_folder, training_img_output_folder)
        print("处理完成！")
    except Exception as e:
        print(f"处理过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
