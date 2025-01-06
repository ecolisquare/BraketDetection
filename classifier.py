import json
from element import *

def load_classification_table(file_path):
    """
    Load the classification table from a JSON file.
    """
    with open(file_path, 'r') as f:
        classification_table = json.load(f)  # Load the JSON file
    return classification_table

# 严格的匹配算法（角隅孔个数，自由边轮廓，固定边轮廓都必须完全匹配）
def strict_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence):
    matched_type = None
    for key, row in classification_table.items():
        # Step 1: Match cornerhole_nums
        if row["cornerhole_nums"] != conerhole_num:
            continue
        
        # Step 2: Match free_edges (both normal and reversed)
        if row["free_edges"] != free_edges_sequence and row["free_edges"] != reversed_free_edges_sequence:
            continue

        # Step 3: Match non-free edges sequence in order
        if len(edges_sequence) != len(row["non_free_edges"]):
            continue

        non_free_edges_match = True
        for i, non_free in enumerate(row["non_free_edges"]):
            # Check if the type and edges of the current non-free match in sequence
            if [non_free["type"], non_free["edges"]] != edges_sequence[i]:
                    non_free_edges_match = False
                    break

        # If matching non-free edges in order is successful
        if non_free_edges_match:
            matched_type = key  # Use the key like 'type_1' as matched type
            break

        # Step 4: If non-free edges didn't match in order, try reversing the edges sequence for matching
        non_free_edges_match = True
        for i, non_free in enumerate(row["non_free_edges"]):
            # Check if the type and edges of the current non-free match in reversed sequence
            if [non_free["type"], non_free["edges"]] != reversed_edges_sequence[i]:
                    non_free_edges_match = False
                    break

        # If matching non-free edges in reverse is successful
        if non_free_edges_match:
            matched_type = key  # Use the key like 'type_1' as matched type
            break
    
    return matched_type if matched_type is not None else "Unclassified"

# 宽松的匹配算法（不考虑角隅孔个数，自由边和固定边与模板的匹配设置一定容错）
def unrestricted_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence):
    matched_type = None
    free_edges_tolerance = 1
    non_free_edges_tolerance = 2
    for key, row in classification_table.items():
        # 计数自由边不匹配数量
        free_edges_set = set(row["free_edges"])
        input_free_edges_set = set(free_edges_sequence)
        unmatched_free_edges = input_free_edges_set - free_edges_set
        unmatched_free_count = len(unmatched_free_edges)

        # 如果不匹配的自由边超出容忍阈值，则跳过
        if unmatched_free_count > free_edges_tolerance:
            continue

        # 计数非自由边不匹配数量
        non_free_edges = row["non_free_edges"]
        if len(edges_sequence) != len(non_free_edges):
            continue

        unmatched_non_free_count = 0
        for i, non_free in enumerate(non_free_edges):
            input_edge_type, input_edge_shapes = edges_sequence[i]

            # 判断类型是否匹配
            type_match = (non_free["type"] == input_edge_type)

            # 判断形状是否有交集
            shape_match = bool(set(non_free["edges"]).intersection(input_edge_shapes))

            if not (type_match and shape_match):
                unmatched_non_free_count += 1

            # 如果不匹配数量超过容忍阈值，直接跳出
            if unmatched_non_free_count > non_free_edges_tolerance:
                break

        # 如果非自由边匹配不符合要求，跳过当前模板
        if unmatched_non_free_count > non_free_edges_tolerance:
            continue

        # 如果当前模板通过所有判定，则匹配
        matched_type = key
        break

    return matched_type if matched_type is not None else "Unclassified"

def generate_key(edge):
    # Generate a key that considers both original and reversed order
    return min(tuple(edge), tuple(reversed(edge)))

def conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence):
    matched_type = None
    non_conerhole_edges = []
    reversed_non_conerhole_edges = []
    conerhole_count = {}
    # 去掉非自由边轮廓中的角隅孔，只保留固定边
    for i in range(len(edges_sequence)):
        if(edges_sequence[i][0] != "cornerhole"):
            non_conerhole_edges.append(edges_sequence[i][1])
        # 记录角隅孔字典
        else:
            key = generate_key(edges_sequence[i][1])
            conerhole_count[key] = conerhole_count.get(key, 0) + 1
        if(reversed_edges_sequence[i][0] != "conerhole"):
            reversed_non_conerhole_edges.append(reversed_edges_sequence[i][1])
    for key, row in classification_table.items():
        # step1: 自由边轮廓严格匹配
        if row["free_edges"] != free_edges_sequence and row["free_edges"] != reversed_free_edges_sequence:
            continue

        # step2: 非自由边轮廓去除角隅孔后严格匹配
        temp_sequence = []
        temp_conerhole_count = {}
        for i, non_free in enumerate(row["non_free_edges"]):
            # Check if the type and edges of the current non-free match in sequence
            if non_free["type"] == "constraint":
                temp_sequence.append(non_free["edges"])
            elif non_free["type"] == "cornerhole":
                key = generate_key(non_free["edges"])
                temp_conerhole_count[key] = temp_conerhole_count.get(key, 0) + 1

        if temp_sequence != non_conerhole_edges and temp_sequence != reversed_non_conerhole_edges:
            continue

        # step3: 角隅孔计数匹配
        if temp_conerhole_count != conerhole_count:
            continue

        matched_type = key if matched_type is None else f"{matched_type}, {key}"

    return matched_type if matched_type is not None else "Unclassified"

def poly_classifier(poly_refs, conerhole_num, poly_free_edges, edges, classification_file_path, info_json_path, keyname, is_output_json = False):
    classification_table = load_classification_table(classification_file_path)

    # Step 1: 获取角隅孔数

    # Step 2: 获取自由边的轮廓
    free_edges_sequence = []
    for i, seg in enumerate(poly_free_edges[0]):
        if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
            if (i == 0 or i == len(poly_free_edges[0]) - 1) and seg.length() < 25:
                free_edges_sequence.append("toe")
            else:
                free_edges_sequence.append("line")
        elif isinstance(seg.ref, DArc):
            free_edges_sequence.append("arc")
    reversed_free_edges_sequence = free_edges_sequence[::-1]  # Reverse free edges

    # Step 3: 以自由边为分界，获取固定边和角隅孔组成的顺序轮廓
    cycle_edges = edges + edges
    al_edges = []
    start = False
    for edge in cycle_edges:
        # 如果遇到第一个非 Cornerhole 和非 Constraint 边，开始收集
        if not start and edge[0].isCornerhole == False and edge[0].isConstraint == False:
            start = True
        
        # 如果已经开始收集，且当前边是 Cornerhole 或 Constraint，加入 al_edges
        elif start and (edge[0].isCornerhole or edge[0].isConstraint):
            al_edges.append(edge)
        
        # 如果已经开始收集，且当前边不再是 Cornerhole 或 Constraint，停止收集
        elif start and edge[0].isCornerhole == False and edge[0].isConstraint == False:
            start = False

    # Step 4: 构建 edges_sequence 和 reversed_edges_sequence
    edges_sequence = []
    reversed_edges_sequence = []
    for edge in al_edges:
        if edge[0].isCornerhole:
            type = "cornerhole"
        elif edge[0].isConstraint:
            type = "constraint"
        else:
            type = None
        
        if type is not None:
            seq = []
            for seg in edge:
                if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
                    seq.append("line")
                elif isinstance(seg.ref, DArc):
                    seq.append("arc")
            edges_sequence.append([type, seq])
            reversed_edges_sequence.insert(0, [type, list(reversed(seq))])

    # 对肘板轮廓进行输出
    if is_output_json:
        geometry_info = {
            keyname: {
                "free_edges_sequence": free_edges_sequence,
                "non_free_edges_sequence": edges_sequence
            }
        }
        try:
            # 检查目标文件是否存在，如果存在则读取并更新
            try:
                with open(info_json_path, "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = {}

            # 更新数据
            existing_data.update(geometry_info)

            # 写入文件
            with open(info_json_path, "w") as file:
                json.dump(existing_data, file, indent=4)

        except Exception as e:
            print(f"Error writing to JSON file: {e}")


    # Step 5: 遍历每个肘板类型，进行匹配
    return conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence)
