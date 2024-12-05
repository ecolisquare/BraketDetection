import json
from element import *

def load_classification_table(file_path):
    """
    Load the classification table from a JSON file.
    """
    with open(file_path, 'r') as f:
        classification_table = json.load(f)  # Load the JSON file
    return classification_table

def poly_classifier(poly_refs, conerhole_num, poly_free_edges, edges, classification_file_path):
    classification_table = load_classification_table(classification_file_path)

    # Step 1: 获取角隅孔数

    # Step 2: 获取自由边的轮廓
    free_edges_sequence = []
    for seg in poly_free_edges[0]:
        if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
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

    # Step 5: 遍历每个肘板类型，进行匹配
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
    
    if matched_type is not None:
        result = matched_type
    else:
        result = "Unclassified"

    return result
