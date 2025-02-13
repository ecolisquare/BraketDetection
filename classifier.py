import json
import random
from element import *
def is_vertical_(point1,point2,segment,epsilon=0.05):
    v1=DPoint(point1.x-point2.x,point1.y-point2.y)
    v2=DPoint(segment.start_point.x-segment.end_point.x,segment.start_point.y-segment.end_point.y)
    cross_product=(v1.x*v2.x+v1.y+v2.y)/(DSegment(point1,point2).length()*segment.length())
    if  abs(cross_product) <epsilon:
        return True
    return False 
def is_tangent(line,arc):
    s1,s2=DSegment(arc.ref.center,arc.ref.start_point),DSegment(arc.ref.center,arc.ref.end_point)
    v1,v2=line.start_point,line.end_point
    if v1==arc.ref.start_point or v2==arc.ref.start_point:
        return is_vertical_(v1,v2,s1)
    if v1==arc.ref.end_point or v2==arc.ref.end_point:
        return is_vertical_(v1,v2,s2)
    return True


def find_anno_info(matched_type,all_anno,poly_free_edges):
    radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno=all_anno
    if "DAB(VU-KS)" in matched_type:
        if len(angle_anno)==0:
            return "no_angle"
        else:
            return "angl"
    elif "DAC(VU-R)" in matched_type:
        if len(cornor_anno)==0:
            return "no_anno"
        else:
            return "short_anno"
    elif "DPK(R)" in matched_type:
        if len(parallel_anno)!=0:
            return "short_anno_para"
        elif len(half_anno)!=0:
            return "short_anno"
        else:
            return "long_anno"
    elif "DAB(R-KS)" in matched_type:
        if len(cornor_anno)==0:
            return "no_anno"
        else:
            return "short_anno"
    elif "DPKN(KS-KS)" in matched_type:
        if len(half_anno)!=0:
            return "short_anno"
        else:
            return "long_anno"
    elif "DAD(R)" in matched_type:
        if len(half_anno)!=0:
            return "dist_intersec"
        else:
            return "no_dist"
    elif "DPKN(KS-R)" in matched_type:
        if len(half_anno)!=0:
            return "short_anno"
        else:
            return "long_anno"
    elif "DAB-3(R-KS)" in matched_type:
        free_edges=poly_free_edges[0]
        line=free_edges[2]
        arc1,arc2=free_edges[1],free_edges[3]
        if is_tangent(line,arc1) and is_tangent(line,arc2):
            if len(angle_anno)!=0:
                return "angl_non_free"
            else:
            
                return "dist_intersec"
        else:
            return "no_tangent"
    elif "KL(R)" in matched_type:
        if len(vertical_anno)==2:
            return "tt"
        elif len(vertical_anno)==1:
            return "tf"
        else:
            return "ff"
    elif "KL(KS)" in matched_type:
        if len(vertical_anno)==2:
            return "tt"
        elif len(vertical_anno)==1:
            return "tf"
        else:
            return "ff"
    elif "BR-1(R)" in matched_type:
        if len(angle_anno)!=0:
            return "angl_non_free"
        elif len(half_anno)!=0:
            return "dist_intersec"
        else:
            return "no_anno"
    elif "DPK(R-KS)" in matched_type:
        if len(half_anno)!=0:
            return "short_anno"
        else:
            return "long_anno"
    elif "BR-1(KS)" in matched_type:
        if len(angle_anno)!=0:
            return "angl_non_free"
        else:
            return "dist_intersec"
    elif "LBMA-1(KS)" in matched_type:
        if len(angle_anno)!=0:
            return "angl"
        else:
            return "dist"
    elif "DPV-4(R-KS)" in matched_type:
        if len(parallel_anno)!=0:
            return "D_anno"
        else:
            return "no_anno"
    elif "DAC(KS-KS)" in matched_type:
        if len(d_anno)!=0:
            return "D"
        else:
            return "notD"
    elif "DAE(R-R)" in matched_type:
        if len(d_anno)!=0:
            return "D"
        else:
            return "notD"
    elif "DPK-1（R）" in matched_type:
        if len(non_parallel_anno)!=0:
            return "dist_adja"
        elif len(toe_angle_anno)!=0:
            return "angl_toe"
        elif len(half_anno)!=0:
            return "dist_intersec"
        else:
            return "angl_non_free"


    return None
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
        # if row["cornerhole_nums"] != conerhole_num:
        #     continue
        
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
    if isinstance(edge[0], list):
        new_edge = edge[0]
    else:
        new_edge = edge
    return min(tuple(new_edge), tuple(reversed(new_edge)))

def conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence):
    matched_type = None
    non_conerhole_edges = []
    reversed_non_conerhole_edges = []
    conerhole_count = {}
    unrestricted_cornerhole_count = {}
    # unrestricted_cornerhole_type = [["line"], ["arc"]]
    unrestricted_cornerhole_type = [["line"]]
    unrestricted_cornerhole_num = 0
    # 去掉非自由边轮廓中的角隅孔，只保留固定边
    for i in range(len(edges_sequence)):
        if(edges_sequence[i][0] != "cornerhole"):
            non_conerhole_edges.append(edges_sequence[i][1])
        # 记录角隅孔字典
        else:
            key = generate_key(edges_sequence[i][1])
            conerhole_count[key] = conerhole_count.get(key, 0) + 1
            if edges_sequence[i][1] not in unrestricted_cornerhole_type:
                unrestricted_cornerhole_count[key] = unrestricted_cornerhole_count.get(key, 0) + 1
            else:
                unrestricted_cornerhole_num += 1
        if(reversed_edges_sequence[i][0] != "conerhole"):
            reversed_non_conerhole_edges.append(reversed_edges_sequence[i][1])

    # 考虑趾端和角隅孔各类型数量的严格匹配
    for key_name, row in classification_table.items():
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

        matched_type = key_name if matched_type is None else f"{matched_type}, {key_name}"
    
    # 参考趾端和角隅孔各类型数量的不严格匹配
    if matched_type is None:
        for key_name, row in classification_table.items():
            # step1: 自由边轮廓严格匹配
            if row["free_edges"] != free_edges_sequence and row["free_edges"] != reversed_free_edges_sequence:
                continue

            # step2: 非自由边轮廓去除角隅孔后严格匹配
            temp_sequence = []
            temp_conerhole_count = {}
            temp_unrestricted_cornerhole_num = 0
            for i, non_free in enumerate(row["non_free_edges"]):
                # Check if the type and edges of the current non-free match in sequence
                if non_free["type"] == "constraint":
                    temp_sequence.append(non_free["edges"])
                elif non_free["type"] == "cornerhole":
                    if not (not isinstance(non_free["edges"][0], list) and (non_free["edges"] in unrestricted_cornerhole_type)):
                        key = generate_key(non_free["edges"])
                        temp_conerhole_count[key] = temp_conerhole_count.get(key, 0) + 1
                    else:
                        temp_unrestricted_cornerhole_num += 1

            if temp_sequence != non_conerhole_edges and temp_sequence != reversed_non_conerhole_edges:
                continue

            # step3: 角隅孔计数匹配
            if temp_conerhole_count != unrestricted_cornerhole_count:
                continue

            matched_type = key_name if matched_type is None else f"{matched_type}, {key_name}"

    # 仅参考角隅孔各类型数量的严格匹配

    # 去除趾板类型
    free_edges_sequence_without_toe = []
    reversed_free_edges_sequence_without_toe = []
    for i in range(len(free_edges_sequence)):
        if free_edges_sequence[i] == "toe":
            edge = "line"
        else:
            edge = free_edges_sequence[i]
        free_edges_sequence_without_toe.append(edge)
        if reversed_free_edges_sequence[i] == "toe":
            edge = "line"
        else:
            edge = reversed_free_edges_sequence[i]
        reversed_free_edges_sequence_without_toe.append(edge)

    if matched_type is None:
        for key_name, row in classification_table.items():
            # step1: 自由边轮廓严格匹配
            if row["free_edges"] != free_edges_sequence_without_toe and row["free_edges"] != reversed_free_edges_sequence_without_toe:
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

            matched_type = key_name if matched_type is None else f"{matched_type}, {key_name}"

    # 仅参考角隅孔各类型数量的不严格匹配
    if matched_type is None:
        for key_name, row in classification_table.items():
            # step1: 自由边轮廓严格匹配
            if row["free_edges"] != free_edges_sequence_without_toe and row["free_edges"] != reversed_free_edges_sequence_without_toe:
                continue

            # step2: 非自由边轮廓去除角隅孔后严格匹配
            temp_sequence = []
            temp_conerhole_count = {}
            temp_unrestricted_cornerhole_num = 0
            for i, non_free in enumerate(row["non_free_edges"]):
                # Check if the type and edges of the current non-free match in sequence
                if non_free["type"] == "constraint":
                    temp_sequence.append(non_free["edges"])
                elif non_free["type"] == "cornerhole":
                    if not (not isinstance(non_free["edges"][0], list) and (non_free["edges"] in unrestricted_cornerhole_type)):
                        key = generate_key(non_free["edges"])
                        temp_conerhole_count[key] = temp_conerhole_count.get(key, 0) + 1
                    else:
                        temp_unrestricted_cornerhole_num += 1

            if temp_sequence != non_conerhole_edges and temp_sequence != reversed_non_conerhole_edges:
                continue

            # step3: 角隅孔计数匹配
            if temp_conerhole_count != unrestricted_cornerhole_count:
                continue

            matched_type = key_name if matched_type is None else f"{matched_type}, {key_name}"

    return matched_type if matched_type is not None else "Unclassified"

def poly_classifier(all_anno,poly_refs, texts,dimensions,conerhole_num, poly_free_edges, edges, classification_file_path, info_json_path, keyname, is_output_json = False):
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
    matched_type= conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence)
    if len(matched_type.split(","))<=1:
        return matched_type
    #TODO
    #for each mixed type, use the first type name as key(cluster_name),find the annoation
    anno=find_anno_info(matched_type,all_anno,poly_free_edges)
    random.seed(13)
    if anno is None:
        #default type of the mixed type
        return matched_type
    else:
        #classify inner mixed type by annotation

        # DAB(VU-KS), DAB-1(VU-KS)
        cluster_name="DAB(VU-KS)"
        if cluster_name in matched_type:
            if anno == "no_angle":
                matched_type = "DAB(VU-KS)"
            elif anno == "angl":
                matched_type = "DAB-1(VU-KS)"
            else:
                matched_type = "DAB(VU-KS)"

        # DAC(VU-R), DAC(VUF-R)
        cluster_name="DAC(VU-R)"
        if cluster_name in matched_type:
            if anno == "no_anno":
                matched_type = "DAC(VU-R)"
            elif anno == "short_anno":
                matched_type = "DAC(VUF-R)"
            else:
                matched_type = "DAC(VU-R)"
        
        # DPK(R), LDPK(R-R), LDPK-1(R-R)
        cluster_name="DPK(R)"
        if cluster_name in matched_type:
            if anno == "long_anno":
                matched_type = "DPK(R)"
            elif anno == "short_anno":
                matched_type = "LDPK(R-R)"
            elif anno == "short_anno_para":
                matched_type = "LDPK-1(R-R)"
            else:
                matched_type = "DPK(R)"
        
        # DAB(R-KS), DAB-1(R-KS)
        cluster_name="DAB(R-KS)"
        if cluster_name in matched_type:
            if anno == "no_angle":
                matched_type = "DAB(R-KS)"
            elif anno == "angl":
                matched_type = "DAB-1(R-KS)"
            else:
                matched_type = "DAB(R-KS)"

        # DPKN(KS-KS), LDPKN(KS-KS), DAB-3(KS-KS)
        cluster_name="DPKN(KS-KS)"
        if cluster_name in matched_type:
            if anno == "long_anno":
                matched_type = "DPKN(KS-KS)"
            elif anno == "short_anno":
                matched_type = "LDPKN(KS-KS)"
            else:
                matched_type = random.choice(["DPKN(KS-KS)", "DAB-3(KS-KS)"])

        # DAD(R), DCD(R) 
        cluster_name="DAD(R)"
        if cluster_name in matched_type:
            if anno == "dist_intersec":
                matched_type = "DAD(R)"
            elif anno == "no_dist":
                matched_type = "DCD(R)"
            else:
                matched_type = "DCD(R)"


        # DPKN(R-KS), DPKN(KS-R), LDPKN(KS-R)
        cluster_name="DPKN(R-KS)"
        if cluster_name in matched_type:
            if anno == "long_anno":
                matched_type = random.choice(["DPKN(R-KS)", "DPKN(KS-R)"])
            elif anno == "short_anno":
                matched_type = "LDPKN(KS-R)"
            else:
                matched_type = random.choice(["DPKN(R-KS)", "DPKN(KS-R)"])

        # DAB-3(R-KS), LDBA(R-KS), BCB-1(R-KS)
        cluster_name="DAB-3(R-KS)"
        if cluster_name in matched_type:
            if anno == "angl_non_free":
                matched_type = "DAB-3(R-KS)"
            elif anno == "dist_intersec":
                matched_type = "LDBA(R-KS)"
            elif anno == "no_tangent":
                matched_type = "BCB-1(R-KS)"
            else:
                matched_type = "LDBA(R-KS)"

        # KL(R), KL-1(R), KL-2(R)
        cluster_name="KL(R)"
        if cluster_name in matched_type:
            if anno == "ff":
                matched_type = "KL(R)"
            elif anno == "tt":
                matched_type = "KL-1(R)"
            elif anno == "tf":
                matched_type = "KL-2(R)"
            else:
                matched_type = "KL(R)"
        
        # KL(KS), KL-1(KS), KL-2(KS)
        cluster_name="KL(KS)"
        if cluster_name in matched_type:
            if anno == "ff":
                matched_type = "KL(KS)"
            elif anno == "tt":
                matched_type = "KL-1(KS)"
            elif anno == "tf":
                matched_type = "KL-2(KS)"
            else:
                matched_type = "KL(KS)"
        
        # DPK-1(R), DPK-2(R-R), DPK-5(R-R), LDPK-3(R-R)
        cluster_name="DPK-1(R)"
        if cluster_name in matched_type:
            if anno == "angl_non_free":
                matched_type = "DPK-1(R)"
            elif anno == "dist_adja":
                matched_type = "DPK-2(R-R)"
            elif anno == "angl_toe":
                matched_type = "DPK-5(R-R)"
            elif anno == "dist_intersec":
                matched_type = "LDPK-3(R-R)"
            else:
                matched_type = "DPK-1(R)"
        
        # BR(R), BR-1(R), DAA(R)
        cluster_name="BR(R)"
        if cluster_name in matched_type:
            if anno == "no_anno":
                matched_type = "BR(R)"
            elif anno == "angl_non_free":
                matched_type = "BR-1(R)"
            elif anno == "dist_intersec":
                matched_type = "DAA(R)"
            else:
                matched_type = "BR(R)"
        
        # DPK(R-KS), LDPK-1(KS-R)
        cluster_name="DPK(R-KS)"
        if cluster_name in matched_type:
            if anno == "long_anno":
                matched_type = "DPK(R-KS)"
            elif anno == "short_anno":
                matched_type = "LDPK-1(KS-R)"
            else:
                matched_type = "DPK(R-KS)"
        
        # BR-1(KS), DAA(KS)
        cluster_name="BR-1(KS)"
        if cluster_name in matched_type:
            if anno == "angl_non_free":
                matched_type = "BR-1(KS)"
            elif anno == "dist_intersec":
                matched_type = "DAA(KS)"
            else:
                matched_type = "DAA(KS)"
        
        # DPV-4(R-KS), DPV-6(R-KS)
        cluster_name="DPV-4(R-KS)"
        if cluster_name in matched_type:
            if anno == "D_anno":
                matched_type = "DPV-4(R-KS)"
            elif anno == "no_anno":
                matched_type = "DPV-6(R-KS)"
            else:
                matched_type = "DPV-6(R-KS)"
        
        # LBMA-1(KS), BMA-1(KS)
        cluster_name="LBMA-1(KS)"
        if cluster_name in matched_type:
            if anno == "dist":
                matched_type = "BMA-1(KS)"
            elif anno == "angl":
                matched_type = "LBMA-1(KS)"
            else:
                matched_type = "BMA-1(KS)"
        
        return matched_type