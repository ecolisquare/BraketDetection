import json
import random
from element import *
import copy
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

def is_free_edges_equal(free_edges_sequence, temp):
    if free_edges_sequence == temp:
        return True
    free_edges_sequence_copy = copy.deepcopy(free_edges_sequence)
    free_edges_sequence_copy2 = copy.deepcopy(free_edges_sequence)
    free_edges_sequence_copy2.append("KS_corner")
    free_edges_sequence_copy.insert(0, "KS_corner")
    if free_edges_sequence_copy2 == temp:
        return True
    if free_edges_sequence_copy == temp:
        return True
    free_edges_sequence_copy2.insert(0, "KS_corner")
    if free_edges_sequence_copy2 == temp:
        return True
    return False

def find_anno_info(matched_type,all_anno,poly_free_edges):
    radius_anno,whole_anno,half_anno,cornor_anno,parallel_anno,non_parallel_anno,vertical_anno,d_anno,angle_anno,toe_angle_anno=all_anno
    anno={}
    if ",DAC(VU-R)," in matched_type:
        if len(cornor_anno)==0:
            anno[",DAC(VU-R),"]="no_anno"
        else:
            anno[",DAC(VU-R),"]="short_anno"
    else:
        anno[",DAC(VU-R),"]="none"
    # elif "DAB(VU-KS)" in matched_type:
    #     if len(angle_anno)==0:
    #         return "no_angle"
    #     else:
    #         return "angl"
    # print(matched_type)
    if ",DPK(R)," in matched_type:
        if len(parallel_anno)!=0:
            anno[",DPK(R),"]="short_anno_para"
        elif len(half_anno)!=0:
            anno[",DPK(R),"]= "short_anno"
        else:
            anno[",DPK(R),"]= "long_anno"
    else:
        anno[",DPK(R),"]="none"
    # elif "DAB(R-KS)" in matched_type:
    #     if len(cornor_anno)==0:
    #         return "no_anno"
    #     else:
    #         return "short_anno"
    if ",DPKN(KS-KS)," in matched_type:
        if len(half_anno)!=0:
            anno[",DPKN(KS-KS),"]="short_anno"
        else:
            anno[",DPKN(KS-KS),"]="long_anno"
    else:
        anno[",DPKN(KS-KS),"]="none"
    if ",DAD(R)," in matched_type:
        if len(half_anno)!=0:
            anno[",DAD(R),"]="dist_intersec"
        else:
            anno[",DAD(R),"]="no_dist"
    else:
        anno[",DAD(R),"]="none"
    if ",DPKN(KS-R)," in matched_type:
        if len(half_anno)!=0:
            anno[",DPKN(KS-R),"]="short_anno"
        else:
            anno[",DPKN(KS-R),"]="long_anno"
    else:
        anno[",DPKN(KS-R),"]="none"
    if ",DAB-3(R-KS)," in matched_type:
        free_edges=poly_free_edges[0]
        line=free_edges[1] if isinstance(free_edges[1],DLine) else free_edges[2]
        arc=free_edges[1] if isinstance(free_edges[1],DArc) else free_edges[2]
        if isinstance(line,DLine) and isinstance(arc,DArc) and is_tangent(line,arc):
            if len(angle_anno)!=0:
                anno[",DAB-3(R-KS),"]="angl_non_free"
            else:
            
                anno[",DAB-3(R-KS),"]="dist_intersec"
        else:
            anno[",DAB-3(R-KS),"]="no_tangent"
    else:
        anno[",DAB-3(R-KS),"]="none"
    if ",KL(R)," in matched_type:
        if len(vertical_anno)==2:
            anno[",KL(R),"]="tt"
        elif len(vertical_anno)==1:
            anno[",KL(R),"]= "tf"
        else:
            anno[",KL(R),"]= "ff"
    else:
        anno[",KL(R),"]="none"
    if ",KL(KS)," in matched_type:
        if len(vertical_anno)==2:
            anno[",KL(KS),"]= "tt"
        elif len(vertical_anno)==1:
            anno[",KL(KS),"]= "tf"
        else:
            anno[",KL(KS),"]= "ff"
    else:
        anno[",KL(KS),"]="none"
    if ",BR(R)," in matched_type:
        if len(angle_anno)!=0:
            anno[",BR(R),"]="angl_non_free"
        elif len(half_anno)!=0:
            anno[",BR(R),"]= "dist_intersec"
        else:
            anno[",BR(R),"]= "no_anno"
    else:
        anno[",BR(R),"]="none"
    if ",DPK(R-KS)," in matched_type:
        if len(half_anno)!=0:
            anno[",DPK(R-KS),"]="short_anno"
        else:
            anno[",DPK(R-KS),"]= "long_anno"
    else:
        anno[",DPK(R-KS),"]="none"
    if ",BR-1(KS)," in matched_type:
        if len(angle_anno)!=0:
            anno[",BR-1(KS),"]="angl_non_free"
        else:
            anno[",BR-1(KS),"]= "dist_intersec"
    else:
        anno[",BR-1(KS),"]="none"
    if ",LBMA-1(KS)," in matched_type:
        if len(angle_anno)!=0:
            anno[",LBMA-1(KS),"]="angl"
        else:
            anno[",LBMA-1(KS),"]= "dist"
    else:
        anno[",LBMA-1(KS),"]="none"
    if ",DPV-4(R-KS)," in matched_type:
        if len(parallel_anno)!=0:
            anno[",DPV-4(R-KS),"]="D_anno"
        else:
            anno[",DPV-4(R-KS),"]= "no_anno"
    else:
        anno[",DPV-4(R-KS),"]="none"
    if ",DAC(KS-KS)," in matched_type:
        if len(d_anno)!=0:
            anno[",DAC(KS-KS),"]="D"
        else:
            anno[",DAC(KS-KS),"]="notD"
    else:
        anno[",DAC(KS-KS),"]="none"
    if ",DAC(R-R)," in matched_type:
        if len(d_anno)!=0:
            anno[",DAC(R-R),"]="D_anno"
        else:
            anno[",DAC(R-R),"]= "no_anno"
    else:
        anno[",DAC(R-R),"]="none"
    if ",DPK-1(R)," in matched_type:
        if len(non_parallel_anno)!=0:
            anno[",DPK-1(R),"]="dist_adja"
        elif len(toe_angle_anno)!=0:
            anno[",DPK-1(R),"]= "angl_toe"
        elif len(half_anno)!=0:
            anno[",DPK-1(R),"]= "dist_intersec"
        else:
            anno[",DPK-1(R),"]= "angl_non_free"
    else:
        anno[",DPK-1(R),"]="none"
    if ",DPKN(VU-R)," in matched_type:
       
        if len(half_anno)!=0:
            anno[",DPKN(VU-R),"]="short_anno"
        else:
            anno[",DPKN(VU-R),"]= "long_anno"
    else:
        anno[",DPKN(VU-R),"]="none"
    if ",DPKN(KS-KS)," in matched_type:
           
        if len(half_anno)!=0:
            anno[",DPKN(KS-KS),"]="short_anno"
        else:
            anno[",DPKN(KS-KS),"]= "long_anno"
    else:
        anno[",DPKN(KS-KS),"]="none"
    if ",DPK(VU)," in matched_type:
           
        if len(half_anno)!=0:
            anno[",DPK(VU),"]="short_anno"
        elif len(parallel_anno)!=0:
            anno[",DPK(VU),"]= "short_anno_para"
        else:
            anno[",DPK(VU),"]="long_anno"
    else:
        anno[",DPK(VU),"]="none"
    
    if ",DME-1(R-KS)," in matched_type:
        if len(angle_anno)!=0:
            anno[",DME-1(R-KS),"]="angl"
        else:
            anno[",DME-1(R-KS),"]="dist"
    else:
        anno[",DME-1(R-KS),"]="none"
    
    
    return anno

def load_classification_table(file_path):
    """
    Load the classification table from a JSON file.
    """
    with open(file_path, 'r',encoding='UTF-8') as f:
        classification_table = json.load(f)  # Load the JSON file
    return classification_table


def generate_key(edge):
    # Generate a key that considers both original and reversed order
    if isinstance(edge[0], list):
        new_edge = edge[0]
    else:
        new_edge = edge
    return min(tuple(new_edge), tuple(reversed(new_edge)))
def is_toe(free_edge,cons_edge,max_free_edge_length):
    if (free_edge.length()<56 or free_edge.length()<=0.105*max_free_edge_length) and is_vertical_(free_edge.start_point,free_edge.end_point,cons_edge,epsilon=0.35):
        return True
    return False
def is_ks_corner(free_edge,last_free_edge,cons_edge,max_free_edge_length):
    if (not is_toe(free_edge,cons_edge,max_free_edge_length)) and (not is_vertical_(free_edge.start_point,free_edge.end_point,cons_edge,epsilon=0.35)) and isinstance(last_free_edge.ref,DLine) and free_edge.length() <= 100:
        return True
    return False

# 自由边约束过滤
def free_edges_sequence_classifier(classification_table, free_edges_sequence, reversed_free_edges_sequence, matched_type_list):
    matched_type = None
    # 自由边有趾端的匹配
    for b_type in matched_type_list:
        temp_free_edge_seq = classification_table[b_type]["free_edges"]
        if is_free_edges_equal(free_edges_sequence, temp_free_edge_seq) or is_free_edges_equal(reversed_free_edges_sequence, temp_free_edge_seq):
            matched_type = b_type if matched_type is None else f'{matched_type},{b_type}'
    
    # 自由边无趾端的匹配
    if matched_type is None:
        free_edges_sequence_copy = copy.deepcopy(free_edges_sequence)
        reversed_free_edges_sequence_copy = copy.deepcopy(reversed_free_edges_sequence)
        if free_edges_sequence_copy[0] == "toe":
            free_edges_sequence_copy[0] = "line"
        if free_edges_sequence_copy[-1] == "toe":
            free_edges_sequence_copy[-1] = "line"
        if reversed_free_edges_sequence_copy[0] == "toe":
            reversed_free_edges_sequence_copy[0] = "line"
        if reversed_free_edges_sequence_copy[-1] == "toe":
            reversed_free_edges_sequence_copy[-1] = "line"
        for b_type in matched_type_list:
            temp_free_edge_seq = classification_table[b_type]["free_edges"]
            if is_free_edges_equal(free_edges_sequence_copy, temp_free_edge_seq) or is_free_edges_equal(reversed_free_edges_sequence_copy, temp_free_edge_seq):
                matched_type = b_type if matched_type is None else f'{matched_type},{b_type}'
    
    return matched_type if matched_type is not None else "Unclassified"

# 固定边和角隅孔约束匹配
def conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence):
    matched_type = []
    non_conerhole_edges = []
    reversed_non_conerhole_edges = []
    conerhole_count = {}
    unrestricted_cornerhole_count = {}
    # unrestricted_cornerhole_type = [["line"], ["arc"]]
    unrestricted_cornerhole_type = [["line"]]
    unrestricted_cornerhole_num = 0
    # 去掉非自由边轮廓中的角隅孔，只保留固定边，对角隅孔进行统计
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

    # 固定边轮廓严格匹配+角隅孔各类型数量的不严格匹配（直线角隅孔可能不画）
    for key_name, row in classification_table.items():
        # step1: 自由边轮廓严格匹配
        # if not is_free_edges_equal(free_edges_sequence, row["free_edges"]) and not is_free_edges_equal(reversed_free_edges_sequence, row["free_edges"]):
        #     continue

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

        matched_type.append(key_name)

    return matched_type


def tidy_matched_type(matched_type):
    types=matched_type.split(',')
    result=''
    for ty in types:
        if ty!='':
            result=result+ty+','
    if result=='':
        result="Unclassified"        
    return result[:-1]

def find_cons_edge(poly_refs,seg):
    for s in poly_refs:
        if (not s.isCornerhole) and (not s.isConstraint):
            continue
        if s.start_point==seg.start_point or s.start_point == seg.end_point or s.end_point ==seg.start_point or s.end_point==seg.end_point:
            return s
        
def poly_classifier(all_anno,poly_refs, texts,dimensions,conerhole_num, poly_free_edges, edges, classification_file_path,info_json_path, keyname, is_output_json = False):
    classification_table = load_classification_table(classification_file_path)

    # Step 1: 获取角隅孔数

    # Step 2: 获取自由边的轮廓
    free_edges_sequence = []
    max_free_edge_length=float("-inf")
    for seg in poly_free_edges[0]:
        max_free_edge_length=max(max_free_edge_length,seg.length())
    for i, seg in enumerate(poly_free_edges[0]):
        if isinstance(seg.ref, DLine) or isinstance(seg.ref, DLwpolyline):
            if (i == 0 or i == len(poly_free_edges[0]) - 1):
                if i==0:
                    last_free_edge=poly_free_edges[0][1]
                else:
                    last_free_edge=poly_free_edges[0][-2]
                cons_edge=find_cons_edge(poly_refs,seg)
                # print(cons_edge)
                if is_toe(seg,cons_edge,max_free_edge_length):
                    free_edges_sequence.append("toe")
                elif is_ks_corner(seg,last_free_edge,cons_edge,max_free_edge_length):
                    free_edges_sequence.append("KS_corner")
                else:
                    free_edges_sequence.append("line")
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


    # step5: 固定边轮廓严格匹配+角隅孔非严格匹配（直线角隅孔可能不画）
    matched_type_list = conerhole_free_classifier(classification_table, conerhole_num, free_edges_sequence, reversed_free_edges_sequence, edges_sequence, reversed_edges_sequence)
    
    # step6: 自由边的筛选
    matched_type = free_edges_sequence_classifier(classification_table, free_edges_sequence,reversed_free_edges_sequence, matched_type_list)

    # print(matched_type)
    if len(matched_type.split(","))<=1 and matched_type!="Unclassified":
        return matched_type,classification_table[matched_type]
    elif matched_type=="Unclassified":
        return matched_type,None
    
    # step7: 易混淆筛选
    #for each mixed type, use the first type name as key(cluster_name),find the annoation

    # # 边界区分易混淆类细化
    # # DPK(VU-R1), DPK(VU-R)
    # cluster_name = "DPK(VU-R1)"
    # if cluster_name in matched_type:
    #     mixed_types = ["DPK(VU-R1)", "DPK(VU-R)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    
    # # BMA(VU), LBMA(R), LBMA(KS)
    # cluster_name = "BMA(VU)"
    # if cluster_name in matched_type:
    #     mixed_types = ["BMA(VU)", "LBMA(R)", "LBMA(KS)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    
    # # DMC-1(R-KS), DMC-1(KS-R)
    # cluster_name = "DMC-1(R-KS)"
    # if cluster_name in matched_type:
    #     mixed_types = ["DMC-1(R-KS)", "DMC-1(KS-R)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)

    # # DPKN(R-KS), DPKN(KS-R)
    # cluster_name = "DPKN(R-KS)"
    # if cluster_name in matched_type:
    #     mixed_types = ["DPKN(R-KS)", "DPKN(KS-R)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)

    # # BMB(R-KS), BMB(KS-R)
    # cluster_name = "BMB(R-KS)"
    # if cluster_name in matched_type:
    #     mixed_types = ["BMB(R-KS)", "BMB(KS-R)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    
    # # DPV(VU-KS), DPV-H(VU-KS)
    # cluster_name = "DPV(VU-KS)"
    # if cluster_name in matched_type:
    #     mixed_types = ["DPV(VU-KS)", "DPV-H(VU-KS)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)

    # # DPV-4(VU-VU), DPV-4(VVU-VVU)
    # cluster_name = "DPV-4(VU-VU)"
    # if cluster_name in matched_type:
    #     mixed_types = ["DPV-4(VU-VU)", "DPV-4(VVU-VVU)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)

    # # BR-2(KS-KS), BR-2(R-KS)
    # cluster_name = "BR-2(KS-KS)"
    # if cluster_name in matched_type:
    #     mixed_types = ["BR-2(KS-KS)", "BR-2(R-KS)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    
    # # DBA(R-KS), LDPKN-3(KS-R)
    # # cluster_name = "DBA(R-KS)"
    # # if cluster_name in matched_type:
    # #    mixed_types = ["DBA(R-KS)", "LDPKN-3(KS-R)"]
    # #    matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    
    # # DPKN-2(R-R) , DPKN-2(2R-KS)
    # cluster_name = "DPKN-2(R-R)"
    # if cluster_name in matched_type:
    #     mixed_types = ["DPKN-2(R-R) ", "DPKN-2(2R-KS)"]
    #     matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)
    matched_type=","+matched_type+','

    # anno=find_anno_info(matched_type,all_anno,poly_free_edges)
    
    # #classify inner mixed type by annotation

    # # DAB(VU-KS), DAB-1(VU-KS)
    # # cluster_name="DAB(VU-KS)"
    # # if cluster_name in matched_type:
    # #     if anno == "no_angle":
    # #         matched_type = "DAB(VU-KS)"
    # #     elif anno == "angl":
    # #         matched_type = "DAB-1(VU-KS)"
    # #     else:
    # #         matched_type = "DAB(VU-KS)"
    
    # # DAC(VU-R), DAC(VUF-R)
    # # print(anno)
    # cluster_name=",DAC(VU-R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "no_anno":
    #         matched_type = matched_type.replace(',DAC(VUF-R),',',,')
    #     elif anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DAC(VU-R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DAC(VUF-R),',',,')
    
    # # DPK(R), LDPK(R-R), LDPK-1(R-R)
    # cluster_name=",DPK(R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "long_anno":
    #         matched_type = matched_type.replace(',LDPK(R-R),',',,')
    #         matched_type = matched_type.replace(',LDPK-1(R-R),',',,')
    #     elif anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DPK(R),',',,')
    #         matched_type = matched_type.replace(',LDPK-1(R-R),',',,')
    #     elif anno[cluster_name] == "short_anno_para":
    #         matched_type = matched_type.replace(',LDPK(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK(R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',LDPK(R-R),',',,')
    #         matched_type = matched_type.replace(',LDPK-1(R-R),',',,')
    
    # # DAB(R-KS), DAB-1(R-KS)
    # # cluster_name="DAB(R-KS)"
    # # if cluster_name in matched_type:
    # #     if anno == "no_angle":
    # #         matched_type = "DAB(R-KS)"
    # #     elif anno == "angl":
    # #         matched_type = "DAB-1(R-KS)"
    # #     else:
    # #         matched_type = "DAB(R-KS)"

    # # DPKN(KS-KS), LDPKN(KS-KS)
    # cluster_name=",DPKN(KS-KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "long_anno":
    #         matched_type = matched_type.replace(',LDPKN(KS-KS),',',,')
    #     elif anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DPKN(KS-KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',LDPKN(KS-KS),',',,')

    # # DAD(R), DCD(R) 
    # cluster_name=",DAD(R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "dist_intersec":
    #         matched_type = matched_type.replace(',DCD(R),',',,')
    #     elif anno[cluster_name] == "no_dist":
    #         matched_type = matched_type.replace(',DAD(R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DAD(R),',',,')


    # # DPKN(KS-R), LDPKN(KS-R)
    # cluster_name=",DPKN(KS-R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "long_anno":
    #         matched_type = matched_type.replace(',LDPKN(KS-R),',',,')
    #     elif anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DPKN(KS-R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',LDPKN(KS-R),',',,')

    # # DPKN(VU-R), LDPKN(VU-R)
    # cluster_name=",DPKN(VU-R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "long_anno":
    #         matched_type = matched_type.replace(',LDPKN(VU-R),',',,')
    #     elif anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DPKN(VU-R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',LDPKN(VU-R),',',,')
    
    # # # DPKN(KS-KS), LDPKN(KS-KS)
    # # cluster_name="DPKN(KS-KS)"
    # # if cluster_name in matched_type:
    # #     if anno[cluster_name] == "long_anno":
    # #         matched_type = "DPKN(KS-KS)"
    # #     elif anno[cluster_name] == "short_anno":
    # #         matched_type = "LDPKN(KS-KS)"
    # #     else:
    # #         matched_type = "DPKN(KS-KS)"
    # # DAB-3(R-KS), LDBA(R-KS), BCB-1(R-KS)
    # cluster_name=",DAB-3(R-KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "angl_non_free":
    #         matched_type = matched_type.replace(',LDBA(R-KS),',',,')
    #         matched_type = matched_type.replace(',BCB-1(R-KS),',',,')
    #     elif anno[cluster_name] == "dist_intersec":
    #         matched_type = matched_type.replace(',DAB-3(R-KS),',',,')
    #         matched_type = matched_type.replace(',BCB-1(R-KS),',',,')
    #     elif anno[cluster_name] == "no_tangent":
    #         matched_type = matched_type.replace(',DAB-3(R-KS),',',,')
    #         matched_type = matched_type.replace(',LDBA(R-KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DAB-3(R-KS),',',,')
    #         matched_type = matched_type.replace(',BCB-1(R-KS),',',,')

    # # KL(R), KL-1(R), KL-2(R)
    # cluster_name=",KL(R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "ff":
    #         matched_type = matched_type.replace(',KL-0(R),',',,')
    #         matched_type = matched_type.replace(',KL-1(R),',',,')
    #         matched_type = matched_type.replace(',KL-2(R),',',,')
    #     elif anno[cluster_name] == "tt":
    #         matched_type = matched_type.replace(',KL(R),',',,')
    #         matched_type = matched_type.replace(',KL-0(R),',',,')
    #         matched_type = matched_type.replace(',KL-2(R),',',,')
    #     elif anno[cluster_name] == "tf":
    #         matched_type = matched_type.replace(',KL-0(R),',',,')
    #         matched_type = matched_type.replace(',KL(R),',',,')
    #         matched_type = matched_type.replace(',KL-1(R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',KL-0(R),',',,')
    #         matched_type = matched_type.replace(',KL-2(R),',',,')
    #         matched_type = matched_type.replace(',KL-1(R),',',,')
    
    # # KL(KS), KL-1(KS), KL-2(KS)
    # cluster_name=",KL(KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "ff":
    #         matched_type = matched_type.replace(',KL-0(KS),',',,')
    #         matched_type = matched_type.replace(',KL-2(KS),',',,')
    #         matched_type = matched_type.replace(',KL-1(KS),',',,')
    #     elif anno[cluster_name] == "tt":
    #         matched_type = matched_type.replace(',KL-0(KS),',',,')
    #         matched_type = matched_type.replace(',KL-2(KS),',',,')
    #         matched_type = matched_type.replace(',KL(KS),',',,')
    #     elif anno[cluster_name] == "tf":
    #         matched_type = matched_type.replace(',KL-0(KS),',',,')
    #         matched_type = matched_type.replace(',KL(KS),',',,')
    #         matched_type = matched_type.replace(',KL-1(KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',KL-0(KS),',',,')
    #         matched_type = matched_type.replace(',KL-2(KS),',',,')
    #         matched_type = matched_type.replace(',KL-1(KS),',',,')
    
    # # DPK-1(R), DPK-2(R-R), DPK-5(R-R), LDPK-3(R-R)
    # cluster_name=",DPK-1(R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "angl_non_free":
    #         matched_type = matched_type.replace(',DPK-2(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK-5(R-R),',',,')
    #         matched_type = matched_type.replace(',LDPK-3(R-R),',',,')
    #     elif anno[cluster_name] == "dist_adja":
    #         matched_type = matched_type.replace(',DPK-5(R-R),',',,')
    #         matched_type = matched_type.replace(',LDPK-3(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK-1(R),',',,')
    #     elif anno[cluster_name] == "angl_toe":
    #         matched_type = matched_type.replace(',DPK-2(R-R),',',,')
    #         matched_type = matched_type.replace(',LDPK-3(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK-1(R),',',,')
    #     elif anno[cluster_name] == "dist_intersec":
    #         matched_type = matched_type.replace(',DPK-2(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK-5(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK-1(R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DPK-2(R-R),',',,')
    #         matched_type = matched_type.replace(',DPK-5(R-R),',',,')
    #         matched_type = matched_type.replace(',LDPK-3(R-R),',',,')
    
    # # BR(R), BR-1(R), DAA(R)
    # cluster_name=",BR(R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "no_anno":
    #         matched_type = matched_type.replace(',BR-1(R),',',,')
    #         matched_type = matched_type.replace(',DAA(R),',',,')
    #     elif anno[cluster_name] == "angl_non_free":
    #         matched_type = matched_type.replace(',DAA(R),',',,')
    #         matched_type = matched_type.replace(',BR(R),',',,')
    #     elif anno[cluster_name] == "dist_intersec":
    #         matched_type = matched_type.replace(',BR-1(R),',',,')
    #         matched_type = matched_type.replace(',BR(R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',BR-1(R),',',,')
    #         matched_type = matched_type.replace(',DAA(R)),',',,')
    
    # # DPK(R-KS), LDPK-1(KS-R)
    # cluster_name=",DPK(R-KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "long_anno":
    #         matched_type = matched_type.replace(',LDPK-1(KS-R),',',,')
    #     elif anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DPK(R-KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',LDPK-1(KS-R),',',,')
    
    # # BR-1(KS), DAA(KS)
    # cluster_name=",BR-1(KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "angl_non_free":
    #         matched_type = matched_type.replace(',DAA(KS),',',,')
    #     elif anno[cluster_name] == "dist_intersec":
    #         matched_type = matched_type.replace(',BR-1(KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',BR-1(KS),',',,')
    
    # # DPV-4(R-KS), DPV-6(R-KS)
    # cluster_name=",DPV-4(R-KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "D_anno":
    #         matched_type = matched_type.replace(',DPV-6(R-KS),',',,')
    #     elif anno[cluster_name] == "no_anno":
    #         matched_type = matched_type.replace(',DPV-4(R-KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DPV-4(R-KS),',',,')
    
    # # LBMA-1(KS), BMA-1(KS)
    # cluster_name=",LBMA-1(KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "dist":
    #         matched_type = matched_type.replace(',LBMA-1(KS),',',,')
    #     elif anno[cluster_name] == "angl":
    #         matched_type = matched_type.replace(',BMA-1(KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',LBMA-1(KS),',',,')
    
    # # DAC(KS-KS),DAE(KS-KS)
    # cluster_name=",DAC(KS-KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "D":
    #         matched_type = matched_type.replace(',DAE(KS-KS),',',,')
    #     elif anno[cluster_name] == "notD":
    #         matched_type = matched_type.replace(',DAC(KS-KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DAC(KS-KS),',',,')

    # # DAC(R-R),DAE(R-R)
    # cluster_name=",DAC(R-R),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "D_anno":
    #         matched_type = matched_type.replace(',DAC(R-R),',',,')
    #     elif anno[cluster_name] == "no_anno":
    #         matched_type = matched_type.replace(',DAE(R-R),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DAE(R-R)),',',,')

    # #DPK(VU),DPKN-3(VU-VU),LDPK(VU-VU)
    # cluster_name=",DPK(VU),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "short_anno":
    #         matched_type = matched_type.replace(',DPK(VU),',',,')
    #         matched_type = matched_type.replace(',DPKN-3(VU-VU),',',,')
    #     elif anno[cluster_name] == "short_anno_para":
    #         matched_type = matched_type.replace(',DPK(VU),',',,')
    #         matched_type = matched_type.replace(',LDPK(VU-VU),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DPKN-3(VU-VU),',',,')
    #         matched_type = matched_type.replace(',LDPK(VU-VU),',',,')

    # #DME-1(R-KS),DME-2(R-KS)
    # cluster_name =",DME-1(R-KS),"
    # if cluster_name in matched_type:
    #     if anno[cluster_name] == "angl":
    #         matched_type = matched_type.replace(',DME-1(R-KS),',',,')
    #     elif anno[cluster_name] == "dist":
    #         matched_type = matched_type.replace(',DME-2(R-KS),',',,')
    #     else:
    #         matched_type = matched_type.replace(',DME-2(R-KS),',',,')

    matched_type=tidy_matched_type(matched_type)

    # step8: 考虑整体轮廓筛选
    if len(matched_type.split(","))<=1 and matched_type!="Unclassified":
        return matched_type,classification_table[matched_type]
    elif matched_type=="Unclassified":
        return matched_type,None
    # 去除自由边所有"KS_corner"以及轮廓中所有"["cornerhole", ["line"]]"
    free_edges_sequence = [item for item in free_edges_sequence if item != "KS_corner"]
    reversed_free_edges_sequence = [item for item in reversed_free_edges_sequence if item != "KS_corner"]
    edges_sequence = [item for item in edges_sequence if item != ["cornerhole", ["line"]]]
    reversed_edges_sequence = [item for item in reversed_edges_sequence if item != ["cornerhole", ["line"]]]

    edges_sequence.insert(0,["free", free_edges_sequence])
    reversed_edges_sequence.insert(0, ["free", reversed_free_edges_sequence])
    mixed_types = matched_type.split(',')
    matched_type = refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence)


    output_template=None
    # if matched_type == "Unclassified":
    #     pass
    # else:
    #     if len(matched_type.split(","))>1:
    #         first_type=matched_type.split(",")[0]
    #     else:
    #         first_type=matched_type

    #     #根据fisrt_type 进行输出的格式化
    #     output_template=classification_table[first_type]
    return matched_type,output_template

# 整体轮廓过滤    
def refine_poly_classifier(classification_table, mixed_types, edges_sequence, reversed_edges_sequence):
    matched_type = None
    max_similarity = -1  # 初始化最大相似度为 -1

    for type in mixed_types:
        temp_free_edge_seq = classification_table[type]["free_edges"]
        temp_edge_seq = classification_table[type]["non_free_edges"]
        tmp_edge_seq = []
        for edge in temp_edge_seq:
            if edge["type"] == "cornerhole" and edge["edges"] == ["line"]:
                continue
            else:
                tmp_edge_seq.append([edge["type"], edge["edges"]])
        temp_free_edge_seq = [item for item in temp_free_edge_seq if item != "KS_corner"]

        tmp_edge_seq.insert(0, ["free", temp_free_edge_seq])
        # 计算与 edges_sequence 的相似度
        similarity = calculate_similarity(tmp_edge_seq, edges_sequence)
        # 计算与 reversed_edges_sequence 的相似度
        reversed_similarity = calculate_similarity(tmp_edge_seq, reversed_edges_sequence)

        # 取两者中的最大值
        current_max_similarity = max(similarity, reversed_similarity)

        # 如果当前相似度大于最大相似度，则更新 matched_type
        if current_max_similarity > max_similarity:
            max_similarity = current_max_similarity
            matched_type = type
        elif current_max_similarity == max_similarity:
            matched_type = type if matched_type is None else f"{matched_type},{type}"

    return matched_type

def calculate_similarity(seq1, seq2):
    """
    计算两个序列的相似度。
    这里使用简单的匹配比例作为相似度度量。
    """
    if not seq1 or not seq2:
        return 0

    match_count = 0
    min_length = min(len(seq1), len(seq2))

    for i in range(min_length):
        if seq1[i] == seq2[i]:
            match_count += 1

    return match_count / min_length
