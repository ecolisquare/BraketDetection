# import ezdxf
# import ezdxf.bbox
# import ezdxf.disassemble
# import ezdxf.entities 
# import json
# import os
# import math
# import numpy as np
# from pathlib import Path
# # Entity 
# # (g)type
# # (g)color
# # (g) bound : {x1,y1,x2,y2} # left-bottom # right-top
# # (g) layerName

# # (text)insert
# # (text)content
# # (text)height

# # (line)start
# # (line)end
# # (line)linetype

# def getColor(entity):
#     try:
#         directColor = entity.dxf.color
#         if directColor == 0:
#             layer = entity.doc.layers.get(entity.dxf.layer)
#             return layer.color
#         elif directColor == 256:
#             fa = entity.source_block_reference
#             if (fa is None):
#                 layer = entity.doc.layers.get(entity.dxf.layer)
#                 return layer.color
#             else:
#                 raise NotImplementedError('随块，但是块的颜色是fa.color吗？')
#                 return fa.color

#         else:
#             return directColor
#     except NotImplementedError as e:
#         return 256

# def getLineWeight(doc,entity):
#     try:
#         dircetLw = entity.dxf.lineweight
#         if dircetLw == -1:
#             layer = doc.layers.get(entity.dxf.layer)
#             return getLineWeight(doc,layer)
#         elif dircetLw == -2:
#             raise NotImplementedError('随块尚未实现')
#         elif dircetLw == -3:
#             return doc.header.get('$LWDEFAULT',0) # dxf未说明则代表是0
#         else:
#             return dircetLw
#     except NotImplementedError as e:
#         print(e)
#         exit()

# def convertEntity(type,entity):
#     bb= ezdxf.bbox.extents([entity])
#     return {
#         'type' : type,
#         'color' : getColor(entity),
#         'layerName' : entity.dxf.layer,
#         'handle':entity.dxf.handle,
#         'bound' :{
#             'x1' : bb.extmin[0],
#             'y1' : bb.extmin[1],
#             'x2' : bb.extmax[0],
#             'y2' : bb.extmax[1] ,
#         },
#     }

# def convertCircle(entity : ezdxf.entities.Circle):
#     mid = convertEntity('circle',entity)
#     mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
#     mid['radius'] = entity.dxf.radius
#     return mid
  
# def convertText(entity: ezdxf.entities.Text):
#     mid = convertEntity('text',entity)
#     mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
#     mid['content'] = entity.dxf.text
#     mid['height'] = entity.dxf.height
#     return mid

# def convertLine(entity:ezdxf.entities.LineEdge):
#     mid = convertEntity('line',entity)
#     mid['start'] = [entity.dxf.start[0],entity.dxf.start[1]]
#     mid['end'] = [entity.dxf.end[0],entity.dxf.end[1]]
#     mid['linetype'] = entity.dxf.linetype
#     return mid

# def convertArc(entity:ezdxf.entities.ArcEdge):
#     mid = convertEntity('arc',entity)
#     mid['linetype'] = entity.dxf.linetype
#     mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
#     mid['radius'] = entity.dxf.radius
#     mid['startAngle'] = entity.dxf.start_angle
#     mid['endAngle'] = entity.dxf.end_angle
#     # mid['isClosed'] = entity.is_closed

#     return mid

# def convertPolyLine(entity:ezdxf.entities.Polyline):
#     mid = convertEntity('polyline',entity)
#     mid['isClosed'] = entity.is_closed
#     mid['linetype'] = entity.dxf.linetype
#     mid['vertices'] = [[x.dxf.location[0],x.dxf.location[1]] for x in entity.vertices]
#     return mid

# def convertSpline(entity:ezdxf.entities.Spline):
#     mid = convertEntity('spline',entity)
#     mid['linetype'] = entity.dxf.linetype
#     mid['vertices'] = [x.tolist()[0:2] for x in entity.control_points]
#     return mid

# def convertLWPolyline(entity:ezdxf.entities.Spline):
#     mid = convertEntity('lwpolyline',entity)
#     mid['isClosed'] = entity.is_closed
#     mid['linetype'] = entity.dxf.linetype
#     mid['hasArc'] = entity.has_arc

#     # mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
#     # if not entity.has_arc:
#     #     mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
#     #     mid['verticesType']  = [["line"] for x in entity.vertices()]
#     # else:
#     mid['vertices'] = []
#     mid['verticesType'] = []

#     # print(entity.dxf.handle, entity.dxf.count)
#     for i in range(entity.dxf.count  - 1):
#         start_point = entity.__getitem__(i)
#         end_point = entity.__getitem__(i + 1)
#         # print(start_point[-1])

#         if start_point[-1] != 0.:
#             arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
#             center, start_angle, end_angle, radius  = arc 
#             x, y = center
#             mid["vertices"].append([x, y, start_angle, end_angle, radius])
#             mid['verticesType'].append("arc")
#         else:
#             x_s, y_s = start_point[:2]
#             x_e, y_e = end_point[:2]
#             mid["vertices"].append([x_s, y_s, x_e, y_e])
#             mid['verticesType'].append("line")
            
#         # if entity.is_closed:
#         #     start_point = entity.__getitem__(entity.dxf.count - 1)
#         #     end_point = entity.__getitem__(0)

#         #     if start_point[-1] != 0.:
#         #             arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
#         #             center, start_angle, end_angle, radius  = arc 
#         #             x, y = center
#         #             mid["vertices"].append([x, y, start_angle, end_angle, radius])
#         #             mid['verticesType'].append("arc")
#         #     else:
#         #         x, y = start_point[:2]
#         #         mid["vertices"].append([x, y])
#         #         mid['verticesType'].append("line")


#     # for i in range(num_vertices):
#     #     # print(vertices)
#     #     print(vertices[i])
    

#     return mid

# def convertDimension(entity:ezdxf.entities.Dimension):
#     mid = convertEntity('dimension',entity)
#     mid['measurement'] = entity.dxf.actual_measurement
#     mid['text'] = entity.dxf.text
#     mid['dimtype'] = entity.dxf.dimtype
#     mid['textpos'] = [entity.dxf.text_midpoint[0],entity.dxf.text_midpoint[1]]
#     mid['defpoint1'] = [entity.dxf.defpoint[0],entity.dxf.defpoint[1]]
#     mid['defpoint2'] = [entity.dxf.defpoint2[0],entity.dxf.defpoint2[1]]
#     mid['defpoint3'] = [entity.dxf.defpoint3[0],entity.dxf.defpoint3[1]]
#     mid['defpoint4'] = [entity.dxf.defpoint4[0],entity.dxf.defpoint4[1]]
#     mid['defpoint5'] = [entity.dxf.defpoint5[0],entity.dxf.defpoint5[1]]
    

#     return mid

# def convertSolid(entity:ezdxf.entities.Solid):
#     mid = convertEntity('solid',entity)
#     mid['vtx0'] = [entity.dxf.vtx0[0],entity.dxf.vtx0[1]]
#     mid['vtx1'] = [entity.dxf.vtx1[0],entity.dxf.vtx1[1]]
#     mid['vtx2'] = [entity.dxf.vtx2[0],entity.dxf.vtx2[1]]
#     mid['vtx3'] = [entity.dxf.vtx3[0],entity.dxf.vtx3[1]]
#     return mid

# def convertAttdef(entity:ezdxf.entities.AttDef):
#     mid = convertEntity('attdef',entity)
#     mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
#     mid['content'] = entity.dxf.text
#     mid['height'] = entity.dxf.height
#     mid['text'] = entity.dxf.text
#     mid['rotation'] = entity.dxf.rotation
#     return mid

# def convertInsert(entity:ezdxf.entities.Insert, block_list):
#     mid = convertEntity('insert',entity)
#     mid['blockName'] = entity.dxf.name
#     mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
#     mid['scales'] = [entity.dxf.xscale, entity.dxf.yscale]
#     mid['rotation'] = entity.dxf.rotation
    

#     attrib_list = []
#     for attrib in entity.attribs:
#         print(entity.dxf.name, attrib.dxf.handle, attrib.dxf.tag, attrib.dxf.text)
#         attrib_list.append({
#             "attribHandle":attrib.dxf.handle, 
#             "attribTag":attrib.dxf.tag,
#             "attribText":attrib.dxf.text
#         })
#     # print(entity.dxf.handle, mid["bound"])

#     mid["attribs"] = attrib_list

#     for k in mid["bound"]:
#         if np.isinf(mid["bound"][k]):
#             mid = None
#             break

#     if entity.dxf.name not in block_list:
#         block_list.append(entity.dxf.name)


#     # print(mid)
#     return mid, block_list

# def convertMText(entity:ezdxf.entities.MText):
#     mid = convertEntity('mtext',entity)
#     mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
#     mid['width'] = entity.dxf.width
#     mid['text'] = entity.dxf.text
#     return mid

# def convertEllipse(entity:ezdxf.entities.Ellipse):
#     mid = convertEntity('ellipse',entity)
#     mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
#     mid['major_axis'] =[entity.dxf.major_axis[0],entity.dxf.major_axis[1]]
#     mid['ratio'] = entity.dxf.ratio
#     print('In convert : 遇到椭圆')
#     return mid

# def convertRegion(entity:ezdxf.entities.Region):
#     mid = convertEntity('region',entity)
#     print('In convert : 遇到Region,包围盒会变成无穷')
#     return mid


# def convertLeader(entity:ezdxf.entities.Leader):
#     mid = convertEntity('leader',entity)
#     return mid


# def convertBlocks(blocks:ezdxf.sections.blocks, block_list):

#     block_info = {}

#     for block_name in block_list:

#         block = blocks.get(block_name)
#         res = []
#         for e in block:
#             if (isEntityHidden(e)):
#                 print(f'检测到隐藏元素 {e.dxf.handle} {e.dxftype()}')
#                 continue
#             j = None
#             if (e.dxftype() == 'LINE'):
#                 j = convertLine(e)
#             elif e.dxftype() == 'CIRCLE':
#                 j = convertCircle(e)
#             elif e.dxftype() == 'TEXT':
#                 # 空文字直接删了
#                 if (e.dxf.text != ''):
#                     j = convertText(e)
#             elif e.dxftype() == 'ARC':
#                 j = convertArc(e)
#             elif e.dxftype() == 'POLYLINE':
#                 j = convertPolyLine(e)
#             elif e.dxftype() == 'SPLINE':
#                 j = convertSpline(e)
#             elif e.dxftype() == 'LWPOLYLINE':
#                 j = convertLWPolyline(e)
#             elif e.dxftype() == 'DIMENSION':
#                 j = convertDimension(e)
#             elif e.dxftype() == 'ATTDEF':
#                 j = convertAttdef(e)
#             elif e.dxftype() == 'MTEXT':
#                 j = convertMText(e)
#             elif e.dxftype() == 'ELLIPSE':
#                 j = convertEllipse(e)
#             elif e.dxftype() == 'REGION':
#                 print('region 没什么好获取的信息')
#                 # j = convertRegion(e)
#             elif e.dxftype() == 'HATCH':
#                 continue                 #未实现
#             elif e.dxftype() == 'OLE2FRAME':
#                 continue                #未实现
#             elif e.dxftype() == 'LEADER':
#                 j = convertLeader(e)               #未实现
#             elif e.dxftype() == 'INSERT':
#                 continue                #未实现
#             elif e.dxftype() == 'SOLID':
#                 # j = convertSolid(e)
#                 continue               #存在问题
#             else:
#                 # raise NotImplementedError(f'遇到了未见过的类型 {e.dxftype()}')
#                 print(f'Block中遇到了未见过的类型 {e.dxftype()} ')
#             if(j is not None):
#                 res.append(j)
        
#         block_info[block_name] = res

#     return block_info

# def isEntityHidden(entity):
#     """
#     判断给定的实体是否是隐藏的。

#     参数:
#     entity (DXFEntity): 要检查的实体对象。

#     返回:
#     bool: 如果实体隐藏返回True，否则返回False。
#     """
#     # 获取实体所在的图层
#     doc = entity.doc
#     layer = doc.layers.get(entity.dxf.layer)
    
#     # 检查图层状态
#     is_layer_off = layer.is_off()
#     # is_layer_frozen = layer.is_frozen_in_layout(model_space)

#     # # # 检查实体自身的可见性
#     # is_entity_color_invisible = entity.dxf.color == 0
#     # is_entity_ltype_invisible = entity.dxf.linetype == "None"
#     # is_entity_invisible = entity.dxf.invisible
#     # # 结论
#     # return is_layer_off  or is_entity_invisible or is_entity_color_invisible or is_entity_ltype_invisible 
#     return is_layer_off


# # 此函数内部与converBlocks内部有些不同，例如convertBlock还要处理Solid,ATTDEF，故暂且用到convertBlock中
# def analyzeNonBlockEntity(e):
#     j = None
#     type = e.dxftype()
#     if (type == 'LINE'):
#         j = convertLine(e)
#     elif type == 'CIRCLE':
#         j = convertCircle(e)
#     elif type == 'TEXT':
#         # 空文字直接删了
#         if (e.dxf.text != ''):
#             j = convertText(e)
#     elif type == 'ARC':
#         j = convertArc(e)
#     elif type == 'SOLID':
#         j = convertSolid(e)  
#     elif type == 'POLYLINE':
#         j = convertPolyLine(e)
#     elif type == 'SPLINE':
#         j = convertSpline(e)
#     elif type == 'LWPOLYLINE':
#         j = convertLWPolyline(e)
#     elif type == 'DIMENSION':
#         j = convertDimension(e)
#     elif type == 'MTEXT':
#         j = convertMText(e)
#     elif type == 'ELLIPSE':
#         j = convertEllipse(e)
#     elif type == 'REGION':
#         print('region 没什么好获取的信息')
#         # j = convertRegion(e)
#     # elif type == 'HATCH':
#     #     pass            #未实现
#     # elif type == 'OLE2FRAME':
#     #     pass            #未实现
#     elif type == 'LEADER':
#         j = convertLeader(e)           

#     else:
#         # raise NotImplementedError(f'遇到了未见过的类型 {type}')
#         print(e.dxf.handle)
#         print(f'Nonblock中遇到了未见过的类型 {e.dxf.handle, type}')

#     return j

# # dxfname不需要携带.dxf后缀
# def dxf2json(dxfpath,dxfname,output_folder):
#     print('----dxf2json START-----')
#     dxfname=dxfname.split('.')[0]
#     relpath = os.path.join(dxfpath,dxfname)
#     relpath_dxf = relpath + '.dxf'
#     print('dxf relative path is ' + relpath)
#     doc = ezdxf.readfile(relpath_dxf,encoding='utf-8')
#     msp = doc.modelspace()
#     res = [] 
#     block_list = []



#     for e in msp:
#         if (isEntityHidden(e)):
#             print(f'检测到隐藏元素 {e.dxf.handle} {e.dxftype()}')
#             continue
#         j = None
#         try:
#             # print(e.dxf.handle, e.dxftype())
#             if (e.dxftype() == 'INSERT'):
#                 j, block_list = convertInsert(e, block_list)
#             else:
#                 j = analyzeNonBlockEntity(e)
#         except Exception as e:
#             print(e)

#         if(j is not None):
#             res.append(j)


#     blocks = convertBlocks(doc.blocks, block_list)
    
#     res = [res, blocks]
#     json_str= json.dumps(res, indent=4)
#     json_name = os.path.join(output_folder, dxfname) + ".json"
#     # json_name = relpath + '.json'
#     with open(json_name,'w',encoding='utf-8') as f:
#         f.write(json_str)
#         print(f'Writing to {json_name} finished!!')
#         print('-----dxf2json END------')

# if __name__ == "__main__":
    
#     dxfpath = '/home/user10/code/BraketDetection/data'
#     dxfname = 'small11.dxf'
#     dxf2json(dxfpath,dxfname, dxfpath)
    # folder_path = '/home/user10/code/BraketDetection/data'
    # output_foler='/home/user10/code/BraketDetection/data'
    # for filename in os.listdir(folder_path):
    #     # 检查文件是否是JSON文件
    #     if filename.endswith('.dxf'):
    #         file_path = os.path.join(folder_path, filename)
    #         name = os.path.splitext(filename)[0]
    #         output_path = os.path.join(output_foler, name)
    #         # training_data_output_path = os.path.join(training_data_output_folder, name)
    #         print(f"正在处理文件: {file_path}")
            
    #         # 打开并读取JSON文件内容
    #         try:
    #             dxf2json(folder_path,filename, output_foler)
    #         except Exception as e:
    #             print(f"处理文件 {file_path} 时出错: {e}")


    # dxfpath = './split'
    # folder_path = Path('split')
    # names = [f.stem for f in folder_path.glob('*.dxf')]
    # for name in names:
    #     dxf2json(dxfpath,name)
    
   

import ezdxf
import ezdxf.bbox
import ezdxf.disassemble
import ezdxf.entities 
import json
import os
import math
import numpy as np
import math
from ezdxf.xclip import XClip
# Entity 
# (g)type
# (g)color
# (g) bound : {x1,y1,x2,y2} # left-bottom # right-top
# (g) layerName

# (text)insert
# (text)content
# (text)height

# (line)start
# (line)end
# (line)linetype
PI_CIRCLE = 2*3.1415

def approximate_equal(a:float ,b :float,ellipse = 1e-6):
    if (abs(a-b) < ellipse):
        return True
    return False

def vector_to_angle(vector):
    x, y = vector
    angle = math.atan2(y, x)    
    if angle < 0:
        angle += PI_CIRCLE
    return angle

def getColor(entity):
    try:
        directColor = entity.dxf.color
        if directColor == 0:
            return 7
        elif directColor == 256:
            fa = entity.source_block_reference
            if (fa is None):
                layer = entity.doc.layers.get(entity.dxf.layer)
                return layer.color
            else:
                raise NotImplementedError('随块，但是块的颜色是fa.color吗？')
                return fa.color

        else:
            return directColor
    except NotImplementedError as e:
        return 256

def getLineWeight(doc,entity):
    try:
        dircetLw = entity.dxf.lineweight
        if dircetLw == -1:
            layer = doc.layers.get(entity.dxf.layer)
            return getLineWeight(doc,layer)
        elif dircetLw == -2:
            raise NotImplementedError('随块尚未实现')
        elif dircetLw == -3:
            return doc.header.get('$LWDEFAULT',0) # dxf未说明则代表是0
        else:
            return dircetLw
    except NotImplementedError as e:
        print(e)
        exit()


def getLineType(doc, entity):
    try:
        linetype = entity.dxf.linetype

        if linetype == "BYLAYER":
            layer = doc.layers.get(entity.dxf.layer)
            return layer.dxf.linetype
        
        elif linetype == "ByBlock":

            blockname = entity.dxf.owner
            block = doc.blocks.get(blockname)

            print("Byblock", entity.dxf.handle,blockname, block)

            if block:
                linetype = "Byblock " + block.dxf.linetype
            else:
                linetype = "Byblock"

            return linetype
        else:
            return linetype
    except NotImplemented as e:
        print(e)
        exit()


def convertEntity(type,entity):
    bb= ezdxf.bbox.extents([entity])
    return {
        'type' : type,
        'color' : getColor(entity),
        'layerName' : entity.dxf.layer,
        'handle':entity.dxf.handle,
        'bound' :{
            'x1' : bb.extmin[0],
            'y1' : bb.extmin[1],
            'x2' : bb.extmax[0],
            'y2' : bb.extmax[1] ,
        },
    }

def convertCircle(entity : ezdxf.entities.Circle):
    mid = convertEntity('circle',entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['radius'] = entity.dxf.radius
    return mid
  
def convertText(entity: ezdxf.entities.Text):
    insert_point = [entity.dxf.insert[0], entity.dxf.insert[1]]
    x = insert_point[0]
    y = insert_point[1]
    c_height = entity.dxf.height
    c_width = c_height * entity.dxf.width
    rotation = entity.dxf.rotation
    flag = int(entity.dxf.text_generation_flag) if entity.dxf.hasattr("text_generation_flag") else 0
    mirrored_x = bool(flag & 2)  # 左右反向
    mirrored_y = bool(flag & 4)  # 上下倒置
    halign = entity.dxf.halign
    valign = entity.dxf.valign
    theta = math.radians(rotation)
    content = entity.dxf.text
    width = len(content) * c_width

    dx, dy = 0, 0

    # 水平对齐
    if halign == 4:
        dx = -width / 2
    elif halign == 2:
        dx = -width

    # 垂直对齐
    if valign == 3:
        dy = -c_height
    elif valign == 2:
        dy = -c_height / 2

    # 四个角点（局部坐标）
    local_corners = [
        (dx, dy),                  # 左下
        (dx + width, dy),          # 右下
        (dx + width, dy + c_height), # 右上
        (dx, dy + c_height)          # 左上
    ]

    if mirrored_x:
        local_corners = [(-cx, cy) for cx, cy in local_corners]
    if mirrored_y:
        local_corners = [(cx, -cy) for cx, cy in local_corners]
    
    world_corners = []
    for cx, cy in local_corners:
        wx = x + cx * math.cos(theta) - cy * math.sin(theta)
        wy = y + cx * math.sin(theta) + cy * math.cos(theta)
        world_corners.append((wx, wy))


    mid = convertEntity('text',entity)
    mid['insert'] = insert_point
    mid['content'] = content
    mid['height'] = c_height
    mid['rotation'] = rotation
    mid['width'] = entity.dxf.width
    mid['bound'] = {
        'x1': min(p[0] for p in world_corners),
        'x2': max(p[0] for p in world_corners),
        'y1': min(p[1] for p in world_corners),
        'y2': max(p[1] for p in world_corners)
    }
    return mid

def convertLine(doc, entity:ezdxf.entities.LineEdge):
    mid = convertEntity('line',entity)
    mid['start'] = [entity.dxf.start[0],entity.dxf.start[1]]
    mid['end'] = [entity.dxf.end[0],entity.dxf.end[1]]
    mid['linetype'] = getLineType(doc, entity)
    return mid

def convertArc(doc, entity:ezdxf.entities.ArcEdge):
    mid = convertEntity('arc',entity)
    mid['linetype'] = getLineType(doc, entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['radius'] = entity.dxf.radius
    mid['startAngle'] = entity.dxf.start_angle
    mid['endAngle'] = entity.dxf.end_angle
    # mid['isClosed'] = entity.is_closed

    return mid

def convertPolyLine(doc, entity:ezdxf.entities.Polyline):
    mid = convertEntity('polyline',entity)
    mid['isClosed'] = entity.is_closed
    mid['linetype'] = getLineType(doc, entity)
    mid['vertices'] = [[x.dxf.location[0],x.dxf.location[1]] for x in entity.vertices]
    return mid

def convertSpline(doc, entity:ezdxf.entities.Spline):
    mid = convertEntity('spline',entity)
    mid['linetype'] = getLineType(doc, entity)
    mid['vertices'] = [x.tolist()[0:2] for x in entity.control_points]
    return mid

def convertLWPolyline(doc, entity:ezdxf.entities.Spline):
    mid = convertEntity('lwpolyline',entity)
    mid['isClosed'] = entity.is_closed
    mid['linetype'] = getLineType(doc, entity)
    mid['hasArc'] = entity.has_arc

    # mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
    # if not entity.has_arc:
    #     mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
    #     mid['verticesType']  = [["line"] for x in entity.vertices()]

    mid['vertices'] = []
    mid['verticesType'] = []
    mid['verticesWidth'] = []
    # print(entity.dxf.handle, entity.dxf.count)
    for p in entity.get_points():
        x,y,s_w,e_w,bulge=p
        mid['verticesWidth'].append([s_w,e_w])
    for i in range(entity.dxf.count  - 1):
        start_point = entity.__getitem__(i)
        end_point = entity.__getitem__(i + 1)
        # print(start_point[-1])

        if start_point[-1] != 0.:
            arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
            center, start_angle, end_angle, radius  = arc 
            x, y = center
            mid["vertices"].append([x, y, start_angle, end_angle, radius])
            mid['verticesType'].append("arc")
        else:
            x1, y1 = start_point[:2]
            x2, y2 = end_point[:2]
            mid["vertices"].append([x1, y1, x2, y2])
            mid['verticesType'].append("line")
        
    if entity.is_closed:
        start_point = entity.__getitem__(entity.dxf.count - 1)
        end_point = entity.__getitem__(0)

        if start_point[-1] != 0.:
                arc = ezdxf.math.bulge_to_arc(start_point[:2], end_point[:2], start_point[-1])
                center, start_angle, end_angle, radius  = arc 
                x, y = center
                mid["vertices"].append([x, y, start_angle, end_angle, radius])
                mid['verticesType'].append("arc")
        else:
            x1, y1 = start_point[:2]
            x2, y2 = end_point[:2]

            mid["vertices"].append([x1, y1, x2, y2])
            mid['verticesType'].append("line")


    # for i in range(num_vertices):
    #     # print(vertices)
    #     print(vertices[i])
    

    return mid

def convertDimension(entity:ezdxf.entities.Dimension):
    mid = convertEntity('dimension',entity)
    mid['measurement'] = entity.dxf.actual_measurement
    mid['text'] = entity.dxf.text
    mid['dimtype'] = entity.dxf.dimtype
    mid['textpos'] = [entity.dxf.text_midpoint[0],entity.dxf.text_midpoint[1]]
    mid['defpoint1'] = [entity.dxf.defpoint[0],entity.dxf.defpoint[1]]
    mid['defpoint2'] = [entity.dxf.defpoint2[0],entity.dxf.defpoint2[1]]
    mid['defpoint3'] = [entity.dxf.defpoint3[0],entity.dxf.defpoint3[1]]
    mid['defpoint4'] = [entity.dxf.defpoint4[0],entity.dxf.defpoint4[1]]
    mid['defpoint5'] = [entity.dxf.defpoint5[0],entity.dxf.defpoint5[1]]
    

    return mid

def convertSolid(entity:ezdxf.entities.Solid):
    mid = convertEntity('solid',entity)
    mid['vtx0'] = [entity.dxf.vtx0[0],entity.dxf.vtx0[1]]
    mid['vtx1'] = [entity.dxf.vtx1[0],entity.dxf.vtx1[1]]
    mid['vtx2'] = [entity.dxf.vtx2[0],entity.dxf.vtx2[1]]
    mid['vtx3'] = [entity.dxf.vtx3[0],entity.dxf.vtx3[1]]
    return mid

def convertAttdef(entity:ezdxf.entities.AttDef):
    mid = convertEntity('attdef',entity)
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['content'] = entity.dxf.text
    mid['height'] = entity.dxf.height
    mid['text'] = entity.dxf.text
    mid['rotation'] = entity.dxf.rotation
    return mid

def convertInsert(entity:ezdxf.entities.Insert, block_list):
    mid = convertEntity('insert',entity)
    mid['blockName'] = entity.dxf.name
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['scales'] = [entity.dxf.xscale, entity.dxf.yscale]
    mid['rotation'] = entity.dxf.rotation
    
    clip = XClip(entity)

    if clip.has_clipping_path and clip.is_clipping_enabled:

        # print(entity.dxf.name, entity.dxf.handle, clip.get_wcs_clipping_path().vertices)

        coords = np.array(clip.get_wcs_clipping_path().vertices)
        # print(coords)

        x1 = np.min(coords[:, 0])
        x2 = np.max(coords[:, 0])
        
        x1 = max(x1, mid["bound"]["x1"])
        x2 = min(x2, mid["bound"]["x2"])

        y1 = np.min(coords[:, 1])
        y2 = np.max(coords[:, 1])

        y1 = max(y1, mid["bound"]["y1"])
        y2 = min(y2, mid["bound"]["y2"])

        mid["bound"]={
            'x1' : x1,
            'y1' : y1,
            'x2' : x2,
            'y2' : y2
        }
    
    attrib_list = []
    for attrib in entity.attribs:
        print(entity.dxf.name, attrib.dxf.handle, attrib.dxf.tag, attrib.dxf.text)
        attrib_list.append({
            "attribHandle":attrib.dxf.handle, 
            "attribTag":attrib.dxf.tag,
            "attribText":attrib.dxf.text
        })
    # print(entity.dxf.handle, mid["bound"])

    mid["attribs"] = attrib_list

    for k in mid["bound"]:
        if np.isinf(mid["bound"][k]):
            mid = None
            break

    if entity.dxf.name not in block_list:
        block_list.append(entity.dxf.name)


    # print(mid)
    return mid, block_list

def convertMText(entity:ezdxf.entities.MText):
    text_content = entity.dxf.text
    text_content = text_content.replace('\P','\n')
    lines =text_content.split('\n')

    base_point = entity.dxf.insert
    c_height = entity.dxf.char_height
    c_width = c_height * 0.8
    width = entity.dxf.width
    rotation = entity.dxf.rotation
    line_spacing = entity.dxf.line_spacing_factor if entity.dxf.hasattr("line_spacing_factor") else 1.0
    attach = entity.dxf.attachment_point

    # 文本高度
    n_lines = len(lines)
    rect_height = c_height * (1.0 + (n_lines - 1) * line_spacing)

    # 文本宽度
    rect_width = 0.0
    for line in lines:
        rect_width = max(rect_width, c_width * len(line))
    
    # 计算整体左上角基准点
    if attach in (1, 4, 7):
        base_dx = 0
    elif attach in (2, 5, 8):
        base_dx = -rect_width / 2
    else:
        base_dx = -rect_width
    
    if attach in (1, 2, 3):
        base_dy = 0
    elif attach in (4, 5, 6):
        base_dy = rect_height / 2
    else:
        base_dy = rect_height
    
    base_top_left = [base_point[0] + base_dx, base_point[1] + base_dy]

    # 构建每一行text
    mids = []
    for i, line in enumerate(lines):
        y_offset = -i * c_height * line_spacing * 5 / 3 - c_height
        line_width = len(line) * c_width
        if attach in (1, 4, 7):
            x_offset = 0
        elif attach in (2, 5, 8):
            x_offset = (rect_width - line_width) / 2
        else:
            x_offset = rect_width - line_width
        
        insert_point = [base_top_left[0] + x_offset, base_top_left[1] + y_offset]

        mid = convertEntity('text', entity)
        mid['insert'] = insert_point
        mid['content'] = line
        mid['height'] = c_height
        mid['rotation'] = 0
        mid['width'] = 0.8
        mid['bound'] = {
            'x1': insert_point[0],
            'x2': insert_point[0] + line_width,
            'y1': insert_point[1],
            'y2': insert_point[1] + c_height
            }
        mids.append(mid)

    return mids

def convertEllipse(entity:ezdxf.entities.Ellipse):
    mid = convertEntity('ellipse',entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['major_axis'] =[entity.dxf.major_axis[0],entity.dxf.major_axis[1]]
    mid['ratio'] = entity.dxf.ratio
    mid['extrusion'] = [entity.dxf.extrusion[0],entity.dxf.extrusion[1],entity.dxf.extrusion[2]]
    mid['start_param'] = entity.dxf.start_param
    mid['end_param'] = entity.dxf.end_param
    # 下面自行计算起始角度和终止角度 start_theta end_theat
    start_theta = entity.dxf.start_param
    end_theta = entity.dxf.end_param
    # 如果是翻转了的椭圆，角度 = 2pi - theta
    if (approximate_equal(mid['extrusion'][2],-1)):
        start_theta,end_theta = PI_CIRCLE -end_theta,PI_CIRCLE - start_theta 
    # 如果主轴方向指向负半轴 角度 += pi 
    major_axis_rotation = vector_to_angle(mid['major_axis'])
    start_theta = (start_theta + major_axis_rotation) % (PI_CIRCLE)
    end_theta = (end_theta + major_axis_rotation) % (PI_CIRCLE)
    if (start_theta > end_theta ):
        if approximate_equal(end_theta,0,1e-2):
            end_theta = 6.282
        if approximate_equal(start_theta,6.282,1e-2):
            start_theta =0
    mid['calc_start_theta'] = start_theta
    mid['calc_end_theta'] = end_theta

    return mid

def convertRegion(entity:ezdxf.entities.Region):
    mid = convertEntity('region',entity)
    print('In convert : 遇到Region,包围盒会变成无穷')
    return mid

def convertHatch(entity:ezdxf.entities.Hatch):
    
    mid = convertEntity('hatch', entity)
    
    x1 = mid["bound"]["x1"]
    x2 = mid["bound"]["x2"]
    y1 = mid["bound"]["y1"]
    y2 = mid["bound"]["y2"]
    
    s = math.fabs((x2 - x1) * (y2 - y1))


    mid["paths"]=[]

    for edges in entity.paths.paths:
        p = []

        if isinstance(edges, ezdxf.entities.boundary_paths.EdgePath):
            edges = edges.edges
            for edge in edges:
                
                if isinstance(edge, ezdxf.entities.boundary_paths.LineEdge):
                    res = {}
                    res["edge_type"] = "line"
                    res["coords"] = [
                        edge.start[0], 
                        edge.start[1], 
                        edge.end[0], 
                        edge.end[1]
                    ]
                    p.append(res)
                    

                elif isinstance(edge, ezdxf.entities.boundary_paths.EllipseEdge):

                    res = {}
                    res["edge_type"] = "ellipse"
                    
                    res['center'] = [edge.center[0],edge.center[1]]
                    res['major_axis'] =[edge.major_axis[0],edge.major_axis[1]]
                    res['ratio'] = edge.ratio

                    res['start_param'] = edge.start_param
                    res['end_param'] = edge.end_param
                    # 下面自行计算起始角度和终止角度 start_theta end_theat
                    start_theta = edge.start_param
                    end_theta = edge.end_param

                    # 如果主轴方向指向负半轴 角度 += pi 
                    major_axis_rotation = vector_to_angle(res['major_axis'])
                    start_theta = (start_theta + major_axis_rotation) % (PI_CIRCLE)
                    end_theta = (end_theta + major_axis_rotation) % (PI_CIRCLE)
                    if (start_theta > end_theta ):
                        if approximate_equal(end_theta,0,1e-2):
                            end_theta = 6.282
                        if approximate_equal(start_theta,6.282,1e-2):
                            start_theta =0
                    res['calc_start_theta'] = start_theta
                    res['calc_end_theta'] = end_theta

                    p.append(res)

                elif isinstance(edge, ezdxf.entities.boundary_paths.ArcEdge):
                    res = {}
                    res["edge_type"] = "arc"

                    res["center"] = [edge.center[0], edge.center[1]]
                    res['radius'] = edge.radius
                    res['start_angle'] = edge.start_angle
                    res['end_angle'] = edge.end_angle

                    p.append(res)
                

            
        elif isinstance(edges, ezdxf.entities.boundary_paths.PolylinePath):

            res = {}
            res['edge_type'] = "polyline"
            res["coords"] = []
            for v in edges.vertices:
                res["coords"].append([v[0], v[1]])

            p.append(res)
        mid['paths'].append(p)
    # print(type())


    
    return mid

def convertLeader(entity:ezdxf.entities.Leader):
    mid = convertEntity('leader',entity)
    return mid


def convertBlocks(doc, block_list):


    blocks = doc.blocks
    block_info = {}

    for block in blocks:    
        block_name = block.name
        res = []
        for e in block:
            if (isEntityHidden(e)):
                # print(f'块内检测到隐藏元素 {e.dxf.handle} {e.dxftype()}')
                continue
            j = None
            if (e.dxftype() == 'LINE'):
                j = convertLine(doc, e)
            elif e.dxftype() == 'CIRCLE':
                j = convertCircle(e)
            elif e.dxftype() == 'TEXT':
                # 空文字直接删了
                if (e.dxf.text != ''):
                    j = convertText(e)
            elif e.dxftype() == 'ARC':
                j = convertArc(doc, e)
            elif e.dxftype() == 'POLYLINE':
                j = convertPolyLine(doc, e)
            elif e.dxftype() == 'SPLINE':
                j = convertSpline(doc, e)
            elif e.dxftype() == 'LWPOLYLINE':
                j = convertLWPolyline(doc, e)
            elif e.dxftype() == 'DIMENSION':
                j = convertDimension(e)
            elif e.dxftype() == 'ATTDEF':
                j = convertAttdef(e)
            elif e.dxftype() == 'MTEXT':
                j = convertMText(e)
            elif e.dxftype() == 'ELLIPSE':
                j = convertEllipse(e)
            elif e.dxftype() == 'REGION':
                print('region 没什么好获取的信息')
                # j = convertRegion(e)
            elif e.dxftype() == 'HATCH':
                j = convertHatch(e)
            elif e.dxftype() == 'OLE2FRAME':
                continue                #未实现
            elif e.dxftype() == 'LEADER':
                j = convertLeader(e)               #未实现
            elif e.dxftype() == 'INSERT':
                j, _ = convertInsert(e, [])


            elif e.dxftype() == 'SOLID':
                j = convertSolid(e)
                # continue               #存在问题
            else:
                # raise NotImplementedError(f'遇到了未见过的类型 {e.dxftype()}')
                # print(f'Block中遇到了未见过的类型 {e.dxftype()} ')
                continue
            if(j is not None):
                if isinstance(j, list):
                    res.extend(j)
                else:
                    res.append(j)
        
        block_info[block_name] = res

    return block_info

def isEntityHidden(entity):
    """
    判断给定的实体是否是隐藏的。

    参数:
    entity (DXFEntity): 要检查的实体对象。

    返回:
    bool: 如果实体隐藏返回True，否则返回False。
    """
    # 获取实体所在的图层
    doc = entity.doc
    layer = doc.layers.get(entity.dxf.layer)
    
    # 检查图层状态
    is_layer_off = layer.is_off()
    is_layer_frozen = layer.is_frozen()
    # is_layer_frozen = layer.is_frozen_in_layout(model_space)

    # # 检查实体自身的可见性
    # is_entity_color_invisible = entity.dxf.color == 0
    # is_entity_ltype_invisible = entity.dxf.linetype == "None"
    is_entity_invisible = entity.dxf.invisible
    # 结论
    return is_layer_off  or is_entity_invisible or is_layer_frozen 


# 此函数内部与converBlocks内部有些不同，例如convertBlock还要处理Solid,ATTDEF，故暂且用到convertBlock中
def analyzeNonBlockEntity(doc, e):
    j = None
    type = e.dxftype()
    if (type == 'LINE'):
        j = convertLine(doc, e)
    elif type == 'CIRCLE':
        j = convertCircle(e)
    elif type == 'TEXT':
        # 空文字直接删了
        if (e.dxf.text != ''):
            j = convertText(e)
    elif type == 'ARC':
        j = convertArc(doc, e)
    elif type == 'SOLID':
        j = convertSolid(e)  
    elif type == 'POLYLINE':
        j = convertPolyLine(doc, e)
    elif type == 'SPLINE':
        j = convertSpline(doc, e)
    elif type == 'LWPOLYLINE':
        j = convertLWPolyline(doc, e)
    elif type == 'DIMENSION':
        j = convertDimension(e)
    elif type == 'MTEXT':
        j = convertMText(e)
    elif type == 'ELLIPSE':
        j = convertEllipse(e)
    elif type == 'REGION':
        print('region 没什么好获取的信息')
        # j = convertRegion(e)
    elif type == 'HATCH':
        j = convertHatch(e)
    # elif type == 'OLE2FRAME':
    #     pass            #未实现
    elif type == 'LEADER':
        j = convertLeader(e)           

    else:
        # raise NotImplementedError(f'遇到了未见过的类型 {type}')
        print(e.dxf.handle)
        # print(f'Nonblock中遇到了未见过的类型 {e.dxf.handle, type}')

    return j

# dxfname不需要携带.dxf后缀
def dxf2json(dxfpath,dxfname,output_folder):
    print('----dxf2json START-----')
    dxfname=dxfname.split('.')[0]
    relpath = os.path.join(dxfpath,dxfname)
    relpath_dxf = relpath + '.dxf'
    print('dxf relative path is ' + relpath)
    doc = ezdxf.readfile(relpath_dxf,encoding='utf-8')
    msp = doc.modelspace()
    res = [] 
    block_list = []



    for e in msp:
        if (isEntityHidden(e)):
            print(f'检测到隐藏元素 {e.dxf.handle} {e.dxftype()} 位置{ezdxf.bbox.extents([e])}')
            continue
        j = None
        try:
            # print(e.dxf.handle, e.dxftype())
            if (e.dxftype() == 'INSERT'):
                j, block_list = convertInsert(e, block_list)
            else:
                j = analyzeNonBlockEntity(doc, e)
        except Exception as e:
            print(e)

        if(j is not None):
            if isinstance(j, list):
                res.extend(j)
            else:
                res.append(j)


    blocks = convertBlocks(doc, block_list)
    
    res = [res, blocks]
    json_str= json.dumps(res, indent=4)
    json_name = os.path.join(output_folder, dxfname) + ".json"
    # json_name = relpath + '.json'
    with open(json_name,'w',encoding='utf-8') as f:
        f.write(json_str)
        print(f'Writing to {json_name} finished!!')
        print('-----dxf2json END------')

if __name__ == "__main__":
    
    dxfpath = './data/data'
    dxfname = 'test12.dxf'
    dxf2json(dxfpath,dxfname, dxfpath)
    # folder_path = './data/dimension_data'
    # output_foler='./data/dimension_data'
    # for filename in os.listdir(folder_path):
    #     # 检查文件是否是JSON文件
    #     if filename.endswith('.dxf'):
    #         file_path = os.path.join(folder_path, filename)
    #         name = os.path.splitext(filename)[0]
    #         output_path = os.path.join(output_foler, name)
    #         # training_data_output_path = os.path.join(training_data_output_folder, name)
    #         print(f"正在处理文件: {file_path}")
            
    #         # 打开并读取JSON文件内容
    #         try:
    #             dxf2json(folder_path,filename, output_foler)
    #         except Exception as e:
    #             print(f"处理文件 {file_path} 时出错: {e}")


    # dxfpath = './split'
    # folder_path = Path('split')
    # names = [f.stem for f in folder_path.glob('*.dxf')]
    # for name in names:
    #     dxf2json(dxfpath,name)
    
   

