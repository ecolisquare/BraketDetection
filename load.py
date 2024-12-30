import ezdxf
import ezdxf.bbox
import ezdxf.disassemble
import ezdxf.entities 
import json
import os
import math
import numpy as np
from pathlib import Path
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

def getColor(entity):
    try:
        directColor = entity.dxf.color
        if directColor == 0:
            layer = entity.doc.layers.get(entity.dxf.layer)
            return layer.color
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
    mid = convertEntity('text',entity)
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['content'] = entity.dxf.text
    mid['height'] = entity.dxf.height
    return mid

def convertLine(entity:ezdxf.entities.LineEdge):
    mid = convertEntity('line',entity)
    mid['start'] = [entity.dxf.start[0],entity.dxf.start[1]]
    mid['end'] = [entity.dxf.end[0],entity.dxf.end[1]]
    mid['linetype'] = entity.dxf.linetype
    return mid

def convertArc(entity:ezdxf.entities.ArcEdge):
    mid = convertEntity('arc',entity)
    mid['linetype'] = entity.dxf.linetype
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['radius'] = entity.dxf.radius
    mid['startAngle'] = entity.dxf.start_angle
    mid['endAngle'] = entity.dxf.end_angle
    # mid['isClosed'] = entity.is_closed

    return mid

def convertPolyLine(entity:ezdxf.entities.Polyline):
    mid = convertEntity('polyline',entity)
    mid['isClosed'] = entity.is_closed
    mid['linetype'] = entity.dxf.linetype
    mid['vertices'] = [[x.dxf.location[0],x.dxf.location[1]] for x in entity.vertices]
    return mid

def convertSpline(entity:ezdxf.entities.Spline):
    mid = convertEntity('spline',entity)
    mid['linetype'] = entity.dxf.linetype
    mid['vertices'] = [x.tolist()[0:2] for x in entity.control_points]
    return mid

def convertLWPolyline(entity:ezdxf.entities.Spline):
    mid = convertEntity('lwpolyline',entity)
    mid['isClosed'] = entity.is_closed
    mid['linetype'] = entity.dxf.linetype
    mid['hasArc'] = entity.has_arc

    # mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
    if not entity.has_arc:
        mid['vertices'] = [[x[0].item(),x[1].item()] for x in entity.vertices()]
        mid['verticesType']  = [["line"] for x in entity.vertices()]
    else:
        mid['vertices'] = []
        mid['verticesType'] = []

        # print(entity.dxf.handle, entity.dxf.count)
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
                x, y = start_point[:2]
                mid["vertices"].append([x, y])
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
                x, y = start_point[:2]
                mid["vertices"].append([x, y])
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
    mid = convertEntity('mtext',entity)
    mid['insert'] = [entity.dxf.insert[0],entity.dxf.insert[1]]
    mid['width'] = entity.dxf.width
    mid['text'] = entity.dxf.text
    return mid

def convertEllipse(entity:ezdxf.entities.Ellipse):
    mid = convertEntity('ellipse',entity)
    mid['center'] = [entity.dxf.center[0],entity.dxf.center[1]]
    mid['major_axis'] =[entity.dxf.major_axis[0],entity.dxf.major_axis[1]]
    mid['ratio'] = entity.dxf.ratio
    print('In convert : 遇到椭圆')
    return mid

def convertRegion(entity:ezdxf.entities.Region):
    mid = convertEntity('region',entity)
    print('In convert : 遇到Region,包围盒会变成无穷')
    return mid


def convertLeader(entity:ezdxf.entities.Leader):
    mid = convertEntity('leader',entity)
    return mid


def convertBlocks(blocks:ezdxf.sections.blocks, block_list):

    block_info = {}

    for block_name in block_list:

        block = blocks.get(block_name)
        res = []
        for e in block:
            if (isEntityHidden(e)):
                print(f'检测到隐藏元素 {e.dxf.handle} {e.dxftype()}')
                continue
            j = None
            if (e.dxftype() == 'LINE'):
                j = convertLine(e)
            elif e.dxftype() == 'CIRCLE':
                j = convertCircle(e)
            elif e.dxftype() == 'TEXT':
                # 空文字直接删了
                if (e.dxf.text != ''):
                    j = convertText(e)
            elif e.dxftype() == 'ARC':
                j = convertArc(e)
            elif e.dxftype() == 'POLYLINE':
                j = convertPolyLine(e)
            elif e.dxftype() == 'SPLINE':
                j = convertSpline(e)
            elif e.dxftype() == 'LWPOLYLINE':
                j = convertLWPolyline(e)
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
                continue                 #未实现
            elif e.dxftype() == 'OLE2FRAME':
                continue                #未实现
            elif e.dxftype() == 'LEADER':
                j = convertLeader(e)               #未实现
            elif e.dxftype() == 'INSERT':
                continue                #未实现
            elif e.dxftype() == 'SOLID':
                # j = convertSolid(e)
                continue               #存在问题
            else:
                # raise NotImplementedError(f'遇到了未见过的类型 {e.dxftype()}')
                print(f'Block中遇到了未见过的类型 {e.dxftype()} ')
            if(j is not None):
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
    # is_layer_frozen = layer.is_frozen_in_layout(model_space)

    # # # 检查实体自身的可见性
    # is_entity_color_invisible = entity.dxf.color == 0
    # is_entity_ltype_invisible = entity.dxf.linetype == "None"
    # is_entity_invisible = entity.dxf.invisible
    # # 结论
    # return is_layer_off  or is_entity_invisible or is_entity_color_invisible or is_entity_ltype_invisible 
    return is_layer_off


# 此函数内部与converBlocks内部有些不同，例如convertBlock还要处理Solid,ATTDEF，故暂且用到convertBlock中
def analyzeNonBlockEntity(e):
    j = None
    type = e.dxftype()
    if (type == 'LINE'):
        j = convertLine(e)
    elif type == 'CIRCLE':
        j = convertCircle(e)
    elif type == 'TEXT':
        # 空文字直接删了
        if (e.dxf.text != ''):
            j = convertText(e)
    elif type == 'ARC':
        j = convertArc(e)
    elif type == 'SOLID':
        j = convertSolid(e)  
    elif type == 'POLYLINE':
        j = convertPolyLine(e)
    elif type == 'SPLINE':
        j = convertSpline(e)
    elif type == 'LWPOLYLINE':
        j = convertLWPolyline(e)
    elif type == 'DIMENSION':
        j = convertDimension(e)
    elif type == 'MTEXT':
        j = convertMText(e)
    elif type == 'ELLIPSE':
        j = convertEllipse(e)
    elif type == 'REGION':
        print('region 没什么好获取的信息')
        # j = convertRegion(e)
    # elif type == 'HATCH':
    #     pass            #未实现
    # elif type == 'OLE2FRAME':
    #     pass            #未实现
    elif type == 'LEADER':
        j = convertLeader(e)           

    else:
        # raise NotImplementedError(f'遇到了未见过的类型 {type}')
        print(e.dxf.handle)
        print(f'Nonblock中遇到了未见过的类型 {e.dxf.handle, type}')

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
            print(f'检测到隐藏元素 {e.dxf.handle} {e.dxftype()}')
            continue
        j = None
        try:
            # print(e.dxf.handle, e.dxftype())
            if (e.dxftype() == 'INSERT'):
                j, block_list = convertInsert(e, block_list)
            else:
                j = analyzeNonBlockEntity(e)
        except Exception as e:
            print(e)

        if(j is not None):
            res.append(j)


    blocks = convertBlocks(doc.blocks, block_list)
    
    res = [res, blocks]
    json_str= json.dumps(res, indent=4)
    json_name = os.path.join(output_folder, dxfname) + ".json"
    # json_name = relpath + '.json'
    with open(json_name,'w',encoding='utf-8') as f:
        f.write(json_str)
        print(f'Writing to {json_name} finished!!')
        print('-----dxf2json END------')

if __name__ == "__main__":
    
    dxfpath = '/home/user10/code/BraketDetection/data'
    dxfname = 'FR22te.dxf'
    dxf2json(dxfpath,dxfname, dxfpath)
    # folder_path = '/home/user10/code/BraketDetection/data'
    # output_foler='/home/user10/code/BraketDetection/data'
    # for filename in os.listdir(folder_path):
    #     # 检查文件是否是JSON文件
    #     if filename.endswith('.json'):
    #         file_path = os.path.join(folder_path, filename)
    #         name = os.path.splitext(filename)[0]
    #         output_path = os.path.join(output_foler, name)
    #         # training_data_output_path = os.path.join(training_data_output_folder, name)
    #         print(f"正在处理文件: {file_path}")
            
            # 打开并读取JSON文件内容
            try:
                dxf2json(folder_path,filename, output_foler)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")


    # dxfpath = './split'
    # folder_path = Path('split')
    # names = [f.stem for f in folder_path.glob('*.dxf')]
    # for name in names:
    #     dxf2json(dxfpath,name)
    
   

