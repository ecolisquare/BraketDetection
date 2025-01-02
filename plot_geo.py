import matplotlib.pyplot as plt
import numpy as np
from element import *
import os
def p_minus(a,b):
    return DPoint(a.x-b.x,a.y-b.y)
def p_add(a,b):
    return DPoint(a.x+b.x,a.y+b.y)
def p_mul(a,k):
    return DPoint(a.x*k,a.y*k)
# 输出多边形所在原始图形
def plot_geometry(segments, path):
    fig, ax = plt.subplots()
    for segment in segments:
        if isinstance(segment.ref, DLine):
            x_values = [segment.ref.start_point.x, segment.ref.end_point.x]
            y_values = [segment.ref.start_point.y, segment.ref.end_point.y]
            ax.plot(x_values, y_values, 'b-', lw=2)
        
        elif isinstance(segment.ref, DLwpolyline):
            x_values = [p.x for p in segment.ref.points]
            y_values = [p.y for p in segment.ref.points]
            if segment.ref.isClosed:
                x_values.append(segment.ref.points[0].x)
                y_values.append(segment.ref.points[0].y)
            ax.plot(x_values, y_values, 'g-', lw=2)
        
        elif isinstance(segment.ref, DArc):
            if segment.ref.end_angle < segment.ref.start_angle:
                end_angle = segment.ref.end_angle + 360
            else:
                end_angle = segment.ref.end_angle
            theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
            x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
            y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
            ax.plot(x_arc, y_arc, 'r-', lw=2)

    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()

def coordinatesmap_(p:DPoint,insert,scales,rotation):
    rr=rotation/180*math.pi
    cosine=math.cos(rr)
    sine=math.sin(rr)

    # x,y=(p[0]*scales[0]+100)/200,(p[1]*scales[1]+100)/200
    x,y=((cosine*p[0]-sine*p[1])*scales[0])+insert[0],((sine*p[0]+cosine*p[1])*scales[1])+insert[1]
    return DPoint(x,y)
def transform_point_(point,meta):
    return coordinatesmap_(point,meta.insert,meta.scales,meta.rotation)

#输出封闭多边形
def plot_polys(point_map,segments,path):
    fig, ax = plt.subplots()
    # for segment in segments:
    #     vs,ve=segment.start_point,segment.end_point
    #     if len(point_map[vs])>2:
    #         # print(p.x,p.y)
    #         ax.plot(vs.x, vs.y, 'g.')
    #         # ax.text(vs.x, vs.y,str([s.length() for s in point_map[vs]]), fontsize=12, color='blue', rotation=90)
    #     if len(point_map[ve])>2:
    #         ax.plot(ve.x, ve.y, 'g.')
    #         # ax.text(ve.x, ve.y, str([s.length() for s in point_map[ve]]), fontsize=12, color='blue', rotation=90)
    for i, segment in enumerate(segments):
        x_values = [segment.start_point.x, segment.end_point.x]
        y_values = [segment.start_point.y, segment.end_point.y]
        color = 'red' if i % 2 == 0 else 'blue'
        ax.plot(x_values, y_values, color=color, lw=2)  # 使用不同的颜色
    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()


def expandFixedLengthGeo(segList,dist,both=True):


    new_seglist=[] 
    n=len(segList)

    for seg in segList:
        p1=seg[0]
        p2=seg[1]
        v=(p2[0]-p1[0],p2[1]-p1[1])
        l=math.sqrt(v[0]*v[0]+v[1]*v[1])
        if l<=0.25:
            continue
        v=(v[0]/l*dist,v[1]/l*dist)
        vs=DPoint(p1[0]-v[0],p1[1]-v[1]) if both else DPoint(p1[0],p1[1])
        ve=DPoint(p2[0]+v[0],p2[1]+v[1])
        new_seglist.append(DSegment(vs,ve,seg.ref))
    return new_seglist

def plot_info_poly(segments, path,texts,dimensions,stifferners,others=[]):
    fig, ax = plt.subplots()
    t_map={}
    for segment in stifferners:
        x_values = [segment.start_point.x, segment.end_point.x]
        y_values = [segment.start_point.y, segment.end_point.y]
        ax.plot(x_values, y_values, color='green', linestyle='--', linewidth=2)
    # for segment in others:
    #     x_values = [segment.start_point.x, segment.end_point.x]
    #     y_values = [segment.start_point.y, segment.end_point.y]
    #     ax.plot(x_values, y_values, color='blue', linestyle='--', linewidth=2)
    for t_t in texts:
        t=t_t[0]
        pos=t_t[1]
        content=t.content
        if pos not in t_map:
            t_map[pos]=[]
            t_map[pos].append([content,t_t[2],t_t[3]])
        else:
            t_map[pos].append([content,t_t[2],t_t[3]])
    for pos,cs in t_map.items():
        ax.text(pos.x, pos.y, cs, fontsize=12, color='blue', rotation=0)
    for d_t in dimensions:
        d=d_t[0]
        pos=d_t[1]
        content=d.text
        if  d.dimtype==32 or d.dimtype==33 or d.dimtype==161 or d.dimtype==160:
            l0=p_minus(d.defpoints[0],d.defpoints[2])
            l1=p_minus(d.defpoints[1],d.defpoints[2])
            d10=l0.x*l1.x+l0.y*l1.y
            d00=l0.x*l0.x+l0.y*l0.y
            if d00 <1e-4:
                x=d.defpoints[1]
            else:
                x=p_minus(p_add(d.defpoints[1],l0),p_mul(l0,d10/d00))
            d1,d2,d3,d4=d.defpoints[0], x,d.defpoints[1],d.defpoints[2]
            ss=[DSegment(d1,d4),DSegment(d4,d3),DSegment(d3,d2)]
            sss=[DSegment(d2,d1)]
            ss=expandFixedLengthGeo(ss,25,True)
            sss=expandFixedLengthGeo(sss,100,True)
            q=sss[0].end_point
            sss=expandFixedLengthGeo(sss,100,False)
            for s in ss:
                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
            for s in sss:
                ax.plot([s.start_point.x,s.end_point.x], [s.start_point.y,s.end_point.y], color="#FF0000", lw=2,linestyle='--')
            ax.arrow(sss[0].end_point.x, sss[0].end_point.y, d1.x-sss[0].end_point.x, d1.y-sss[0].end_point.y, head_width=20, head_length=20, fc='red', ec='red')
            ax.arrow(sss[0].start_point.x, sss[0].start_point.y, d2.x-sss[0].start_point.x, d2.y-sss[0].start_point.y, head_width=20, head_length=20, fc='red', ec='red')
            perp_vx, perp_vy = sss[0].start_point.x - sss[0].end_point.x, sss[0].start_point.y-sss[0].end_point.y
            rotation_angle = np.arctan2(-perp_vy, -perp_vx) * (180 / np.pi)
            ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
        
        elif d.dimtype==37:
            a,b,o=d.defpoints[1],d.defpoints[2],d.defpoints[3]
            r=DSegment(d.defpoints[0],o).length()
            ra=DSegment(a,o).length()
            rb=DSegment(b,o).length()
            oa_=p_mul(p_minus(a,o),r/ra)
            ob_=p_mul(p_minus(b,o),r/rb)
            ao_=p_mul(oa_,-1)
            bo_=p_mul(ob_,-1)
            a_= p_add(o, oa_)
            b_ = p_add(o, ob_)
            ia_=p_add(o,ao_)
            ib_=p_add(o,bo_)
            delta=p_mul(DPoint(oa_.y,-oa_.x),3)
            sp=p_add(delta,a_)
            ax.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red',linestyle='--')
            ax.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red',linestyle='--')
            ax.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
            q=p_mul(p_add(a_,sp),0.5)
            rotation_angle = np.arctan2(-delta.y, delta.x) * (180 / np.pi)
            ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
        elif d.dimtype==34:
            a,b_,b,o=d.defpoints[0],d.defpoints[1],d.defpoints[2],d.defpoints[3]
            r=DSegment(d.defpoints[4],o).length()
            ra=DSegment(a,o).length()
            rb=DSegment(b,o).length()
            oa_=p_mul(p_minus(a,o),r/ra)
            ob_=p_mul(p_minus(b,o),r/rb)
            ao_=p_mul(oa_,-1)
            bo_=p_mul(ob_,-1)
            a_= p_add(o, oa_)
            b_ = p_add(o, ob_)
            ia_=p_add(o,ao_)
            ib_=p_add(o,bo_)
            delta=p_mul(DPoint(oa_.y,-oa_.x),3)
            sp=p_add(delta,a_)
            ax.arrow(ia_.x, ia_.y, a_.x-ia_.x,a_.y-ia_.y, head_width=20, head_length=20, fc='red', ec='red',linestyle='--')
            ax.arrow(ib_.x, ib_.y, b_.x-ib_.x,b_.y-ib_.y, head_width=20, head_length=20, fc='red', ec='red',linestyle='--')
            ax.plot([sp.x,a_.x], [sp.y,a_.y], color="#FF0000", lw=2,linestyle='--')
            q=p_mul(p_add(a_,sp),0.5)
            rotation_angle = np.arctan2(delta.y, -delta.x) * (180 / np.pi)
            ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
        elif d.dimtype==163:
            a,b=d.defpoints[0],d.defpoints[3]
            o=p_mul(p_add(a,b),0.5)
            ss=[DSegment(a,b)]
            ss=expandFixedLengthGeo(ss,100)
            ss=expandFixedLengthGeo(ss,100,False)
            a_,b_=ss[0].start_point,ss[0].end_point
            ax.arrow(a_.x, a_.y, a.x-a_.x,a.y-a_.y, head_width=20, head_length=20, fc='red', ec='red')
            ax.arrow(b_.x, b_.y, b.x-b_.x,b.y-b_.y, head_width=20, head_length=20, fc='red', ec='red')
            q=p_add(p_mul(b_,0.7),p_mul(b,0.3))
            ab=p_minus(b,a)
            rotation_angle = np.arctan2(ab.y, -ab.x) * (180 / np.pi)
            ax.text(q.x, q.y, d.text,rotation=rotation_angle,color="#EEC933", fontsize=15)
            
            
    for i, segment in enumerate(segments):
        if segment.isCornerhole:
            color = "#FF0000"
        elif segment.isConstraint:
            color = "#00FF00"
            if segment.isPart:
                color="#00AA00"
        else:
            color = "#0000FF"
        if isinstance(segment.ref, DArc):
            if segment.ref.end_angle < segment.ref.start_angle:
                end_angle = segment.ref.end_angle + 360
            else:
                end_angle = segment.ref.end_angle
            theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
            x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
            y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
            # p=transform_point_(DPoint(x_arc,y_arc),segment.ref.meta)
            ax.plot(x_arc, y_arc, color, lw=2)
        else:
            x_values = [segment.start_point.x, segment.end_point.x]
            y_values = [segment.start_point.y, segment.end_point.y]
            ax.plot(x_values, y_values, color, lw=2)
    ax.set_aspect('equal', 'box')
    plt.savefig(path)
    plt.close()

def outputRes(segments,point_map,polys,resPNGPath,drawIntersections=False,drawLines=False,drawPolys=False):
    fig, ax = plt.subplots()
    if drawLines:
        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k-')
    if drawIntersections:
        for p,ss in point_map.items():
            if len(ss)>1:
                # print(p.x,p.y)
                plt.plot(p.x, p.y, 'r.')
    if drawPolys:
        for poly in polys:
            for i, segment in enumerate(poly):
                if segment.isCornerhole:
                    color = "red"
                elif segment.isConstraint:
                    color = "green"
                else:
                    color = "blue"
                if isinstance(segment.ref, DArc):
                    if segment.ref.end_angle < segment.ref.start_angle:
                        end_angle = segment.ref.end_angle + 360
                    else:
                        end_angle = segment.ref.end_angle
                    theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
                    x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
                    y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
                    # p=transform_point_(DPoint(x_arc,y_arc),segment.ref.meta)
                    ax.plot(x_arc, y_arc, color, lw=2)
                else:
                    x_values = [segment.start_point.x, segment.end_point.x]
                    y_values = [segment.start_point.y, segment.end_point.y]
                    ax.plot(x_values, y_values, color, lw=2)

    plt.gca().axis('equal')
    plt.savefig(resPNGPath)
    print(f"结果图保存于:{resPNGPath}")
    fig.clf()

def output_training_img(polys, segments, training_img_output_folder, name):
    # 如果输出文件夹不存在，则创建
    os.makedirs(training_img_output_folder, exist_ok=True)

    for i, poly in enumerate(polys):
        fig, ax = plt.subplots()

        output_file = os.path.join(training_img_output_folder, f"{name}_{i}.png")

        for seg in segments:
            vs, ve = seg.start_point, seg.end_point
            plt.plot([vs.x, ve.x], [vs.y, ve.y], 'k-')

        for segment in poly:
            color = "red"

            if isinstance(segment.ref, DArc):
                if segment.ref.end_angle < segment.ref.start_angle:
                    end_angle = segment.ref.end_angle + 360
                else:
                    end_angle = segment.ref.end_angle
                theta = np.linspace(np.radians(segment.ref.start_angle), np.radians(end_angle), 100)
                x_arc = segment.ref.center.x + segment.ref.radius * np.cos(theta)
                y_arc = segment.ref.center.y + segment.ref.radius * np.sin(theta)
                ax.plot(x_arc, y_arc, color, lw=2)
            else:
                x_values = [segment.start_point.x, segment.end_point.x]
                y_values = [segment.start_point.y, segment.end_point.y]
                ax.plot(x_values, y_values, color, lw=2)
            
        plt.gca().axis('equal')
        plt.savefig(output_file)
        plt.close()