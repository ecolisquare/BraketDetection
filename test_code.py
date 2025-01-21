# def clip_line(rect_x_min, rect_x_max, rect_y_min, rect_y_max, start_point, end_point):
#     """
#     裁剪线段使其位于包围盒内，如果线段在包围盒内，则返回裁剪后的起点和终点。
#     :param rect_x_min: 包围盒最小x坐标
#     :param rect_x_max: 包围盒最大x坐标
#     :param rect_y_min: 包围盒最小y坐标
#     :param rect_y_max: 包围盒最大y坐标
#     :param start_point: 线段起点 (x1, y1)
#     :param end_point: 线段终点 (x2, y2)
#     :return: (裁剪后的起点, 裁剪后的终点)，如果线段完全在包围盒外，返回None
#     """
#     x1, y1 = start_point
#     x2, y2 = end_point

#     dx = x2 - x1
#     dy = y2 - y1

#     p = [-dx, dx, -dy, dy]
#     q = [x1 - rect_x_min, rect_x_max - x1, y1 - rect_y_min, rect_y_max - y1]

#     t_min = 0
#     t_max = 1

#     for i in range(4):
#         if p[i] == 0:  # 线段平行于边界
#             if q[i] < 0:
#                 return None  # 完全在外部
#         else:
#             t = q[i] / p[i]
#             if p[i] < 0:
#                 t_min = max(t_min, t)  # 更新 t_min
#             else:
#                 t_max = min(t_max, t)  # 更新 t_max

#     if t_min > t_max:
#         return None  # 完全在外部

#     # 计算裁剪后的起点和终点
#     clipped_start = (x1 + t_min * dx, y1 + t_min * dy)
#     clipped_end = (x1 + t_max * dx, y1 + t_max * dy)

#     return clipped_start, clipped_end


# # 示例用法
# rect_x_min = 2
# rect_x_max = 6
# rect_y_min = 2
# rect_y_max = 6
# start_point = (3, 3)
# end_point = (4, 4)

# result = clip_line(rect_x_min, rect_x_max, rect_y_min, rect_y_max, start_point, end_point)
# if result:
#     print(f"裁剪后的起点: {result[0]}, 终点: {result[1]}")
# else:
#     print("线段完全在包围盒外")



from utils import calculate_prior_angle
from element import *
a1=calculate_prior_angle(DPoint(1233,1241),DPoint(747,1121),DPoint(739,1078))
a2=calculate_prior_angle(DPoint(1233,1241),DPoint(747,1121),DPoint(723,1085))
a3=calculate_prior_angle(DPoint(3099,1347),DPoint(3099,1721),DPoint(3070,1730))
a4=calculate_prior_angle(DPoint(3099,1347),DPoint(3099,1721),DPoint(3099,1771))
print(a1,a2,a3,a4)
