# from scipy.optimize import differential_evolution
# import numpy as np

# # 定义目标函数
# def objective(x):
#     p, N = x
#     return p * N

# # 定义约束函数
# def constraint(x):
#     p, N = x
#     return (1 - p) ** (4 * N) - 1e-6  # Constraint must be <= 0

# # 定义边界
# bounds = [(1e-6, 1 - 1e-6), (1, 4)]  # p in (0, 1), N is a positive integer

# # 自定义约束处理（惩罚法）
# def penalty(x):
#     cons = constraint(x)
#     return max(0, cons) * 1e6  # Add a large penalty for violations

# # 定义优化目标（带约束）
# def constrained_objective(x):
#     return objective(x) + penalty(x)

# # 使用 differential_evolution 进行全局优化
# result = differential_evolution(
#     constrained_objective,
#     bounds=bounds,
#     strategy="best1bin",
#     mutation=(0.5, 1),
#     recombination=0.7,
#     disp=True,
# )

# # 结果
# p_opt, N_opt = result.x
# N_opt = int(round(N_opt))  # Round N to the nearest integer
# print(f"Optimal p: {p_opt}")
# print(f"Optimal N: {N_opt}")
# print(f"Minimum value: {objective((p_opt, N_opt))}")

# s=set()
# s.add(("1",120,"123"))
# print(("1",120,"2123") in s)

from bracket_parameter_extraction import *

# Example usage
if __name__ == "__main__":
    labels = [
        # (" 45 ", "no annotation line"), (" text 120", "no annotation line"), ("120DH ", "top"), ("other 45DH", "bottom"),
        # ("  B150X10A ", "top"), ("  FB120X10   ", "bottom"), ("FL150", "bottom"),
        # (" BK01 extra ", "top"), ("R300", "no annotation line")
        ("100X12","bottom"),("   100X12","bottom"),(" B150  ~DH  ","top"),(" B150  ~DH  ","bottom")
    ]
    for label, position in labels:
        result = parse_elbow_plate(label, annotation_position=position)
        print(f"Parse Result ({position}):", result)
