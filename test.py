from element import *
import networkx as nx
edges=[]
edges.append(DSegment(DPoint(0,0),DPoint(1,0),None))
edges.append(DSegment(DPoint(0,0),DPoint(0,1),None))
edges.append(DSegment(DPoint(0,0),DPoint(0.5,0.5),DLine()))
edges.append(DSegment(DPoint(0,1),DPoint(0.5,0.5),DLine()))
edges.append(DSegment(DPoint(1,0),DPoint(0.5,0.5),DLine()))
# 创建无向图并添加节点和边
G = nx.Graph()
# 添加一些边，形成多个闭合路径
G.add_edges_from(edges)
# 查找所有的基本环
cycles = nx.cycle_basis(G)
print(cycles)
