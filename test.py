from element import *
edge1=DSegment(DPoint(1.001,1.002),DPoint(2.003,2.102),None)
edge2=DSegment(DPoint(1.000,1.053),DPoint(2.014,2.135),DLine())
print(edge1==edge2)
print(hash(edge1)==hash(edge2))
edge_set=set()
edge_set.add(edge1)
print(edge2 in edge_set)