import math
from collections import deque
import time

class DPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"DPoint({self.x}, {self.y})"

    def __eq__(self, other):
        return isinstance(other, DPoint) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

class DSegment:
    def __init__(self, start_point, end_point, ref=None):
        self.start_point = start_point
        self.end_point = end_point
        self.ref = ref

    def __repr__(self):
        return f"DSegment({self.start_point}, {self.end_point}, ref={self.ref})"

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle (in degrees) formed by three points p1, p2, and p3.
    The angle is measured from vector (p2 -> p1) to vector (p2 -> p3).
    """
    v1 = (p1.x - p2.x, p1.y - p2.y)
    v2 = (p3.x - p2.x, p3.y - p2.y)
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    angle = math.atan2(det, dot_product)  # Angle in radians
    angle_deg = math.degrees(angle)

    # Ensure the angle is in the range [0, 360]
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg

def dfs_paths_with_repline(graph, repline, max_length, timeout=5):
    """
    Perform DFS to find a closed loop starting at repline.start_point,
    with the first segment being repline and ending back at repline.start_point.
    Prioritize edges with the largest counterclockwise angle change within [10, 170] degrees.
    """
    start_time = time.time()
    start_point = repline.start_point
    stack = [(repline.end_point, [start_point, repline.end_point], [repline])]  # Start with repline
    visited_edges = {(repline.start_point, repline.end_point)}  # Mark the first edge as visited

    while stack:
        current_time = time.time()
        if (current_time - start_time) >= timeout:
            print(f'{start_point} search timed out')
            break

        current_point, point_path, seg_path = stack.pop()

        if len(seg_path) >= max_length:
            continue

        if current_point == start_point and len(seg_path) > 1:
            return seg_path  # Return the first valid closed loop

        # Get neighbors and sort them by counterclockwise angle change
        neighbors = []
        for neighbor, ref in graph.get(current_point, []):
            edge = (current_point, neighbor)
            if edge not in visited_edges:
                neighbors.append((neighbor, ref))

        # Sort neighbors by the counterclockwise angle change
        if len(point_path) > 1:
            prev_point = point_path[-2]
            neighbors.sort(key=lambda nbr: calculate_angle(prev_point, current_point, nbr[0]), reverse=True)

        for neighbor, ref in neighbors:
            angle = calculate_angle(point_path[-2], current_point, neighbor) if len(point_path) > 1 else 0
            if len(point_path) == 2 or 10 <= angle <= 170:  # Allow all angles for the first segment
                edge = (current_point, neighbor)
                visited_edges.add(edge)

                new_point_path = point_path + [neighbor]
                new_seg = DSegment(current_point, neighbor, ref)
                new_seg_path = seg_path + [new_seg]

                stack.append((neighbor, new_point_path, new_seg_path))

    return []  # Return an empty path if no closed loop is found

def process_repline_with_repline(repline, graph, segmentation_config):
    """
    Process a single repline using the DFS algorithm to find a closed loop
    starting and ending at repline.start_point.
    """
    path = dfs_paths_with_repline(graph, repline, segmentation_config.path_max_length, segmentation_config.timeout)
    return path

def build_graph(segments):
    """
    Build a graph from segments where each node is a point, and edges are defined by segments.
    """
    graph = {}

    for seg in segments:
        p1, p2 = seg.start_point, seg.end_point

        if p1 not in graph:
            graph[p1] = []
        if p2 not in graph:
            graph[p2] = []

        graph[p1].append((p2, seg.ref))
        graph[p2].append((p1, seg.ref))

    return graph

# Example usage
segments = [
    DSegment(DPoint(0, 0), DPoint(1, 0), ref="A"),
    DSegment(DPoint(1, 0), DPoint(1, 1), ref="B"),
    DSegment(DPoint(1, 1), DPoint(0, 1), ref="C"),
    DSegment(DPoint(0, 1), DPoint(0, 0), ref="D")
]

# Build graph
graph = build_graph(segments)

# Use DFS to find a path
repline = DSegment(DPoint(0, 0), DPoint(1, 0), ref="A")
segmentation_config = type("SegConfig", (), {"path_max_length": 10, "timeout": 5})
path = process_repline_with_repline(repline, graph, segmentation_config)

# Output path
print("Found path:", path)
