class SegmentationConfig:
    def __init__(self):
        self.verbose=True

        self.json_path=""
        
        self.line_expand_length=10

        self.line_image_path="./output/line.png"
        self.draw_intersections=False
        self.draw_segments=True
        self.line_image_drawPolys=True

        self.draw_poly_nums=1000
        self.poly_image_dir="./output"
        self.draw_polys=True
        self.draw_geometry=True

        self.segment_filter_length=12
        self.segment_filter_iters=25

        self.segment_split_epsilon=0.25

        self.intersection_epsilon=1e-9

        self.bbox_area=3000
        self.bbox_ratio=3


        self.remove_tolerance=1e-5


        self.eps=25.0
        self.min_samples=1

        self.path_max_length = 15
        self.path_min_length = 3

        self.poly_info_dir = "./output"
