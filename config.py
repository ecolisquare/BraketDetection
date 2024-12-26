class SegmentationConfig:
    def __init__(self):
        self.verbose=True

        self.json_path=""
        
        self.line_expand_length=13
        self.arc_expand_length=8
        self.line_image_path="./output/line.png"
        self.draw_intersections=False
        self.draw_segments=True
        self.line_image_drawPolys=True

        self.draw_poly_nums=1000
        self.poly_image_dir="./output"
        self.draw_polys=False
        self.draw_geometry=False

        self.segment_filter_length=0.25
        self.segment_filter_iters=100

        self.segment_split_epsilon=0.25

        self.intersection_epsilon=1e-9

        self.bbox_area=3000
        self.bbox_ratio=6


        self.remove_tolerance=1e-5


        self.eps=20
        self.min_samples=1

        self.path_max_length = 20
        self.path_min_length = 3

        self.poly_info_dir = "./output"
        self.res_image_path = "./output/res.png"

        self.max_workers=-1
        self.timeout=2

        self.type_path = "./type.json"

        self.dxf_output_folder = "./output/"