class SegmentationConfig:
    def __init__(self):
        self.verbose=True

        self.json_path="/home/user10/code/BraketDetection/data/split/FR18-3.json"
        
        self.line_expand_length=10

        self.line_image_path="/home/user10/code/BraketDetection/output/line.png"
        self.draw_intersections=False
        self.draw_segments=True

        self.draw_poly_nums=1000
        self.poly_image_dir="/home/user10/code/BraketDetection/output"
        self.draw_polys=True
        self.draw_geometry=True

        self.segment_filter_length=12
        self.segment_filter_iters=10

        self.segment_split_epsilon=0.25

        self.intersection_epsilon=1e-9

        self.bbox_area=3000
        self.bbox_ratio=6


        self.remove_tolerance=1e-5
