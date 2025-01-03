import re

from bracket_parameter_extraction import *
# Example usage
if __name__ == "__main__":
    labels = [
        # (" 45 ", "no annotation line", False), (" text 120", "no annotation line", False), ("120%DH ", "top", False), ("other 45DH", "bottom", False),
        # ("  B100X10CH ", "top", False), ("  FB120X10   ", "bottom", False), ("FL150", "bottom", False),
        # (" BK01 extra ", "top", False), ("R300", "no annotation line", False), ("~DH120", "top", False),
        # ("FB150X10", "bottom", True), ("FL200", "bottom", False),
        # (" FB150X12 ~ DH ", "bottom", False),
        # ("   AC ", "bottom", False)
        ("20.0", "top", False),("$~ F123", "bottom", False),("100.8X20.6 &~ AH", "bottom", True),("  100.0456 ", "other", False)
    ]
    for label, position, is_fb_flag in labels:
        result = parse_elbow_plate(label, annotation_position=position, is_fb=is_fb_flag)
        print(f"Parse Result ({position}, is_fb={is_fb_flag}):", result)
        print(is_useful_text(label))
