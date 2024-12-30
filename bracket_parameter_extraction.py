import re
def is_useful_text(content):
    label = content.strip()
    if label=="*":
        return True
    pattern_b = r"\s*(?:B)?(?P<primary>\d+)(?:X(?P<thickness>\d+))?\s*~?\s*(?P<material>A|DH)?"
    pattern_fb = r"\s*(?:FB)?(?P<section_height>\d+)(?:X(?P<fb_thickness>\d+))?"
    pattern_fl = r"\s*(?:Fl)?(?P<section_height>\d+)"
    pattern_bk = r"\s*BK(?P<bk_code>\d{2})"
    pattern_r = r"\s*R(?P<radius>\d+)"
    return re.fullmatch(pattern_b, label) or re.fullmatch(pattern_fb, label) or re.fullmatch(pattern_fl, label) or re.fullmatch(pattern_bk, label) or re.fullmatch(pattern_r, label)
    # match_annotation = re.search(r"[A-Z]*\d+[A-Z0-9X~]*", label)
    # if match_annotation:
    #     return True
    # else:
    #     if label=="*":
    #         return True
    #     return None
def parse_elbow_plate(label, annotation_position=None):
    """
    Parse elbow plate annotation parameters, including arm length, thickness, material, reinforcement edges, and other information.

    :param label: str, Annotation string of the elbow plate, e.g., "B150X10A", "FB120X10", "BK01", "R300"
    :param annotation_position: str, Annotation position: "top", "bottom", or "no annotation line"
    :return: dict, A dictionary containing parameters, e.g., {"Arm Length": 150, "Thickness": 10, "Material": "A"} or None if parsing fails.
    """
    # Strip unnecessary spaces and extract relevant portion
    label = label.strip()
    match_annotation = re.search(r"[A-Z]*\d+[A-Z0-9X~]*", label)
    if match_annotation:
        label = match_annotation.group(0)
    else:
        return None

    # Define regular expressions
    pattern_b = r"\s*(?:B)?(?P<primary>\d+)(?:X(?P<thickness>\d+))?\s*~?\s*(?P<material>A|DH)?"
    pattern_fb = r"\s*(?:FB)?(?P<section_height>\d+)(?:X(?P<fb_thickness>\d+))?"
    pattern_fl = r"\s*(?:Fl)?(?P<section_height>\d+)"
    pattern_bk = r"\s*BK(?P<bk_code>\d{2})"
    pattern_r = r"\s*R(?P<radius>\d+)"

    # Check different annotation types
    if annotation_position == "top":
        # Annotations at the top prioritize parsing as B, BK, or R type
        if match := re.fullmatch(pattern_b, label):
            primary = int(match.group("primary"))
            thickness = int(match.group("thickness")) if match.group("thickness") else None
            material = match.group("material") if match.group("material") else "A"

            arm_length = None
            if primary > 50:  # Assume values greater than 50 are arm lengths, and values less than or equal to 50 are thicknesses
                arm_length = primary
            else:
                thickness = primary if thickness is None else thickness

            if arm_length is not None and arm_length < 100:
                return None
            if thickness is not None and thickness > 50:
                return None

            return {
                "Type": "B",
                "Arm Length": arm_length,
                "Thickness": thickness,
                "Material": material,
            }

        elif match := re.fullmatch(pattern_bk, label):
            bk_code = match.group("bk_code")
            return {
                "Type": "BK",
                "Typical Section Code": bk_code,
            }

        elif match := re.fullmatch(pattern_r, label):
            radius = int(match.group("radius"))
            return {
                "Type": "R",
                "Radius": radius,
            }

    elif annotation_position == "bottom":
        # Annotations at the bottom prioritize parsing as FB, FL, or R type
        if match := re.fullmatch(pattern_fb, label):
            section_height = int(match.group("section_height"))
            fb_thickness = int(match.group("fb_thickness")) if match.group("fb_thickness") else None
            #print(section_height,fb_thickness)
            if section_height < 100:
                return None
            if fb_thickness is not None and fb_thickness > 50:
                return None

            return {
                "Type": "FB",
                "Section Height": section_height,
                "Thickness": fb_thickness,
            }

        elif match := re.fullmatch(pattern_fl, label):
            section_height = int(match.group("section_height"))

            if section_height < 100:
                return None

            return {
                "Type": "FL",
                "Section Height": section_height,
            }

        elif match := re.fullmatch(pattern_r, label):
            radius = int(match.group("radius"))
            return {
                "Type": "R",
                "Radius": radius,
            }

    elif annotation_position == "other":
        # Without an annotation line, parse as R type or a simple numerical value
        if match := re.fullmatch(pattern_r, label):
            radius = int(match.group("radius"))
            return {
                "Type": "R",
                "Radius": radius,
            }

        elif label.isdigit():
            value = int(label)
            return {
                "Type": "Numeric",
                "Value": value,
            }

    # If no type matches
    return None