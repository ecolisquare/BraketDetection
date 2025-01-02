import re
def is_useful_text(content=""):
    label = content.strip()
    if label==None:
        return False
    if label=="*" or label.isdigit():
        return True
    patterns=[
        r"(?:B)?(?P<primary>\d+)(?:X(?P<thickness>\d+))?\s*(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?",
        r"(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?",
        r"(?:FB)?(?P<primary>\d+)(?:X(?P<fb_thickness>\d+))?\s*(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?",
        r"(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?",
        r"(?:FL)?(?P<section_height>\d+)",
        r"BK(?P<bk_code>\d{2})",
        r"R(?P<radius>\d+)",
    ]
    flag=False
    for pattern in patterns:
        if re.fullmatch(pattern, label):
            flag=True
        if flag:
            break
    return flag
def parse_elbow_plate(label="", annotation_position="other", is_fb=False):
    """
    Parse elbow plate annotation parameters, including arm length, thickness, material, reinforcement edges, and other information.

    :param label: str, Annotation string of the elbow plate, e.g., "B150X10A", "FB120X10", "BK01", "R300"
    :param annotation_position: str, Annotation position: "top", "bottom", or "no annotation line"
    :param is_fb: bool, Force interpretation as FB type when true.
    :return: dict, A dictionary containing parameters, e.g., {"Arm Length": 150, "Thickness": 10, "Material": "A"} or None if parsing fails.
    """
    # Strip unnecessary spaces
    label = label.strip()
    if label is None or label=="":
        return None
    # Define regular expressions
    pattern_b = r"(?:B)?(?P<primary>\d+)(?:X(?P<thickness>\d+))?\s*(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?"
    pattern_b_op=r"(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?"
    pattern_fb = r"(?:FB)?(?P<primary>\d+)(?:X(?P<fb_thickness>\d+))?\s*(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?"
    pattern_fb_op=r"(?:[~%]*?)\s*(?P<material>A|[A-Z]{2,3})?"
    pattern_fl = r"(?:FL)?(?P<section_height>\d+)"
    pattern_bk = r"BK(?P<bk_code>\d{2})"
    pattern_r = r"R(?P<radius>\d+)"

    # Check different annotation types
    if annotation_position == "top":
        #print(label)
        # Annotations at the top prioritize parsing as B, BK, or R type
        if match := re.fullmatch(pattern_b, label):

            primary = int(match.group("primary"))
            thickness = int(match.group("thickness")) if match.group("thickness") else None
            material = match.group("material") if match.group("material") else "A"

            arm_length = None
            if primary >= 100:  # Assume values >= 100 are arm lengths
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
        elif match := re.fullmatch(pattern_b_op, label):
            material = match.group("material") if match.group("material") else None
            if material is not None:
                return {
                    "Type": "B",
                    "Arm Length": None,
                    "Thickness": None,
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
        if is_fb:
            if match := re.fullmatch(pattern_fb, label):
                primary = int(match.group("primary"))
                fb_thickness = int(match.group("fb_thickness")) if match.group("fb_thickness") else None
                material = match.group("material") if match.group("material") else "A"


                section_height = None
                if primary >= 100:  # Assume values >= 100 are arm lengths
                    section_height = primary
                else:
                    fb_thickness = primary if fb_thickness is None else fb_thickness

                if section_height is not None and section_height < 100:
                    return None
                if fb_thickness is not None and fb_thickness > 50:
                    return None
            

                return {
                    "Type": "FB",
                    "Section Height": section_height,
                    "Thickness": fb_thickness,
                    "Material": material,
                }
            elif match := re.fullmatch(pattern_fb_op, label):
                material = match.group("material") if match.group("material") else None
                if material is not None:
                    return {
                        "Type": "FB",
                        "Section Height": None,
                        "Thickness": None,
                        "Material": material,
                    }


        else:
            if match := re.fullmatch(pattern_fl, label):
                section_height = int(match.group("section_height"))

                if section_height < 100:
                    return None

                return {
                    "Type": "FL",
                    "Section Height": section_height,
                }

            elif match := re.fullmatch(pattern_fb, label):
                primary = int(match.group("primary"))
                fb_thickness = int(match.group("fb_thickness")) if match.group("fb_thickness") else None
                material = match.group("material") if match.group("material") else "A"


                section_height = None
                if primary >= 100:  # Assume values >= 100 are arm lengths
                    section_height = primary
                else:
                    fb_thickness = primary if fb_thickness is None else fb_thickness

                if section_height is not None and section_height < 100:
                    return None
                if fb_thickness is not None and fb_thickness > 50:
                    return None
            

                return {
                    "Type": "FB",
                    "Section Height": section_height,
                    "Thickness": fb_thickness,
                    "Material": material,
                }
            elif match := re.fullmatch(pattern_fb_op, label):
                material = match.group("material") if match.group("material") else None
                if material is not None:
                    return {
                        "Type": "FB",
                        "Section Height": None,
                        "Thickness": None,
                        "Material": material,
                    }


        if match := re.fullmatch(pattern_r, label):
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