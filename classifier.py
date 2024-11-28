import csv
from element import *

def load_classification_table(file_path):
    """
    Load the classification table from a CSV file.
    Returns a list of dictionaries with keys: type, cornerhole_nums, and free_edges.
    """
    classification_table = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert cornerhole_nums to integer and free_edges to a list
            row['cornerhole_nums'] = int(row['cornerhole_nums'])
            row['free_edges'] = row['free_edges'].split('|')
            classification_table.append(row)
    return classification_table

def poly_classifier(poly_refs, poly_cornerhole_edges, poly_free_edges):
    classification_file_path = "./type.csv"  # Path to the CSV file
    classification_table = load_classification_table(classification_file_path)

    conerhole_num = len(poly_cornerhole_edges)  # Count cornerholes
    free_edges_sequence = []
    for seg in poly_free_edges[0]:
        if isinstance(seg.ref, DLine):
            free_edges_sequence.append("line")
        elif isinstance(seg.ref, DArc):
            free_edges_sequence.append("arc")
    reversed_free_edges_sequence = free_edges_sequence[::-1]  # Reverse free edges

    # Match against the classification table
    matched_type = None
    for row in classification_table:
        if row['cornerhole_nums'] == conerhole_num and (
            row['free_edges'] == free_edges_sequence or row['free_edges'] == reversed_free_edges_sequence
        ):
            matched_type = row['type']
            break

    if matched_type is not None:
        result = matched_type
    else:
        result = "Unclassified"

    return result

# Example usage
# if __name__ == "__main__":
#     # Example inputs
#     poly_infos = ["poly1", "poly2", "poly3"]
#     poly_cornerhole_edges = [[1, 2], [1], [1, 2, 3]]
#     poly_free_edges = [["line", "arc"], ["arc"], ["arc", "line", "line"]]

#     # Run the classifier
#     classification_results = poly_classifier(poly_infos, poly_cornerhole_edges, poly_free_edges)

#     # Print results
#     for poly_name, poly_type in classification_results.items():
#         print(f"{poly_name}: {poly_type}")
