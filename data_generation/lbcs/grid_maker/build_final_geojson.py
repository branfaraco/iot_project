import hashlib
import os
import json
import h3
from pathlib import Path
from shapely.geometry import Polygon
from grid_maker.utilities.utilities import open_json_file
import math

def extract_code(codes_coverage, level=4):
    """
    Returns the most specific LBCS code with valid coverage.

    Starts from the deepest level (4) and moves up to 1
    until it finds coverage data for that level.

    Args:
        codes_coverage (dict): Mapping of LBCS codes to coverage weights.
        level (int): Starting level of specificity (default is 4).

    Returns:
        str: Best-matching LBCS code, or '0000' if none found.
    """
    if level < 1:
        return "0000"

    level_i_sum = {}
    for code, coverage in codes_coverage.items():
        level_i_prefix = code[:level]

        # Skip partial prefixes like '10', '20', etc.
        if "0" not in level_i_prefix:
            level_i_code = level_i_prefix + "0" * (4 - level)
            level_i_sum[level_i_code] = level_i_sum.get(
                level_i_code, 0) + coverage

    if level_i_sum:
        return max(level_i_sum, key=level_i_sum.get)
    else:
        return extract_code(codes_coverage, level - 1)


# --------------------- COLOR ----------------------------

LBCS_COLOR_HEX = {
    "1000": "#FFFF00",  # Yellow
    "2000": "#FF0000",  # Red
    "3000": "#A020F0",  # Purple
    "4000": "#BEBEBE",  # Gray
    "5000": "#90EE90",  # Light Green
    "6000": "#0000FF",  # Blue
    "7000": "#008B8B",  # Dark Cyan
    "8000": "#551A8B",  # Purple4
    "9000": "#228B22",  # Forest Green
    "0000": "#FFFFFF"   # White (default)
}


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def lighten_color(rgb, factor):
    return tuple(int(c + (255 - c) * factor) for c in rgb)


def vary_color(rgb, variation):
    """
    Slightly vary a color by a small amount (positive or negative).
    """
    return tuple(
        max(0, min(255, c + variation)) for c in rgb
    )

def extract_level(code):
    # Detect level (1–4)
    for i, c in enumerate(code):
        if c == "0":
            return max(1, i) 
    return len(code)

def extract_color(code, level):
    """
    Returns a color for a given LBCS code.
    - Level 1 → base color directly (without lightening or variation)
    - Level 2/3/4 → lighter + slightly varied based on code

    Args:
        code (str): 4-digit LBCS code

    Returns:
        tuple: RGB color tuple
    """
    if not code or len(code) != 4:
        return hex_to_rgb(LBCS_COLOR_HEX["0000"])

    base_code = code[0] + "000"
    base_color = LBCS_COLOR_HEX.get(base_code, "#FFFFFF")
    rgb = hex_to_rgb(base_color)

    if level == 1:
        # Si es nivel 1, usar el color base directamente
        return rgb

    # Aclarado según nivel
    lighten_factor = {2: 0.2, 3: 0.4, 4: 0.6}.get(level, 0.6)
    light_rgb = lighten_color(rgb, lighten_factor)

    # Variación pequeña según el código (consistente)
    hash_int = int(hashlib.md5(code.encode()).hexdigest(), 16)
    variation = (hash_int % 30) - 15  # entre -15 y +14
    varied_rgb = vary_color(light_rgb, variation)

    return varied_rgb

# -------------------------------------------------------


def build_final_geojson(grid_code_coverage, grid_heights, grid_geojson, header, level=1):
    """
    Converts a base grid GeoJSON into a complete annotated GeoJSON.

    For each cell:
    - Assigns its primary LBCS code (based on coverage).
    - Adds color and description metadata.
    - Adds height info (if available).
    - Updates the GeoJSON with general metadata including header and center coordinates.
    - Generates a "types" dictionary with one entry per unique LBCS type used.

    Args:
        grid_code_coverage (dict): Mapping of cell_id to LBCS code weights.
        grid_heights (dict): Mapping of cell_id to average height.
        grid_geojson (dict): GeoJSON with cell geometries and IDs.
        header (dict): Dictionary to be embedded in the GeoJSON's metadata.
        level (int): LBCS depth to extract the main code (default is 1).

    Returns:
        dict: A GeoJSON FeatureCollection enriched with cell metadata and type definitions.
    """

    # Initialize the global metadata
    grid_geojson["properties"] = {
        "header": header,
        "types": {}
    }

    lbcs_path = '../../config/lbcs_sort_desc.json'
    lbcs_sort_desc_dic = open_json_file(lbcs_path)

    type_id_counter = 0

    for cell_feature in grid_geojson["features"]:
        cell_id = cell_feature["properties"]["id"]

        codes = grid_code_coverage[(cell_id)]
        code = extract_code(codes, level)
        code_level = extract_level(code)
        color = extract_color(code, code_level)
        sort_desc = lbcs_sort_desc_dic.get(code, "No description available")
        height = grid_heights.get(cell_id, 0)

        # Update cell properties
        cell_feature["properties"].update({
            "color": color,
            "name": sort_desc,
            "height": [0,height, 50],
            "interactive": True

        })
        
        cell_feature["properties"].setdefault("other_properties", {}).update({
            "main_code": code,
            "lbcs_codes": codes
        })

        # Add to types if not already present
        if sort_desc not in grid_geojson["properties"]["types"]:
            grid_geojson["properties"]["types"][sort_desc] = {
                "name": sort_desc,
                "description": "Basic LBCS code. Refer to the LBCS standard for full details.",
                "color": rgb_to_hex(color),
                "height": [0, 50, 100],
                "LBCS": [{"proportion": 1, "use": {code: 1}}],
                "NAICS": [],
                "interactive": True,
                "id": type_id_counter,
                "editor_properties": {
                    "min_height": 0,
                    "max_height": math.ceil(height)  # << redondeo hacia arriba
                }
            }
            type_id_counter += 1
        else:
            # Actualizar el max_height si el nuevo height es mayor
            current_max = grid_geojson["properties"]["types"][sort_desc]["editor_properties"]["max_height"]
            if height > current_max:
                grid_geojson["properties"]["types"][sort_desc]["editor_properties"]["max_height"] = math.ceil(height)


    # Compute central coordinates for reference in header
    first_coord = grid_geojson["features"][0]["geometry"]["coordinates"][0][0]
    last_coord = grid_geojson["features"][-1]["geometry"]["coordinates"][0][0]
    avg_lat = (first_coord[1] + last_coord[1]) / 2
    avg_lon = (first_coord[0] + last_coord[0]) / 2

    grid_geojson["properties"]["header"]["latitude"] = avg_lat
    grid_geojson["properties"]["header"]["longitude"] = avg_lon

    # change to the GEOGRID and GEOGRIDDATA format

    return grid_geojson
