"""
Main function to generate a complete H3 or square grid with integrated OSM-based land use data.

NOTE: This version is incomplete and still under development.
"""

import h3
import os
import overpass
import requests
import random
import json
import folium
from pyproj import Proj, transform

from grid_maker.query_osm import query_osm, map_osm_lbcs
from grid_maker.coverage_figures import calculate_cells_code_coverage_and_height
from grid_maker.tf_idf import calculate_tfidf
from grid_maker.build_final_geojson import build_final_geojson
from grid_maker.create_grid_geojson import create_grid_geojson
from grid_maker.construct_header import construct_header
from grid_maker.utilities.utilities import open_json_file


def grid_maker(grid_corners_latlon, gridType="h3", cellSize=10, gridRotation=0, regularGrid=False, lbcsDepth=1):
    """
    Generates a grid (H3 or square) over a given polygon and integrates OSM-derived land use data.

    Args:
        grid_corners_latlon (list): List of [lat, lon] coordinates defining the boundary polygon.
        gridType (str): Either "h3" or "squares". Defaults to "h3".
        cellSize (int): Resolution (if H3) or cell side length in meters (if squares). Defaults to 10.
        gridRotation (int): Rotation angle (for square grid only). Defaults to 0.
        regularGrid (bool): If True, keeps full grid (even partially outside); else filters strictly inside. Defaults to False.
        lbcsDepth (int): Level of detail in the LBCS hierarchy. Defaults to 1.

    Returns:
        dict: GeoJSON FeatureCollection with properties including LBCS TF-IDF weights and heights.
    """

    # 1. Create the base grid (H3 or square)
    geojson_grid = create_grid_geojson(
        grid_corners_latlon, gridType, cellSize, gridRotation, regularGrid
    )
    

    # 2. Download OSM data for the grid area
    # NOTE: Currently using cellSize for both H3 resolution and bounding box expansion — this should be improved.
    geojson_data_raw = query_osm(grid_corners_latlon, gridType, cellSize)

    # 3. Map raw OSM tags to LBCS codes
    mapping = open_json_file('../../config/mapping.json')
    geojson_data = map_osm_lbcs(geojson_data_raw, mapping)

    # 4. Calculate LBCS coverage and building heights per cell
    grid_code_coverage, grid_heights = calculate_cells_code_coverage_and_height(
        geojson_grid, geojson_data
    )
    print("[DEV] CODE COVERAGE", random.choice(
        list(grid_code_coverage.items())))
    # 5. Apply TF-IDF weighting to the LBCS code distribution
    grid_code_coverage_tfidf = calculate_tfidf(grid_code_coverage)

    print("[DEV] CODE COVERAGE TFIDF", random.choice(
        list(grid_code_coverage_tfidf.items())))

    header = construct_header(geojson_grid, gridType, cellSize, gridRotation, regularGrid, len(
        grid_code_coverage), lbcsDepth)

    # 6. Build the final GeoJSON with coverage + height + TF-IDF
    grid_geojson = build_final_geojson(
        grid_code_coverage_tfidf, grid_heights, geojson_grid, header, level=lbcsDepth)

    return grid_geojson
