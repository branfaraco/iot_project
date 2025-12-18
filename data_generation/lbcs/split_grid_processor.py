

import os
import json
import hashlib
from typing import List, Tuple, Dict, Any

import numpy as np
from shapely.geometry import Polygon

from grid_maker.query_osm import query_osm, map_osm_lbcs
from grid_maker.coverage_figures import calculate_cells_code_coverage
from grid_maker.tf_idf import calculate_tfidf


def make_rect_grid_from_two_points(p1: Tuple[float, float], p2: Tuple[float, float], rows: int, cols: int) -> Dict[str, Any]:
    """Construct a rectangular grid between two points.

    Parameters
    ----------
    p1, p2 : Tuple[float, float]
        (lat, lon) coordinates of two opposite corners of the bounding box.
    rows, cols : int
        Number of rows and columns in the grid.

    Returns
    -------
    dict
        GeoJSON feature collection where each feature represents a cell.
    """
    lat1, lon1 = p1
    lat2, lon2 = p2
    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
    lat_edges = np.linspace(lat_max, lat_min, rows + 1)
    lon_edges = np.linspace(lon_min, lon_max, cols + 1)
    features: List[Dict[str, Any]] = []
    cell_id = 0
    for r in range(rows):
        for c in range(cols):
            lat_top = lat_edges[r]
            lat_bottom = lat_edges[r + 1]
            lon_left = lon_edges[c]
            lon_right = lon_edges[c + 1]
            poly = Polygon(
                [
                    (lon_left, lat_top),
                    (lon_right, lat_top),
                    (lon_right, lat_bottom),
                    (lon_left, lat_bottom),
                ]
            )
            features.append(
                {
                    "type": "Feature",
                    "properties": {"id": cell_id},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [list(poly.exterior.coords)],
                    },
                }
            )
            cell_id += 1
    return {"type": "FeatureCollection", "features": features}


def _bbox_to_key(bbox: Tuple[float, float, float, float]) -> str:
    """Create a deterministic filename component from a bounding box.

    Converts a tuple (min_lat, min_lon, max_lat, max_lon) into a short
    hexadecimal string using SHA1 hashing to avoid filenames with
    illegal characters.
    """
    m = hashlib.sha1()
    # Use high precision to avoid collisions for nearby boxes
    s = f"{bbox[0]:.6f},{bbox[1]:.6f},{bbox[2]:.6f},{bbox[3]:.6f}"
    m.update(s.encode("utf-8"))
    return m.hexdigest()[:12]


def _bbox_to_corners(bbox: Tuple[float, float, float, float]) -> List[List[float]]:
    """Convert (min_lat, min_lon, max_lat, max_lon) to corner coordinates.

    Returns corners in clockwise order starting from bottom‑left.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    return [
        [min_lat, min_lon],
        [max_lat, min_lon],
        [max_lat, max_lon],
        [min_lat, max_lon],
    ]


def _fetch_and_map(bbox: Tuple[float, float, float, float], grid_type: str, cell_size: int, mapping: Dict[str, List[str]], cache_dir: str, level: int) -> Dict[str, Any]:
    """Perform a single Overpass query and map tags to LBCS codes.

    This helper wraps `query_osm` and `map_osm_lbcs` with caching. If
    the mapped data for this bounding box already exists in
    `cache_dir`, it is loaded instead of making a network call.

    """
    os.makedirs(cache_dir, exist_ok=True)
    key = _bbox_to_key(bbox)
    raw_path = os.path.join(cache_dir, f"raw_{level}_{key}.json")
    mapped_path = os.path.join(cache_dir, f"lbcs_{level}_{key}.json")
    # If mapped data exists, load and return
    if os.path.exists(mapped_path):
        with open(mapped_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Otherwise, attempt to fetch raw OSM data
    corners = _bbox_to_corners(bbox)
    try:
        data_raw = query_osm(corners, grid_type, cell_size)
    except Exception as e:
        raise e
    # Save raw data to cache
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data_raw, f)
    # Map tags to LBCS codes
    data_mapped = map_osm_lbcs(data_raw, mapping)
    # Save mapped data
    with open(mapped_path, "w", encoding="utf-8") as f:
        json.dump(data_mapped, f)
    return data_mapped


def _split_bbox(bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
    """Divide a bounding box into four equal quadrants.

    Returns a list of four bounding boxes in the order: bottom‑left,
    bottom‑right, top‑left, top‑right.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    mid_lat = (min_lat + max_lat) / 2.0
    mid_lon = (min_lon + max_lon) / 2.0
    return [
        (min_lat, min_lon, mid_lat, mid_lon),  # bottom‑left
        (min_lat, mid_lon, mid_lat, max_lon),  # bottom‑right
        (mid_lat, min_lon, max_lat, mid_lon),  # top‑left
        (mid_lat, mid_lon, max_lat, max_lon),  # top‑right
    ]


def _safe_query_recursive(bbox: Tuple[float, float, float, float], grid_type: str, cell_size: int, mapping: Dict[str, List[str]], cache_dir: str, level: int, max_recursion: int, attempt: int = 0) -> List[Dict[str, Any]]:
    """Recursively query Overpass for a bounding box, subdividing on errors.

    Attempts to fetch and map data for `bbox`. If the request fails
    (e.g. due to a timeout, server error, or other exception), the
    bounding box is subdivided and each quadrant is processed
    recursively. Results are concatenated and returned as a list of
    features.
    """
    try:
        geojson = _fetch_and_map(
            bbox, grid_type, cell_size, mapping, cache_dir, level)
        return geojson.get("features", [])
    except Exception:
        # If maximum recursion depth reached, re‑raise the exception
        if level >= max_recursion:
            raise
        # Otherwise, subdivide and recurse
        subboxes = _split_bbox(bbox)
        all_features: List[Dict[str, Any]] = []
        for sub_bbox in subboxes:
            features = _safe_query_recursive(
                sub_bbox, grid_type, cell_size, mapping, cache_dir, level + 1, max_recursion, attempt + 1)
            all_features.extend(features)
        return all_features


def run_split_processing_first_level(
    metadata: Dict[str, Any],
    lbcs_scaling,
    mapping: Dict[str, List[str]],
    cell_size: int = 0,
    cache_dir: str = "split_cache",
    max_recursion: int = 3,
    lbcs_depth: int = 1,
) -> Tuple[Dict[str, Any], List[List[str]]]:
    """Run the split‑grid processing pipeline with preloaded metadata and mapping and return a simplified grid.

    This helper is similar to :func:`run_split_processing_first_level` but accepts
    already parsed metadata and mapping dictionaries instead of file paths.
    It strips away intermediate attributes and attaches only the first‑level
    LBCS code for each cell, returning both the grid and a matrix of these codes.
    """
    rows = int(metadata["rows"] / lbcs_scaling)
    cols = int(metadata["cols"] / lbcs_scaling)

    (min_lat, min_lon), (max_lat, max_lon) = metadata["bbox"]
    # Compute overall bounding box
    full_bbox = (min_lat, min_lon, max_lat, max_lon)
    # Recursively fetch and map features (always squares)
    features = _safe_query_recursive(
        full_bbox, "squares", cell_size, mapping, cache_dir, level=0, max_recursion=max_recursion)
    # Combine features into a single GeoJSON
    combined_geojson = {"type": "FeatureCollection", "features": features}
    # Build grid geometry
    grid_geojson = make_rect_grid_from_two_points(
        (min_lat, min_lon), (max_lat, max_lon), rows, cols)
    # Compute coverage and heights (heights unused)
    coverage = calculate_cells_code_coverage(grid_geojson, combined_geojson)
    # Apply TF‑IDF weighting
    tfidf_dict = calculate_tfidf(coverage)
    # Attach only the first‑level code to each feature and build matrix
    matrix: List[List[str]] = [
        [None for _ in range(cols)] for _ in range(rows)]
    for feature in grid_geojson["features"]:
        cell_id = feature["properties"]["id"]
        r, c = divmod(cell_id, cols)
        codes = tfidf_dict.get(cell_id, {})
        if codes:
            top_code = max(codes.items(), key=lambda x: x[1])[0]
            top_str = str(top_code)
            first_level = top_str[0] + "000"
            feature["properties"] = {"lbcs_first_level": first_level}
            matrix[r][c] = first_level
        else:
            feature["properties"] = {"lbcs_first_level": "9000"}
            matrix[r][c] = "9000"
    return grid_geojson, matrix
