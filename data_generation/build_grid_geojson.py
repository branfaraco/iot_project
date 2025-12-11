# build_grid_geojson.py
import os
import json
import math
from typing import List, Dict, Any

import numpy as np
from shapely.geometry import Polygon, mapping
from pyrosm import OSM

from grid_maker.coverage_figures import calculate_cells_code_coverage_and_height
from grid_maker.tf_idf import calculate_tfidf
from grid_maker.query_osm import map_osm_lbcs


def make_rect_grid_from_two_points(p1, p2, rows: int, cols: int) -> Dict[str, Any]:
    lat1, lon1 = p1
    lat2, lon2 = p2

    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)

    lat_edges = np.linspace(lat_max, lat_min, rows + 1)
    lon_edges = np.linspace(lon_min, lon_max, cols + 1)

    features = []
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
                    "geometry": mapping(poly),
                }
            )
            cell_id += 1

    return {"type": "FeatureCollection", "features": features}


def gdf_to_features(gdf, id_prefix: str) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    if gdf is None or gdf.empty:
        return features

    cols = [c for c in gdf.columns if c != "geometry"]

    for i, row in gdf.iterrows():
        geom_obj = row.geometry
        if geom_obj is None or geom_obj.is_empty:
            continue

        tags = {}
        for c in cols:
            val = row[c]
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                tags[c] = val

        if geom_obj.geom_type == "Polygon":
            coords = [list(geom_obj.exterior.coords)]
            geom_dict = {"type": "Polygon", "coordinates": coords}
        elif geom_obj.geom_type == "MultiPolygon":
            coords = [list(poly.exterior.coords) for poly in geom_obj.geoms]
            geom_dict = {"type": "MultiPolygon", "coordinates": coords}
        elif geom_obj.geom_type == "LineString":
            coords = list(geom_obj.coords)
            geom_dict = {"type": "LineString", "coordinates": coords}
        elif geom_obj.geom_type == "MultiLineString":
            coords = [list(ls.coords) for ls in geom_obj.geoms]
            geom_dict = {"type": "MultiLineString", "coordinates": coords}
        elif geom_obj.geom_type == "Point":
            geom_dict = {"type": "Point", "coordinates": (geom_obj.x, geom_obj.y)}
        else:
            continue

        features.append(
            {
                "type": "Feature",
                "geometry": geom_dict,
                "properties": {"tags": tags, "id": f"{id_prefix}_{i}"},
            }
        )

    return features


def query_osm_local_berlin(osm: OSM) -> Dict[str, Any]:
    buildings = osm.get_buildings()
    landuse = osm.get_landuse()
    pois = osm.get_pois()
    roads = osm.get_network(network_type="all")

    features: List[Dict[str, Any]] = []
    features += gdf_to_features(buildings, "bldg")
    features += gdf_to_features(landuse, "landuse")
    features += gdf_to_features(pois, "poi")
    features += gdf_to_features(roads, "road")

    return {"type": "FeatureCollection", "features": features}


def run_lbcs_pipeline(grid_geojson, osm, mapping_path):
    """
    This is a really delicate process so if it fail in some point we save the intermediate result
    """
    # paths
    root = r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face\BERLIN_reduced\cache_lbcs"
    raw_path   = os.path.join(root, "osm_raw.json")
    lbcs_path  = os.path.join(root, "lbcs_mapped.json")
    cov_path   = os.path.join(root, "lbcs_coverage.json")
    height_path = os.path.join(root, "cell_heights.json")

    # step 1: raw OSM
    if os.path.exists(raw_path):
        geojson_data_raw = json.load(open(raw_path))
    else:
        geojson_data_raw = query_osm_local_berlin(osm)
        json.dump(geojson_data_raw, open(raw_path,"w"))

    # step 2: LBCS mapping
    if os.path.exists(lbcs_path):
        geojson_data = json.load(open(lbcs_path))
    else:
        mapping = json.load(open(mapping_path))
        geojson_data = map_osm_lbcs(geojson_data_raw, mapping)
        json.dump(geojson_data, open(lbcs_path,"w"))

    # step 3: cell coverage + height
    if os.path.exists(cov_path):
        grid_coverage = json.load(open(cov_path))
        grid_heights = json.load(open(height_path))
    else:
        grid_coverage, grid_heights = calculate_cells_code_coverage_and_height(grid_geojson, geojson_data)
        json.dump(grid_coverage, open(cov_path,"w"))
        json.dump(grid_heights, open(height_path,"w"))

    # step 4: TF-IDF
    grid_code_coverage_tfidf = calculate_tfidf(grid_coverage)

    return grid_code_coverage_tfidf



def build_grid() -> Dict[str, Any]:
    # original grid
    H, W = 495, 436
    min_lon, max_lon = 13.189, 13.625
    min_lat, max_lat = 52.359, 52.854

    lon_step = (max_lon - min_lon) / W
    lat_step = (max_lat - min_lat) / H

    # subgrid
    r0, r1 = 10, 274
    c0, c1 = 0, 432

    R_H = r1 - r0
    R_W = c1 - c0

    sub_min_lat = min_lat + r0 * lat_step
    sub_max_lat = min_lat + r1 * lat_step
    sub_min_lon = min_lon + c0 * lon_step
    sub_max_lon = min_lon + c1 * lon_step

    # reduction to 300 m cells (factor 3)
    G_H = R_H // 3
    G_W = R_W // 3

    p1 = (sub_min_lat, sub_min_lon)
    p2 = (sub_max_lat, sub_max_lon)

    return make_rect_grid_from_two_points(p1, p2, G_H, G_W)


def compute_grid_bbox(grid_geojson: Dict[str, Any]) -> tuple[float, float, float, float]:
    coords = np.array(
        [
            [lat, lon]
            for f in grid_geojson["features"]
            for lon, lat in f["geometry"]["coordinates"][0]
        ]
    )
    min_lat, min_lon = coords.min(axis=0)
    max_lat, max_lon = coords.max(axis=0)
    # pyrosm bounding_box = (minx, miny, maxx, maxy) = (lon_min, lat_min, lon_max, lat_max)
    return (min_lon, min_lat, max_lon, max_lat)


def main() -> None:
    data_root = r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face"
    meta_dir = os.path.join(data_root, "BERLIN_reduced")

    berlin_pbf_path = os.path.join(data_root, "berlin-latest.osm.pbf")
    mapping_path = r"C:\Users\user\UPM\Imperial-4año\IoT\Github\data_generation\grid_maker\config\mapping.json"
    out_path = os.path.join(meta_dir, "grid_geojson.json")

    # ------------------------------------------------------------
    # 1. Load metadata (rows, cols, bbox)
    # ------------------------------------------------------------
    meta_path = os.path.join(meta_dir, "metadata.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = meta["rows"]
    cols = meta["cols"]

    # bbox stored as [[min_lat, min_lon], [max_lat, max_lon]]
    (min_lat, min_lon), (max_lat, max_lon) = meta["bbox"]


    # ------------------------------------------------------------
    # 2. Build grid using metadata
    # ------------------------------------------------------------
    p1 = (min_lat, min_lon)
    p2 = (max_lat, max_lon)

    grid = make_rect_grid_from_two_points(p1, p2, rows, cols)

    # ------------------------------------------------------------
    # 3. Load OSM data inside bounding box
    # ------------------------------------------------------------
    bbox = [min_lon, min_lat, max_lon, max_lat]
    osm = OSM(berlin_pbf_path, bounding_box=bbox)

    # ------------------------------------------------------------
    # 4. LBCS pipeline
    # ------------------------------------------------------------
    lbcs = run_lbcs_pipeline(grid, osm, mapping_path)

    # attach tf-idf scores
    for feat in grid["features"]:
        cid = feat["properties"]["id"]
        feat["properties"]["lbcs_tfidf"] = lbcs.get(cid, {})

    # ------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)


if __name__ == "__main__":
    main()
