"""
Calculates the coverage of each grid cell based on tagged OSM geometries.
"""

import json
from geopy.distance import geodesic
from shapely.geometry import Polygon, shape
from shapely.ops import unary_union
from rtree import index
import h3
import random


def normalize(grid_coverage):
    """
    Normalizes the coverage percentages so that the total per cell is 1.0.
    """
    for cell_id in grid_coverage:
        total_coverage = sum(grid_coverage[cell_id].values())
        if total_coverage > 0:
            for tag in grid_coverage[cell_id]:
                grid_coverage[cell_id][tag] /= total_coverage
    return grid_coverage


def point_hexagon(lon, lat, radius):
    """
    Converts a point to a hexagonal polygon with a given radius in meters.
    """
    hexagon_coords = []
    for i in range(6):
        angle = i * 60
        destination = geodesic(kilometers=(radius / 1000)).destination((lat, lon), angle)
        hexagon_coords.append([destination.longitude, destination.latitude])
    return hexagon_coords


def create_rtree_index(grid_polygons):
    """
    Creates an R-tree index for efficient spatial querying of grid polygons.
    """
    idx = index.Index()
    for id in grid_polygons:
        cell = grid_polygons[id]
        bbox = cell.bounds
        idx.insert(id, bbox)
    return idx


def find_intersections_rtree(idx, geo_shape, grid_polygons):
    """
    Finds all cells that intersect with a given shape using the R-tree index.
    """
    possible_cells = list(idx.intersection(geo_shape.bounds))
    intersections = []
    for id in possible_cells:
        cell = grid_polygons[id]
        if geo_shape.intersects(cell):
            intersections.append(id)
    return intersections


def calculate_coverage_percentage(cell, geo_shape):
    """
    Calculates the fraction of the cell area covered by the given shape.
    """
    safe_cell = cell.buffer(0)
    safe_geo = geo_shape.buffer(0)

    try:
        intersection = safe_geo.intersection(safe_cell)
        if not intersection.is_empty:
            return intersection.area / safe_cell.area
    except Exception:
        # Intersection failed due to invalid geometry
        return 0

    return 0


def calculate_cells_code_coverage(grid_geojson, geojson_shapes):
    """
    Assigns coverage percentages to grid cells from geojson shapes.

    Returns:
        tuple: (grid_coverage: dict[tag -> percentage]
    """



    grid_features = grid_geojson["features"]
    geojson_features = geojson_shapes["features"]
    # DEV
    if not grid_features:
        print("[DEV] GEOJSON LEN: grid_features está vacío")
        raise ValueError("El GeoJSON no contiene ninguna característica")

    else:
        print(f"[DEV] GEOJSON LEN: {len(grid_features)}")

    grid_polygons = {
        cell_geojson['properties']['id']: Polygon(cell_geojson['geometry']['coordinates'][0])
        for cell_geojson in grid_features
    }

    idx = create_rtree_index(grid_polygons)
    grid_coverage = {cell_id: {} for cell_id in grid_polygons.keys()}
    

    for geojson in geojson_features:
        geojson_geometry = geojson['geometry']
        geo_shape = None

        if geojson_geometry['type'] == 'Point':
            point_lon, point_lat = geojson_geometry['coordinates']
            hex_coords = point_hexagon(point_lon, point_lat, 10)
            geo_shape = Polygon(hex_coords)

        elif geojson_geometry['type'] == 'LineString':
            line = shape(geojson['geometry'])
            geo_shape = line.buffer(0.000025, cap_style=2)

        else:
            geo_shape = shape(geojson['geometry'])

        intersections = find_intersections_rtree(idx, geo_shape, grid_polygons)

        for id in intersections:
            cell = grid_polygons[id]
            percentage = calculate_coverage_percentage(cell, geo_shape)
            if percentage > 0:
                for tag in geojson['properties']['lbcs_codes']:
                    grid_coverage[id][tag] = grid_coverage[id].get(tag, 0) + percentage

                tags = geojson['properties']['tags']
                
                
    
    k = random.choice(list(grid_coverage))
    print("[DEV] Tipo de las keys", type(k))
    return grid_coverage


