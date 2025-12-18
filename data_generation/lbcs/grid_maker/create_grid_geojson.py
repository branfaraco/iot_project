import h3
import math
import json
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
from shapely.affinity import rotate
from pyproj import Transformer


def create_h3_geojson(grid_corners_latlon, resolution):
    """
    Generates a GeoJSON FeatureCollection from a polygon defined by grid_corners_latlon using H3 cells.
    """
    poly_latlon = h3.LatLngPoly(outer=grid_corners_latlon)
    grid_h3 = h3.h3shape_to_cells(poly_latlon, resolution)

    features = []
    for idx, h3_cell in enumerate(grid_h3):
        polygon = h3.cell_to_boundary(h3_cell)
        polygon_lonlat = [[point[1], point[0]] for point in polygon]
        polygon_lonlat.append(polygon_lonlat[0])

        feature = {
            "type": "Feature",
            "properties": {"id": idx,
                           "other_properties": {
                               "h3_index": h3_cell
                           }},
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_lonlat],
            }
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def create_squares_geojson(grid_corners_latlon, cellSize: int, gridRotation: int, regularGrid: bool):
    """
    Generates a GeoJSON FeatureCollection of square cells over a defined area.
    """
    def convert_coords(coords):
        if isinstance(coords[0][0], (int, float)):
            return [(pt[1], pt[0]) for pt in coords]
        exterior = [(pt[1], pt[0]) for pt in coords[0]]
        holes = [[(pt[1], pt[0]) for pt in ring]
                 for ring in coords[1:]] if len(coords) > 1 else None
        return (exterior, holes)

    try:
        converted = convert_coords(grid_corners_latlon)
        polygon = Polygon(*converted) if isinstance(converted,
                                                    tuple) else Polygon(converted)
    except Exception as e:
        raise ValueError(f"Invalid polygon coordinates: {e}")

    transformer = Transformer.from_crs(
        "epsg:4326", "epsg:3857", always_xy=True)
    polygon_proj = transform(transformer.transform, polygon)

    minx, miny, maxx, maxy = polygon_proj.bounds
    side = cellSize
    grid_squares = []

    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            square = Polygon([
                (x, y),
                (x + side, y),
                (x + side, y + side),
                (x, y + side),
                (x, y)
            ])
            grid_squares.append(square)
            y += side
        x += side

    if gridRotation != 0:
        center = ((minx + maxx) / 2, (miny + maxy) / 2)
        grid_squares = [rotate(sq, gridRotation, origin=center)
                        for sq in grid_squares]

    if not regularGrid:
        grid_squares = [sq for sq in grid_squares if polygon_proj.contains(sq)]

    inverse_transformer = Transformer.from_crs(
        "epsg:3857", "epsg:4326", always_xy=True)

    def project_back(geom):
        return transform(inverse_transformer.transform, geom)

    features = []
    for idx, sq in enumerate(grid_squares):
        sq_latlon = project_back(sq)
        feature = {
            "type": "Feature",
            "properties": {"id": idx},
            "geometry": mapping(sq_latlon)
        }
        features.append(feature)

    return {
        "type": "FeatureCollection",
        "features": features
    }


def create_grid_geojson(grid_corners_latlon, gridType="h3", cellSize=10, gridRotation=0, regularGrid=False):
    """
    Wrapper to generate a grid (H3 or square) in GeoJSON format.
    """
    if not grid_corners_latlon or not isinstance(grid_corners_latlon, list):
        raise ValueError(
            "grid_corners_latlon must be a non-empty list of coordinates.")

    if gridType == "h3":
        return create_h3_geojson(grid_corners_latlon, cellSize)
    elif gridType == "squares":
        return create_squares_geojson(grid_corners_latlon, cellSize, gridRotation, regularGrid)
    else:
        raise ValueError(f"Unsupported grid mode: {gridType}")
