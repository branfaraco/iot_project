"""
Fetches and processes OSM data, mapping it to LBCS codes and exporting it as GeoJSON.
"""

import overpass
import time
import copy
from typing import List
import traceback

# Radius in km for each H3 resolution (from https://h3geo.org/docs/core-library/restable/)
RESOL_RADIUS = {
    0: 1281.256011, 1: 483.0568391, 2: 182.5129565, 3: 68.97922179,
    4: 26.07175968, 5: 9.85409099, 6: 3.724532667, 7: 1.406475763,
    8: 0.531414010, 9: 0.200786148, 10: 0.075863783, 11: 0.028663897,
    12: 0.010830188, 13: 0.004092010, 14: 0.001546100, 15: 0.000584169
}


def query_osm(grid_corners_latlon: List[List[float]], gridType, cellSize: int):
    """
    Queries OSM via Overpass API and returns raw GeoJSON data for a set of tags in a given area.

    Args:
        grid_corners_latlon (list): Polygon of lat/lon points defining the area.
        resolution (int): H3 resolution (used to slightly expand the bounding box).

    Returns:
        dict: Raw GeoJSON response.
    """
    tags = ["landuse", "amenity", "leisure", "building", "highway"]

    latitudes = [corner[0] for corner in grid_corners_latlon]
    longitudes = [corner[1] for corner in grid_corners_latlon]
    if gridType == "h3":
        resolution = cellSize
        degree_buffer = RESOL_RADIUS[resolution] / 111.32  # Approx. km to degrees
    else: # square grid
        degree_buffer = 0.2 / 111.32 # se coge 1 km como buffer directamente
    # degree_buffer = 0
    
    min_lat = min(latitudes) - degree_buffer
    max_lat = max(latitudes) + degree_buffer
    min_lon = min(longitudes) - degree_buffer
    max_lon = max(longitudes) + degree_buffer

    query_body = "".join(
        [f'nwr["{tag}"]({min_lat},{min_lon},{max_lat},{max_lon});\n' for tag in tags])
    query = f"(\n{query_body});\nout geom;"

    for attempt in range(1, 6):
        print(f"[Attempt {attempt}] Querying OSM...")
        try:
            api = overpass.API(timeout=600, debug=False)
            geojson_data = api.get(query, responseformat="geojson")
            print("[Success] OSM query completed.")
            return geojson_data 
        except Exception as e:
            print(f"[Error] Attempt {attempt} failed: {type(e).__name__}: {e}")
            # Stack completo
            #traceback.print_exc()
            # Si la excepción lleva .response (p. ej. un HTTPError), muéstrala
            if hasattr(e, 'response'):
                print("Status code:", e.response.status_code)
                print("Response body:", e.response.text)
            # Opcional: capturar el Timeout de overpass por separado
            # except overpass.errors.TimeoutError as te:
            #     print(f"[TimeoutError después de {te.args[0]}s]")
            if attempt == 5:
                raise
            time.sleep(1)
    


def map_osm_lbcs(geojson_data_raw: dict, mapping: dict):
    """
    Maps OSM tags to LBCS codes using a user-defined mapping.

    Args:
        geojson_data_raw (dict): Raw GeoJSON data with OSM tags.
        mapping (dict): Dictionary mapping "tag:value" to list of LBCS codes.

    Returns:
        dict: Modified GeoJSON with "lbcs_codes" field added to features.
    """
    geojson_data = copy.deepcopy(geojson_data_raw)
    cleaned_features = []

    for feature in geojson_data_raw['features']:
        if 'properties' in feature and 'tags' in feature['properties']:
            tags_osm = feature['properties']['tags']
            lbcs_codes = set()

            for tag_key, tag_value in tags_osm.items():
                tag_osm = f"{tag_key}:{tag_value}"
                if tag_osm in mapping:
                    lbcs_codes |= set(mapping[tag_osm])

            cleaned_feature = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    "tags": tags_osm,
                    "lbcs_codes": list(lbcs_codes),
                    "id": feature['properties'].get('id')
                }
            }
            cleaned_features.append(cleaned_feature)

    geojson_data['features'] = cleaned_features
    print(f"[Info] {len(cleaned_features)} features mapped to LBCS.")
    return geojson_data
