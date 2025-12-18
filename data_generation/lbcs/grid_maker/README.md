# grid_maker

This package provides a set of utilities to build an urban land-use grid aligned with a target area. It can generate either an H3 grid (hexagonal) at a given resolution or a regular square grid, download relevant OpenStreetMap features for the area, map these features to Land-Based Classification System (LBCS) codes, compute the fractional coverage and optional building height per cell, apply TF–IDF weighting to emphasise discriminative land uses, and assemble a final annotated GeoJSON complete with descriptive metadata and colours.

The grid output is intended for downstream tasks such as training the enriched traffic forecasting model: it provides a fixed spatial context that can be concatenated with traffic frames to supply land-use information at each pixel.

## Functionality overview

The package is composed of several modules that together implement the grid construction pipeline:

- `grid_maker.py` – Orchestrates the whole process. Given a list of corner coordinates defining the area of interest, it generates a base grid (H3 or square), queries OpenStreetMap for a defined set of tags, maps the retrieved tags to LBCS codes using the provided mapping, computes per-cell coverage and optional average heights, applies TF–IDF weighting to the coverage histogram, builds a descriptive header and finalises the grid into a single GeoJSON FeatureCollection. This module exposes a `grid_maker()` function rather than a command-line interface; it is intended to be imported and called from other scripts.

- `create_grid_geojson.py` – Provides helpers to construct the base grid geometry. `create_h3_geojson()` converts a polygon into an H3 tessellation at a given resolution, while `create_squares_geojson()` divides the area into square cells of a given side length, with optional rotation or inclusion/exclusion of partially overlapping cells. The wrapper `create_grid_geojson()` chooses between H3 and square modes based on a `gridType` argument.

- `query_osm.py` – Interfaces with the Overpass API to download OpenStreetMap primitives (points, lines and polygons) for a small set of high-level land-use tags (`landuse`, `amenity`, `leisure`, `building`, `highway`). It expands the bounding box slightly to ensure complete coverage at the chosen grid resolution. The module also provides `map_osm_lbcs()`, which transforms raw OSM features into simplified GeoJSON features annotated with the LBCS codes specified in `config/mapping.json`. Only features with mapped tags are retained.

- `coverage_figures.py` – Computes fractional area coverage of each LBCS code per cell. It builds an R-tree index of the grid polygons for efficient spatial queries, intersects each OSM geometry with the potentially overlapping cells and accumulates the area proportions for every LBCS code present. Coverage vectors are normalised such that the weights per cell sum to one. (The code currently contains placeholders for building height integration; this part remains incomplete.)

- `tf_idf.py` – Applies a TF–IDF transformation to the per-cell coverage histograms. This emphasises codes that are distinctive to a particular cell relative to the entire grid by down-weighting ubiquitous classes. The result is a dictionary of TF–IDF weights with the same keys as the input coverage dictionary.

- `build_final_geojson.py` – Assembles the enriched GeoJSON. For each cell, it chooses a primary LBCS code by aggregating the coverage weights up the hierarchy, assigns an RGB colour derived from the LBCS code with controlled variation for sub-levels, and copies the optional height information. It also builds a `types` dictionary that lists every LBCS code encountered with its description (taken from `config/lbcs_sort_desc.json`), colour, and editor properties. Global metadata such as the grid type, resolution, number of cells and approximate centre coordinates are stored in a header property.

- `construct_header.py` – Creates the header dictionary that records the grid type, resolution/cell size, rotation flag, number of cells and chosen LBCS depth. This metadata is embedded in the `properties.header` of the final GeoJSON.

- `config/` – Holds configuration files required by the pipeline:
  - `mapping.json` maps specific OSM tag–value pairs to one or more four-digit LBCS codes.
  - `lbcs_sort_desc.json` associates each LBCS code with a human-readable description. These descriptions are used to populate the `types` section of the output GeoJSON.
  - The `README.md` in this directory briefly summarises the contents.

- `utilities/` – Contains helpers used throughout the package, such as `open_json_file()` for loading configuration files relative to the module’s location.
