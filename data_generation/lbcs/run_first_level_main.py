

import json
import os
import shutil


from data_generation.lbcs.split_grid_processor import run_split_processing_first_level


def main() -> None:

    # Base directory containing metadata.json and where output files
    # should be written
    base = r"C:\Users\user\UPM\Imperial-4año\IoT\Github\hugging_face\BERLIN_reduced"
    metadata_path = os.path.join(base, "metadata.json")
    mapping_path = (
        r"C:\Users\user\UPM\Imperial-4año\IoT\Github\data_generation\grid_maker\config\mapping.json"
    )

    # Load metadata and mapping as Python objects
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Define cache directory inside the base directory.  It will be
    # created automatically if it does not exist.
    cache_dir = os.path.join(base, "cache_split")
    lbcs_scaling = 2
    print("Running split‑grid first‑level LBCS processing…")
    grid_geojson, matrix = run_split_processing_first_level(
        metadata=metadata,
        lbcs_scaling = lbcs_scaling,
        mapping=mapping,
        cache_dir=cache_dir,
        max_recursion=3,
        cell_size=300,
    )
    lbcs_folder = os.path.join(base, f"grid_lbcs-{lbcs_scaling}")


    os.makedirs(lbcs_folder, exist_ok=True)

    # Write outputs
    geojson_out = os.path.join(lbcs_folder, "grid_first_level.geojson")
    matrix_out = os.path.join(lbcs_folder, "lbcs_matrix.json")
    with open(geojson_out, "w", encoding="utf-8") as f:
        json.dump(grid_geojson, f, indent=2)
    with open(matrix_out, "w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)
    print(f"Saved simplified grid to {geojson_out}")
    print(f"Saved matrix to {matrix_out}")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Deleted cache directory: {cache_dir}")


if __name__ == "__main__":
    main()


