import os
import urllib.request

from dotenv import load_dotenv
DATA_ROOT = os.environ["DATA_ROOT"]

def download_berlin_pbf(data_root: str) -> str:
    url = "https://download.geofabrik.de/europe/germany/berlin-latest.osm.pbf"
    local_path = os.path.join(data_root, "berlin-latest.osm.pbf")

    if os.path.exists(local_path):
        print("[Info] Berlin .pbf already exists:", local_path)
        return local_path

    print("[Info] Downloading:", url)
    urllib.request.urlretrieve(url, local_path)
    print("[Info] Downloaded to:", local_path)
    return local_path

berlin_pbf_path = download_berlin_pbf(DATA_ROOT)
