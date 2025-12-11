import os
import json
from typing import Literal, Optional



def open_json_file(relative_path: str):
    """
    Opens a JSON file given its relative path from this script's location.

    Args:
        relative_path (str): Relative path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, relative_path)

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"[Success] Loaded JSON file: {file_path}")
        return data
    except FileNotFoundError:
        print(f"[Error] File not found: {file_path}")
    except json.JSONDecodeError:
        print(f"[Error] Failed to parse JSON from: {file_path}")
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")
