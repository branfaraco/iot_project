from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from collections import defaultdict
import json
import os
from grid_maker.utilities.utilities import open_json_file


def check_keys_consistency(original_dict, final_dict):
    """
    Compares keys in the original and final TF-IDF dictionaries.
    Prints if any codes were unexpectedly added or removed.
    """
    for key, orig_value in original_dict.items():
        orig_codes = set(orig_value.keys())
        final_codes = set(final_dict.get(key, {}).keys())
        new_codes = final_codes - orig_codes
        missing_codes = orig_codes - final_codes
        if new_codes:
            print(f"[Warning] Key {key} has new codes: {new_codes}")
        if missing_codes:
            print(f"[Info] Key {key} is missing codes (probably TF-IDF=0): {missing_codes}")


def calculate_tfidf(grid_code_coverage):
    """
    Applies TF-IDF transformation to a dictionary representing LBCS code coverage per cell.

    Args:
        grid_code_coverage (dict): Dict of {cell_id: {code: weight}} values.

    Returns:
        dict: Transformed dictionary with TF-IDF weights.
    """
    all_codes = set()
    for subdict in grid_code_coverage.values():
        all_codes.update(subdict.keys())

    df = pd.DataFrame.from_dict(grid_code_coverage, orient='index', columns=list(all_codes)).fillna(0)

    tfidf_matrix = TfidfTransformer().fit_transform(df)
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), index=df.index, columns=df.columns)

    tfidf_dict = df_tfidf.to_dict(orient='index')

    # Ensure all original keys are preserved
    for key in grid_code_coverage.keys():
        tfidf_dict.setdefault(key, {})

    # Remove entries with TF-IDF = 0
    for key in tfidf_dict:
        tfidf_dict[key] = {code: val for code, val in tfidf_dict[key].items() if val != 0}

    return tfidf_dict
