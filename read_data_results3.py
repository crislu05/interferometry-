#!/usr/bin/python
import os
import glob
import numpy as np

def read_data3(fname):
    """
    Robust parser for a single file.
    - Skips non-numeric header lines (e.g. 'ADC1 ADC2 ...').
    - Uses first fully numeric line to set the number of columns.
    - Returns a list of columns: [col0, col1, ..., colN-1], each a list[float].
    """
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    numeric_rows = []
    ncols = None

    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue  # skip blank lines
            parts = s.split()

            # try to convert entire line to floats; if any token fails → header or bad line → skip
            try:
                nums = [float(tok) for tok in parts]
            except ValueError:
                continue

            if ncols is None:
                ncols = len(nums)
            elif len(nums) != ncols:
                # enforce constant width; skip malformed rows
                continue

            numeric_rows.append(nums)

    if not numeric_rows:
        raise ValueError(f"No numeric data rows found in {fname}")

    arr = np.asarray(numeric_rows, dtype=float)
    # Return list-of-columns (compatible with your plotting script)
    return [arr[:, i].tolist() for i in range(arr.shape[1])]


def read_data3_many(directory=".", prefix="interferometry_data_"):
    """
    Read ALL interferometry files in a directory.
    Matches any file whose basename starts with `prefix` (extension optional, case-insensitive).

    Returns:
        dict: {basename_without_ext: list_of_columns}
    """
    directory = os.path.abspath(directory)
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and name.lower().startswith(prefix.lower()):
            files.append(path)
    files.sort()

    if not files:
        raise FileNotFoundError(
            f"No files starting with '{prefix}' found in {directory}"
        )

    out = {}
    for fp in files:
        base = os.path.splitext(os.path.basename(fp))[0]
        out[base] = read_data3(fp)
    return out
