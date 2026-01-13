
import pandas as pd
import numpy as np
from pathlib import Path
import re
import engutil
import pandas as pd
import matplotlib.pyplot as plt
import io


def load_klippel_impedance_tf(path):
    def _extract_block_names(lines):
        for l in lines:
            if l.startswith('"') and 'Measured' in l:
                return [s.lower() for s in re.findall(r'"([^"]+)"', l)]
        raise ValueError("Block-name line not found")
    path = Path(path)

    with path.open() as f:
        lines = [l.rstrip() for l in f if l.strip()]

    data_lines = [l for l in lines if not l.startswith('"')]
    rows = [list(map(float, l.split())) for l in data_lines]
    arr = np.asarray(rows)

    ncols = arr.shape[1]
    nblocks = ncols // 2

    block_names = _extract_block_names(lines)[:nblocks]

    out = {}
    for i, name in enumerate(block_names):
        out[name.lower()] = pd.DataFrame({
            "frequency": arr[:, 2*i],
            "value": arr[:, 2*i + 1],
        })

    return out

def load_klippel_displacement_tf(path):
    path = Path(path)

    with path.open() as f:
        lines = [l.rstrip() for l in f if l.strip()]

    # --- extract block names (2nd quoted line)
    header_line = next(
        l for l in lines if l.startswith('"') and 'Measured' in l
    )
    block_names = re.findall(r'"([^"]+)"', header_line)

    # --- numeric data
    data_lines = [l for l in lines if not l.startswith('"')]
    rows = [list(map(float, l.split())) for l in data_lines]
    arr = np.asarray(rows)

    ncols = arr.shape[1]

    out = {}
    col = 0
    for name in block_names:
        # skip blocks that have no data (e.g. "Imported")
        if col + 1 >= ncols:
            break

        f = arr[:, col]
        y = arr[:, col + 1]

        out[name.lower()] = pd.DataFrame(
            {"frequency": f, "value": y}
        )

        col += 2

    return out

def load_klippel_spectrum_tf(file_path_or_buffer):
    """
    Parses the 3-column-pair spectrum file format.
    Returns a dictionary containing three DataFrames.
    """
    # Read the file. 
    # skiprows=3 skips the headers.
    # sep='\t' assumes tab-separated. 
    # header=None prevents it from trying to use the first row of data as names.
    df = pd.read_csv(file_path_or_buffer, sep='\t', skiprows=3, header=None)

    # The file has 6 columns total (0 to 5)
    # We slice them into pairs.
    
    # 1. Signal Lines (Columns 0 and 1)
    signal = df.iloc[:, [0, 1]].copy()
    signal.columns = ['freq', 'db']
    signal = signal.dropna() # Remove empty rows
    signal = signal.sort_values(by='freq') # Ensure sorted for plotting

    # 2. Noise + Distortions (Columns 2 and 3)
    distortion = df.iloc[:, [2, 3]].copy()
    distortion.columns = ['freq', 'db']
    distortion = distortion.dropna()
    distortion = distortion.sort_values(by='freq')

    # 3. Noise Floor (Columns 4 and 5)
    noise_floor = df.iloc[:, [4, 5]].copy()
    noise_floor.columns = ['freq', 'db']
    noise_floor = noise_floor.dropna()
    noise_floor = noise_floor.sort_values(by='freq')

    return {
        "signal": signal,
        "distortion": distortion,
        "noise_floor": noise_floor
    }


