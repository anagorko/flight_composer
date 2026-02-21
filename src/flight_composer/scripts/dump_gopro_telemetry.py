#!/usr/bin/env python3
"""
GPMF Raw Array Extractor

Extracts ALL GPMF streams, correctly handling "Packed Arrays"
(high-frequency sensors like Gyro/Accel).
Outputs raw integer data to CSVs without processing.
"""

import argparse
import logging
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Use gpmf ONLY for the initial binary extraction
try:
    import gpmf.io

    GPMF_AVAILABLE = True
except ImportError:
    GPMF_AVAILABLE = False

from flight_composer import config
from flight_composer.flight_file import find_gopro_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Mapping GPMF type chars to Python struct format chars
# https://github.com/gopro/gpmf-parser
TYPE_MAP = {
    "b": "b",
    "B": "B",
    "s": "h",
    "S": "H",
    "l": "i",
    "L": "I",
    "f": "f",
    "d": "d",
    "J": "Q",
}


def parse_gpmf_deep(payload: bytes) -> Dict[str, List[Any]]:
    """
    Parses GPMF binary. Handles distinct cases:
    1. Standard values.
    2. Nested containers.
    3. Packed arrays (Crucial for GYRO/ACCL).
    """
    results = {}
    i = 0
    length = len(payload)

    while i < length:
        # 1. Read Key (4 bytes)
        key = payload[i : i + 4].decode("latin1", errors="ignore")

        # 2. Read Header
        # Type (1 byte), Size (1 byte), Repeat (2 bytes)
        header = struct.unpack(">cBH", payload[i + 4 : i + 8])
        type_char = header[0].decode("latin1")
        sample_size = header[1]
        repeat = header[2]

        data_len = sample_size * repeat
        total_block = 8 + data_len

        # 4-byte alignment padding
        padding = (4 - (total_block % 4)) % 4
        next_i = i + total_block + padding

        payload_data = payload[i + 8 : i + 8 + data_len]

        # Case A: Nested Container (Recurse)
        if type_char == "\x00":
            nested = parse_gpmf_deep(payload_data)
            for k, v in nested.items():
                if k not in results:
                    results[k] = []
                results[k].extend(v)

        # Case B: Data Stream
        elif type_char in TYPE_MAP:
            fmt_char = TYPE_MAP[type_char]

            # This is the fix: Handling Packed Arrays
            # Calculate how many "items" are in this block
            # e.g. Gyro is X,Y,Z (3 values). If repeat=60, we have 20 samples of 3 values.

            # Struct format for ONE single value
            single_fmt = ">" + fmt_char
            single_size = struct.calcsize(single_fmt)

            # Total elements in this payload
            total_elements = data_len // single_size

            # Unpack everything as a flat list of numbers first
            full_fmt = ">" + (fmt_char * total_elements)

            try:
                flat_values = struct.unpack(full_fmt, payload_data)

                # Organize based on known sensor dimensions
                # Most high-freq sensors (ACCL, GYRO, MAGN) are 3-axis (x,y,z)
                # But sometimes GPMF treats them as a flat array of 'repeat' length.

                # Heuristic: If key implies 3-axis and count is divisible by 3
                if (
                    key in ["ACCL", "GYRO", "MAGN", "GRAV"]
                    and len(flat_values) % 3 == 0
                ):
                    # Group into (x,y,z) tuples
                    grouped = [
                        flat_values[j : j + 3] for j in range(0, len(flat_values), 3)
                    ]
                    if key not in results:
                        results[key] = []
                    results[key].extend(grouped)

                # Standard case (1 value per block, or unknown structure)
                elif total_elements == 1:
                    if key not in results:
                        results[key] = []
                    results[key].append(flat_values[0])
                else:
                    # Generic list handling
                    if key not in results:
                        results[key] = []
                    results[key].append(flat_values)

            except struct.error:
                pass  # Skip corrupted/complex structs

        i = next_i

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-number", type=str, required=True)
    args = parser.parse_args()

    # 1. Get Binary
    mp4_path = find_gopro_file(args.flight_number)
    if not mp4_path:
        print("File not found")
        return

    print(f"Extracting raw streams from {mp4_path.name}...")
    stream_bytes = gpmf.io.extract_gpmf_stream(str(mp4_path))

    if not stream_bytes:
        print("No GPMF stream found.")
        return

    # 2. Deep Parse
    streams = parse_gpmf_deep(stream_bytes)

    # 3. Dump to CSV
    out_dir = Path(config.TRAJECTORY_CSV_DIR) / f"flight_{args.flight_number}_raw_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(streams)} streams. Saving to {out_dir}...")

    for key, data in streams.items():
        if not data:
            continue

        try:
            df = pd.DataFrame(data)

            # Naming columns for convenience if it looks like X,Y,Z
            if len(df.columns) == 3:
                df.columns = ["x", "y", "z"]

            csv_path = out_dir / f"{key}.csv"
            df.to_csv(csv_path, index=False)
            print(f" - {key}: {len(df)} rows")

        except Exception as e:
            print(f"Failed to save {key}: {e}")


if __name__ == "__main__":
    main()
