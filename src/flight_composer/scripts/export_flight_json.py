#!/usr/bin/env python3
"""
GoPro Telemetry Universal Dumper

1. Extracts ALL available GPMF streams (GPS, ACCL, GYRO, CORI, GRAV, etc.).
2. Automatically applies SCAL (scaling) factors to convert raw integers to floats.
3. Preserves all data (NO cropping/filtering).
4. Exports to a large hierarchical JSON file for post-processing.
"""

import argparse
import json
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List, Union

# Dependencies
from flight_composer import config
from flight_composer.flight_file import find_gopro_file

try:
    import gpmf.io

    GPMF_AVAILABLE = True
except ImportError:
    GPMF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- GPMF Type Mapping ---
GPMF_TYPES = {
    "b": "b",
    "B": "B",
    "s": "h",
    "S": "H",
    "l": "i",
    "L": "I",
    "f": "f",
    "d": "d",
}

# --- Approximate Frequencies for time offsets ---
# Used to generate 't_off' (seconds) for animation synchronization.
# These are standard GoPro defaults.
FREQ_MAP = {
    "GPS5": 18.0,
    "ACCL": 200.0,
    "GYRO": 200.0,
    "GRAV": 200.0,
    "CORI": 200.0,  # Camera Orientation (Quaternions)
    "MAGN": 24.0,
    "ISOE": 1.0,  # Metadata usually per frame or less
    "SHUT": 1.0,
    "WBAL": 1.0,
}


class UniversalParser:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.streams = {}  # {'ACCL': [{'v': [1,2,3], 's': [100]}, ...], ...}

    def parse(self):
        self._recurse(self.payload, scale=None)
        return self._post_process()

    def _recurse(self, payload: bytes, scale=None):
        i = 0
        length = len(payload)

        while i < length:
            # Safety check for buffer end
            if i + 8 > length:
                break

            key = payload[i : i + 4].decode("latin1", errors="ignore")
            # Header: Type(1), Size(1), Repeat(2)
            header = struct.unpack(">cBH", payload[i + 4 : i + 8])
            type_char = header[0].decode("latin1")
            item_size = header[1]
            repeat = header[2]

            data_len = item_size * repeat
            # GPMF data is 32-bit aligned
            total_block_size = 8 + data_len + ((4 - (8 + data_len) % 4) % 4)

            chunk = payload[i + 8 : i + 8 + data_len]

            # 1. Update Contextual Scale
            if key == "SCAL":
                if type_char in GPMF_TYPES:
                    fmt = ">" + (GPMF_TYPES[type_char] * repeat)
                    try:
                        scale = struct.unpack(fmt, chunk)
                    except struct.error:
                        pass  # Keep previous scale if corrupt

            # 2. Recurse Nested Container
            elif type_char == "\x00":
                self._recurse(chunk, scale)

            # 3. Parse Sensor Data
            elif type_char in GPMF_TYPES:
                fmt_char = GPMF_TYPES[type_char]
                fmt_size = struct.calcsize(">" + fmt_char)

                # Calculate how many values form one "sample"
                # e.g. ACCL is x,y,z (size=6 bytes usually, shorts).
                # If item_size=6 and fmt_size=2 (short), elements_per_sample = 3.
                if fmt_size > 0:
                    elements_per_sample = item_size // fmt_size
                else:
                    elements_per_sample = 1

                # Unpack all values in this chunk
                total_elements = data_len // fmt_size
                full_fmt = ">" + (fmt_char * total_elements)

                try:
                    raw_flat = struct.unpack(full_fmt, chunk)

                    if key not in self.streams:
                        self.streams[key] = []

                    # Group flat data into samples
                    # If repeat=50, we have 50 samples.
                    # Each sample has `elements_per_sample` values.

                    # Handle single value case efficiently
                    if elements_per_sample <= 1:
                        # Optimization: Store directly
                        for val in raw_flat:
                            self.streams[key].append({"v": [val], "s": scale})
                    else:
                        # Grouping (e.g. triples for vectors)
                        for k in range(0, len(raw_flat), elements_per_sample):
                            sample = raw_flat[k : k + elements_per_sample]
                            self.streams[key].append({"v": list(sample), "s": scale})

                except struct.error:
                    pass  # Skip unparseable chunks

            i += total_block_size

    def _post_process(self) -> Dict[str, List]:
        """Convert Ints to Floats using Scale and format JSON."""
        final_output = {}

        for key, rows in self.streams.items():
            # Skip empty or massive binary blobs that aren't sensors
            if not rows:
                continue

            # Determine Frequency for timing
            freq = FREQ_MAP.get(key, None)

            cleaned_stream = []

            for i, row in enumerate(rows):
                vals = row["v"]
                scale = row["s"]

                # --- SCALING LOGIC ---
                # Divides raw integer by scale factor
                scaled_vals = vals  # Default to raw

                if scale:
                    try:
                        # Case A: 1 Scale factor for N values (common)
                        if len(scale) == 1 and scale[0] != 0:
                            scaled_vals = [x / scale[0] for x in vals]

                        # Case B: N Scale factors for N values (rare, but possible)
                        elif len(scale) == len(vals):
                            scaled_vals = [
                                x / s if s != 0 else x for x, s in zip(vals, scale)
                            ]
                    except Exception:
                        pass  # Fallback to raw on math errors

                # --- FORMATTING ---
                entry = {}

                # Add timestamp if frequency is known
                if freq:
                    entry["t_off"] = round(i / freq, 4)

                # Generic Vector Naming
                if key == "GPS5" and len(scaled_vals) >= 5:
                    entry.update(
                        {
                            "lat": scaled_vals[0],
                            "lon": scaled_vals[1],
                            "alt": scaled_vals[2],
                            "spd2d": scaled_vals[3],
                            "spd3d": scaled_vals[4],
                        }
                    )
                elif len(scaled_vals) == 3:
                    entry.update(
                        {"x": scaled_vals[0], "y": scaled_vals[1], "z": scaled_vals[2]}
                    )
                elif len(scaled_vals) == 4:
                    # Likely Quaternion (CORI)
                    entry.update(
                        {
                            "w": scaled_vals[0],
                            "x": scaled_vals[1],
                            "y": scaled_vals[2],
                            "z": scaled_vals[3],
                        }
                    )
                elif len(scaled_vals) == 1:
                    entry["val"] = scaled_vals[0]
                else:
                    entry["v"] = scaled_vals  # Generic list

                cleaned_stream.append(entry)

            final_output[key] = cleaned_stream

        return final_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-number", type=str, required=True)
    args = parser.parse_args()

    # 1. Locate File
    mp4_path = find_gopro_file(args.flight_number)
    if not mp4_path:
        logger.error(f"Flight {args.flight_number}: File not found.")
        return

    logger.info(f"Processing {mp4_path.name}...")

    # 2. Extract Binary
    if not GPMF_AVAILABLE:
        logger.error("GPMF library not found. Install via 'pip install gpmf'")
        return

    payload = gpmf.io.extract_gpmf_stream(str(mp4_path))
    if not payload:
        logger.error("No GPMF stream found in file.")
        return

    # 3. Universal Parse (Scale + Dump)
    parser = UniversalParser(payload)
    data = parser.parse()

    # 4. Save JSON
    out_file = Path(config.TRAJECTORY_CSV_DIR) / f"flight_{args.flight_number}_all.json"

    # Simple wrapper metadata
    wrapper = {
        "flight_id": args.flight_number,
        "source": mp4_path.name,
        "streams": data,
    }

    with open(out_file, "w") as f:
        # indent=None creates a minified file (smaller size)
        # Use indent=2 if you need to read it manually
        json.dump(wrapper, f, indent=None)

    logger.info(f"Success. Extracted {len(data)} streams.")
    logger.info(f"Saved to: {out_file}")

    # Log stream summary
    for k, v in data.items():
        logger.info(f" - {k}: {len(v)} samples")


if __name__ == "__main__":
    main()
