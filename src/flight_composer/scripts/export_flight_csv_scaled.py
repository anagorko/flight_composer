#!/usr/bin/env python3
"""
GoPro Scaled CSV Dumper

1. Extracts ALL GPMF streams.
2. Applies SCAL (scaling) factors automatically (Raw Int -> Float).
3. Saves each stream to a separate, readable CSV file.
"""

import argparse
import csv
import logging
import struct
from pathlib import Path
from typing import Any, Dict, List

from flight_composer import config
from flight_composer.flight_file import find_gopro_file

try:
    import gpmf.io

    GPMF_AVAILABLE = True
except ImportError:
    GPMF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# --- GPMF Constants ---
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

# Standard Frequencies for calculating t_offset (Time in seconds)
FREQ_MAP = {
    "GPS5": 18.0,
    "ACCL": 200.0,
    "GYRO": 200.0,
    "GRAV": 200.0,
    "CORI": 200.0,
    "IORI": 200.0,
    "WNDM": 20.0,
}


class ScaledCSVDumper:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.streams = {}

    def parse(self):
        self._recurse(self.payload, scale=None)
        return self.streams

    def _recurse(self, payload: bytes, scale=None):
        i = 0
        length = len(payload)
        while i < length:
            if i + 8 > length:
                break
            key = payload[i : i + 4].decode("latin1", errors="ignore")
            header = struct.unpack(">cBH", payload[i + 4 : i + 8])
            type_char = header[0].decode("latin1")
            item_size = header[1]
            repeat = header[2]
            data_len = item_size * repeat
            total_block = 8 + data_len + ((4 - (8 + data_len) % 4) % 4)
            chunk = payload[i + 8 : i + 8 + data_len]

            if key == "SCAL":
                if type_char in GPMF_TYPES:
                    fmt = ">" + (GPMF_TYPES[type_char] * repeat)
                    try:
                        scale = struct.unpack(fmt, chunk)
                    except:
                        pass
            elif type_char == "\x00":
                self._recurse(chunk, scale)
            elif type_char in GPMF_TYPES:
                self._store_stream_data(key, type_char, item_size, repeat, chunk, scale)

            i += total_block

    def _store_stream_data(self, key, type_char, item_size, repeat, chunk, scale):
        fmt_char = GPMF_TYPES[type_char]
        fmt_size = struct.calcsize(">" + fmt_char)
        if fmt_size == 0:
            return

        # Unpack Raw
        total_elements = len(chunk) // fmt_size
        full_fmt = ">" + (fmt_char * total_elements)
        try:
            raw_flat = struct.unpack(full_fmt, chunk)
        except:
            return

        # Determine grouping (e.g. 3 for x,y,z)
        elements_per_sample = 1
        if repeat > 0:
            # item_size is total bytes per sample group
            elements_per_sample = item_size // fmt_size

        # Initialize storage
        if key not in self.streams:
            self.streams[key] = []

        # Process samples in this chunk
        for k in range(0, len(raw_flat), elements_per_sample):
            sample = raw_flat[k : k + elements_per_sample]

            # SCALING
            final_val = list(sample)
            if scale:
                try:
                    if len(scale) == 1 and scale[0] != 0:
                        final_val = [x / scale[0] for x in sample]
                    elif len(scale) == len(sample):
                        final_val = [
                            x / s if s != 0 else x for x, s in zip(sample, scale)
                        ]
                except:
                    pass

            self.streams[key].append(final_val)


def save_streams_to_csv(streams: Dict[str, List], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, data in streams.items():
        if not data:
            continue

        # Determine Header
        sample_len = len(data[0])
        header = ["t_off"]  # First column is always time offset

        if key == "GPS5" and sample_len >= 5:
            header.extend(["lat", "lon", "alt", "spd2d", "spd3d"])
        elif key in ["ACCL", "GYRO", "GRAV", "MAGN"] and sample_len == 3:
            header.extend(["x", "y", "z"])
        elif key in ["CORI", "IORI"] and sample_len == 4:
            header.extend(["w", "x", "y", "z"])
        else:
            header.extend([f"v{i}" for i in range(sample_len)])

        # Calculate Time Offsets
        freq = FREQ_MAP.get(key, 1.0)  # Default to 1Hz if unknown
        if key == "GPS5":
            freq = 18.0

        csv_path = output_dir / f"{key}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i, row in enumerate(data):
                t_off = round(i / freq, 4)
                writer.writerow([t_off] + row)

    logger.info(f"Saved {len(streams)} CSV files to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flight-number", type=str, required=True)
    args = parser.parse_args()

    mp4_path = find_gopro_file(args.flight_number)
    if not mp4_path:
        logger.error("File not found.")
        return

    if not GPMF_AVAILABLE:
        logger.error("pip install gpmf")
        return

    logger.info(f"Processing {mp4_path.name}...")
    payload = gpmf.io.extract_gpmf_stream(str(mp4_path))

    # Parse
    dumper = ScaledCSVDumper(payload)
    streams = dumper.parse()

    # Save
    out_dir = Path(config.TRAJECTORY_CSV_DIR) / f"flight_{args.flight_number}_scaled"
    save_streams_to_csv(streams, out_dir)


if __name__ == "__main__":
    main()
