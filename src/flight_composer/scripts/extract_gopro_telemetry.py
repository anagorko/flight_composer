#!/usr/bin/env python3
"""
Extract Telemetry GPMF Script

Extracts GPS telemetry from GoPro MP4 files into CSVs.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import gpmf.gps
import gpmf.io
import numpy as np
import pandas as pd
import rich.console
import rich.logging

from flight_composer import config
from flight_composer.flight_file import find_gopro_file, list_available_flights

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    log_format = "\\[[bold]%(name)s[/bold]] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=log_format,
        datefmt="[%X]",
        handlers=[
            rich.logging.RichHandler(
                console=rich.console.Console(color_system="auto"),
                show_level=True,
                show_path=False,
                enable_link_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
            )
        ],
    )


def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance calculation in meters."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def extract_gps_data(mp4_path: Path) -> pd.DataFrame | None:
    """Parses GPMF stream and returns a DataFrame with GPS data."""
    try:
        stream = gpmf.io.extract_gpmf_stream(str(mp4_path))
        if not stream:
            return None

        gps_blocks = list(gpmf.gps.extract_gps_blocks(stream))
        if not gps_blocks:
            return None

        # Data accumulators
        data = {
            "time": [],
            "lat": [],
            "lon": [],
            "alt": [],
            "speed": [],
            "speed_3d": [],
            "fix": [],
        }

        GPS_FIX_MAP = {0: "NO_FIX", 2: "FIX_2D", 3: "LOCK_3D"}

        for block in gps_blocks:
            parse = gpmf.gps.parse_gps_block(block)
            count = len(parse.latitude)

            # Timestamp interpolation
            base_ts = datetime.fromisoformat(parse.timestamp.replace("Z", "+00:00"))
            # Generate linear timestamps for this block
            timestamps = [base_ts + timedelta(seconds=i / count) for i in range(count)]

            data["time"].extend(timestamps)
            data["lat"].extend(parse.latitude)
            data["lon"].extend(parse.longitude)
            data["alt"].extend(parse.altitude)
            data["speed"].extend(parse.speed_2d)
            data["speed_3d"].extend(
                parse.speed_3d if hasattr(parse, "speed_3d") else [np.nan] * count
            )
            data["fix"].extend([GPS_FIX_MAP.get(int(parse.fix), "UNKNOWN")] * count)

        df = pd.DataFrame(data)

        # --- Derived Calculations (Vectorized) ---
        if not df.empty:
            # Distance
            df["dist_diff"] = haversine_vectorized(
                df["lat"].shift(), df["lon"].shift(), df["lat"], df["lon"]
            ).fillna(0)
            df["dist"] = df["dist_diff"].cumsum()  # Odometer

            # Gradient (Rise / Run * 100)
            alt_diff = df["alt"].diff()
            df["grad"] = (alt_diff / df["dist_diff"].replace(0, np.nan) * 100).fillna(0)

            # Cleanup
            df.drop(columns=["dist_diff"], inplace=True)

        return df

    except Exception as e:
        logger.error(f"Failed to extract GPS from {mp4_path.name}: {e}")
        return None


def process_flight(flight_num: str, refresh: bool) -> bool:
    """Orchestrates extraction for a single flight."""
    mp4_path = find_gopro_file(flight_num)
    if not mp4_path:
        logger.warning(f"Flight {flight_num}: MP4 not found.")
        return False

    csv_name = mp4_path.stem + ".csv"
    csv_path = Path(config.TRAJECTORY_CSV_DIR) / csv_name

    # Skip if fresh
    if not refresh and csv_path.exists():
        if csv_path.stat().st_mtime > mp4_path.stat().st_mtime:
            logger.debug(f"Flight {flight_num}: Up to date.")
            return True

    logger.info(f"Processing Flight {flight_num}...")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = extract_gps_data(mp4_path)
    if df is not None and not df.empty:
        df.to_csv(csv_path, index=False)
        logger.info(f"Flight {flight_num}: Saved {len(df)} points.")
        return True

    logger.error(f"Flight {flight_num}: No GPS data found.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Extract GoPro GPS Telemetry")
    parser.add_argument("--flight-number", type=str, help="Process specific flight")
    parser.add_argument("--refresh", action="store_true", help="Force re-processing")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    flights = [args.flight_number] if args.flight_number else list_available_flights()

    if not flights:
        logger.warning("No flights found.")
        return

    logger.info(f"Found {len(flights)} flights to process.")

    results = [process_flight(f, args.refresh) for f in flights]

    logger.info(
        f"Summary: {sum(results)} processed, {len(results) - sum(results)} failed/skipped."
    )


if __name__ == "__main__":
    main()
