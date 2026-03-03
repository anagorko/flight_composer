import argparse
import json
import logging

import numpy as np
import pyproj
from shapely.geometry import LineString, mapping
from shapely.ops import transform

from flight_composer.config import config
from flight_composer.flight_track import FlightTrack
from flight_composer.flight_trajectory import FlightTrajectory
from flight_composer.kinematic_spline import KinematicSpline
from flight_composer.logger import setup_logging
from flight_composer.overlay_text import (
    get_flight_map_overlay_text,
    get_gopro_overlay_text,
)
from flight_composer.phase_subtitles import generate_phase_srt
from flight_composer.processing.extract_gopro_data import extract_gopro_data
from flight_composer.processing.extract_igc_data import extract_igc_data
from flight_composer.processing.flight_data_sources import find_data_sources


def main() -> None:
    setup_logging()
    logger = logging.getLogger("flight_actor_data")

    parser = argparse.ArgumentParser(
        description="Create flight actor data file using flight UID",
    )

    parser.add_argument(
        "--flight_uid", help="Flight UID, e.g., 07", default="07", type=str
    )
    args = parser.parse_args()

    flight_uid = args.flight_uid

    flight_data_sources = find_data_sources(flight_uid)
    flight_tag = flight_data_sources.flight_tag

    logger.info(f"Processing flight [green]{flight_tag}[/green].")

    if flight_data_sources.gopro_path:
        # GoPro data is preferred
        logger.info(f"Found GoPro data, {flight_data_sources.gopro_path}.")

        gopro_telemetry = extract_gopro_data(flight_data_sources.gopro_path)
        if gopro_telemetry:
            logger.info(
                f"Extracted MP4 telemetry data: [green]{get_gopro_overlay_text(gopro_telemetry)}[/green]. Fitting GoPro spline."
            )
            track = FlightTrack.from_gopro(gopro_telemetry)
            trajectory = FlightTrajectory.from_track(track)
        else:
            trajectory = None
    elif flight_data_sources.igc_path:
        # IGC if no GoPro data exists
        logger.info(f"Found IGC data, {flight_data_sources.igc_path}.")

        igc_telemetry = extract_igc_data(flight_data_sources.igc_path)
        if igc_telemetry:
            logger.info(
                f"Extracted IGC telemetry data: [green]{get_flight_map_overlay_text(igc_telemetry)}[/green]"
            )

            track = FlightTrack.from_igc(igc_telemetry)
            trajectory = FlightTrajectory.from_track(track)
        else:
            trajectory = None
    else:
        trajectory = None
        logger.error("No data found.")

    if trajectory:
        config.DIR.ACTOR_DATA.mkdir(parents=True, exist_ok=True)
        trajectory_df = trajectory.trajectory_df()

        trajectory_df_parquet_path = (
            config.DIR.ACTOR_DATA / f"{flight_tag}_trajectory_df.parquet"
        )
        trajectory_df.to_parquet(
            trajectory_df_parquet_path, index=False, engine="pyarrow"
        )

        trajectory_df_csv_path = (
            config.DIR.ACTOR_DATA / f"{flight_tag}_trajectory_df.csv"
        )
        trajectory_df.to_csv(trajectory_df_csv_path, index=True)

        logger.info(
            f"Trajectory data saved to {trajectory_df_parquet_path} and {trajectory_df_csv_path}."
        )


if __name__ == "__main__":
    main()
