import argparse
import json
import logging

import numpy as np
import pyproj
from shapely.geometry import LineString, mapping
from shapely.ops import transform

from flight_composer.flight_track import FlightTrack
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
    logger = logging.getLogger("flight_phases")

    parser = argparse.ArgumentParser(
        description="Create phase subtitles for flight recorded with GoPro using flight UID",
    )

    parser.add_argument(
        "--flight_uid", help="Flight UID, e.g., 07", default="07", type=str
    )
    args = parser.parse_args()

    flight_uid = args.flight_uid

    flight_data_sources = find_data_sources(flight_uid)
    flight_tag = flight_data_sources.flight_tag

    logger.info(f"Processing flight [green]{flight_tag}[/green].")

    # GoPro data
    if flight_data_sources.gopro_path:
        logger.info(f"Found GoPro data, {flight_data_sources.gopro_path}.")

        gopro_telemetry = extract_gopro_data(flight_data_sources.gopro_path)
        if gopro_telemetry:
            logger.info(
                f"Extracted MP4 telemetry data: [green]{get_gopro_overlay_text(gopro_telemetry)}[/green]. Fitting GoPro spline."
            )

            gopro_track = FlightTrack.from_gopro(gopro_telemetry)
            generate_phase_srt(
                gopro_track.dataframe,
                "flight_phases_overlay_gopro.srt",
                time_col="timestamp_s",
                phase_col="phase_hmm",
            )
    else:
        logger.info("No GoPro data found.")

    # IGC data
    if flight_data_sources.igc_path:
        logger.info(f"Found IGC data, {flight_data_sources.igc_path}.")

        igc_telemetry = extract_igc_data(flight_data_sources.igc_path)
        if igc_telemetry:
            logger.info(
                f"Extracted IGC telemetry data: [green]{get_flight_map_overlay_text(igc_telemetry)}[/green]"
            )

            # TODO: makes no sense without time_delta to gopro track

            igc_track = FlightTrack.from_igc(igc_telemetry)
            generate_phase_srt(
                igc_track.dataframe,
                "flight_phases_overlay_igc.srt",
                time_col="timestamp_s",
                phase_col="phase_hmm",
            )
    else:
        logger.info("No IGC data found.")


if __name__ == "__main__":
    main()
