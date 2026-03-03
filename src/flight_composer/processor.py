"""
This module implements the data processing pipeline for flight data.
"""

import logging

from flight_composer.config import config
from flight_composer.load_flight_data import load_flight_data
from flight_composer.logger import setup_logging
from flight_composer.overlay_text import (
    get_flight_map_overlay_text,
    get_gopro_overlay_text,
)


def main():
    setup_logging()
    logger = logging.getLogger("flight_composer.processor")

    logger.info("[yellow]Flight Composer Processor[/yellow]")

    flight_uids = []
    if config.DIR.FLIGHTS.exists():
        for yaml_file in config.DIR.FLIGHTS.glob("*.yaml"):
            # Extracts "07" from "07_niskie_ladowanie.yaml"
            uid = yaml_file.name.split("_")[0]
            flight_uids.append(uid)
    flight_uids = sorted(list(set(flight_uids)))

    for flight_uid in flight_uids:
        print()

        ### 1. Load flight data (discovery + extraction in one step)

        flight = load_flight_data(flight_uid)
        flight_tag = flight.metadata.flight_tag

        ### 2. Log extracted telemetry info
        if flight.gopro:
            logger.info(
                f"Extracted [bold]GoPro[/bold] telemetry data: "
                f"[green]{get_gopro_overlay_text(flight.gopro)}[/green]"
            )

        if flight.igc:
            logger.info(
                f"Extracted [bold]IGC[/bold] telemetry data: "
                f"[green]{get_flight_map_overlay_text(flight.igc)}[/green]"
            )

        logger.info(
            f"Flight [green]{flight_tag}[/green] loaded successfully. "
            f"Airfield: {flight.airfield.icao_code}, Glider: {flight.glider_model.name}"
        )


if __name__ == "__main__":
    main()
