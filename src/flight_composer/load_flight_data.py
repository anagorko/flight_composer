import logging
import pathlib

import yaml

from flight_composer.config import config
from flight_composer.flight_data import FlightData, FlightMetadata
from flight_composer.processing.extract_gopro_data import extract_gopro_data
from flight_composer.processing.extract_igc_data import extract_igc_data

logger = logging.getLogger("flight_composer.load_flight")


def load_flight_data(flight_uid: str) -> FlightData:
    """
    Locates and parses the flight metadata YAML, resolves registries,
    locates media files, and assembles the top-level FlightData object.
    """
    # 1. Locate the metadata YAML
    yaml_candidates = list(config.DIR.FLIGHTS.glob(f"{flight_uid}_*.yaml"))

    if not yaml_candidates:
        raise ValueError(
            f"No metadata YAML found for flight UID {flight_uid} in {config.DIR.FLIGHTS}"
        )
    if len(yaml_candidates) > 1:
        raise ValueError(
            f"Multiple metadata YAMLs found for UID {flight_uid}: {yaml_candidates}"
        )

    yaml_path = yaml_candidates[0]

    # 2. Parse the metadata
    with open(yaml_path, "r", encoding="utf-8") as f:
        metadata_dict = yaml.safe_load(f) or {}

    metadata = FlightMetadata(**metadata_dict)
    flight_tag = metadata.flight_tag

    # 3. Resolve Registries
    if metadata.airfield_id not in config.airfields:
        raise KeyError(
            f"Airfield ID '{metadata.airfield_id}' not found in global airfields registry."
        )
    airfield = config.airfields[metadata.airfield_id]

    if metadata.glider_id not in config.gliders:
        raise KeyError(
            f"Glider ID '{metadata.glider_id}' not found in global gliders registry."
        )
    glider_model = config.gliders[metadata.glider_id]

    # 4. Locate Data Sources (using the exact flight_tag from the YAML)
    mp4_path: pathlib.Path | None = None
    igc_path: pathlib.Path | None = None

    # Check for GoPro
    mp4_lower = config.DIR.MP4 / f"{flight_tag}.mp4"
    mp4_upper = config.DIR.MP4 / f"{flight_tag}.MP4"
    if mp4_lower.exists():
        mp4_path = mp4_lower
    elif mp4_upper.exists():
        mp4_path = mp4_upper

    # Check for IGC
    igc_candidate = config.DIR.IGC / f"{flight_tag}.igc"
    if igc_candidate.exists():
        igc_path = igc_candidate

    if not mp4_path and not igc_path:
        logger.warning(
            f"No MP4 or IGC data found for {flight_tag}. FlightData will be empty."
        )

    # 5. Extract Telemetry
    # (Since you extract inside generate_actor_data.py anyway, doing it here
    # guarantees any script using load_flight_data gets a fully ready object)
    gopro_telemetry = extract_gopro_data(mp4_path) if mp4_path else None
    igc_telemetry = extract_igc_data(igc_path) if igc_path else None

    # 6. Assemble and Return
    return FlightData(
        metadata=metadata,
        airfield=airfield,
        glider_model=glider_model,
        gopro=gopro_telemetry,
        gopro_path=mp4_path,
        igc=igc_telemetry,
        igc_path=igc_path,
    )
