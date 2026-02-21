import pathlib
import typing

import pydantic_settings


class FlightComposerDirs(pydantic_settings.BaseSettings):
    PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).parents[2]
    MP4: pathlib.Path = pathlib.Path("/srv/samba/share/GoProFlights")
    IGC: pathlib.Path = PROJECT_ROOT / "IGCData"

    PROCESSED_DATA: pathlib.Path = PROJECT_ROOT / "ProcessedData"
    GPX: pathlib.Path = PROCESSED_DATA / "GPXData"
    TELEMETRY: pathlib.Path = PROCESSED_DATA / "TelemetryData"
    TRAJECTORY: pathlib.Path = PROCESSED_DATA / "TrajectoryData"


class FlightComposerConfig(pydantic_settings.BaseSettings):
    DIR: FlightComposerDirs = FlightComposerDirs()

    # Flight UIDs to be processed
    FLIGHT_UIDS: list[str] = ["07", "08", "09", "10"]

    # Directories to search for flight files by flight UID
    FLIGHT_SEARCH_DIRS: list[pathlib.Path] = [DIR.MP4, DIR.IGC]

    # --- Sensor Noise Models ---
    # Units: Meters

    GPS_ERROR_SIGMA: float = 5.0
    BARO_ERROR_SIGMA: float = 2.0

    # --- Filter Settings ---
    # Units: Hz
    ROLL_DAMP_CUTOFF: float = 1.0

    # --- Spline Interpolation ---
    SPLINE_ORDER: int = 5  # Degree of B-Spline (Quintic)
    KNOT_SPACING: float = 0.5  # Seconds


config = FlightComposerConfig()
