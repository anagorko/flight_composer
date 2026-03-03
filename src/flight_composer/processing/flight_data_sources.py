import pathlib

import pydantic

from flight_composer.config import config


class FlightDataSources(pydantic.BaseModel):
    flight_tag: str
    gopro_path: pathlib.Path | None = None
    igc_path: pathlib.Path | None = None


def find_data_sources(flight_uid: str) -> FlightDataSources:
    mp4_candidates = sorted(
        list(config.DIR.MP4.glob(f"{flight_uid}_*.mp4"))
        + list(config.DIR.MP4.glob(f"{flight_uid}_*.MP4"))
    )
    igc_candidates = sorted(config.DIR.IGC.glob(f"{flight_uid}_*.igc"))

    if len(mp4_candidates) > 1:
        raise ValueError(
            f"Multiple MP4 sources found for UID {flight_uid}: {mp4_candidates}"
        )
    if len(igc_candidates) > 1:
        raise ValueError(
            f"Multiple IGC sources found for UID {flight_uid}: {igc_candidates}"
        )

    mp4_path = mp4_candidates[0] if mp4_candidates else None
    igc_path = igc_candidates[0] if igc_candidates else None

    if mp4_path is None and igc_path is None:
        raise ValueError(f"No source files found for UID {flight_uid}")

    mp4_tag = mp4_path.stem if mp4_path is not None else None
    igc_tag = igc_path.stem if igc_path is not None else None

    flight_tag = mp4_tag or igc_tag

    if flight_tag is None:
        # To shut up static type checker
        raise ValueError(f"Could not determine flight tag for UID {flight_uid}")

    if mp4_tag and igc_tag and mp4_tag != igc_tag:
        raise ValueError(
            f"Mismatched flight tags for UID {flight_uid}: {mp4_tag} vs {igc_tag}"
        )

    return FlightDataSources(
        flight_tag=flight_tag,
        gopro_path=mp4_path,
        igc_path=igc_path,
    )
