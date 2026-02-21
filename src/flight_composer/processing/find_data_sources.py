from flight_composer.config import config
from flight_composer.processing.flight_data import (
    FlightData,
    IGCTelemetry,
    MP4Telemetry,
)


def find_data_sources(flight_data: FlightData) -> FlightData:
    flight_uid = flight_data.uid

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

    mp4_tag = mp4_path.stem if mp4_path else None
    igc_tag = igc_path.stem if igc_path else None

    if mp4_tag and igc_tag and mp4_tag != igc_tag:
        raise ValueError(
            f"Mismatched flight tags for UID {flight_uid}: {mp4_tag} vs {igc_tag}"
        )

    flight_tag = mp4_tag or igc_tag

    mp4_telemetry = MP4Telemetry(mp4_path=mp4_path) if mp4_path else None
    igc_telemetry = IGCTelemetry(igc_path=igc_path) if igc_path else None

    return flight_data.model_copy(
        update=dict(
            flight_tag=flight_tag,
            mp4_telemetry=mp4_telemetry,
            igc_telemetry=igc_telemetry,
        )
    )
