"""Flight data models with WGS 84 → local ENU projection support.

Projection uses Azimuthal Equidistant (aeqd) on the WGS 84 ellipsoid,
producing a Local Tangent Plane in East-North-Up coordinates (metres).
"""

from __future__ import annotations

import datetime
import pathlib
from enum import StrEnum

import pydantic

# ---------------------------------------------------------------------------
# Coordinate models
# ---------------------------------------------------------------------------


class WGS84Coordinate(pydantic.BaseModel):
    """A position on the WGS 84 ellipsoid (degrees + metres)."""

    lat: float
    lon: float
    alt: float


class ENUCoordinate(pydantic.BaseModel):
    """A position in a local East-North-Up tangent plane (metres)."""

    east: float
    north: float
    up: float


# ---------------------------------------------------------------------------
# State classification models
# ---------------------------------------------------------------------------


class FlightState(StrEnum):
    PRE_FLIGHT = "PRE_FLIGHT"
    ROLLING_TAKEOFF = "ROLLING_TAKEOFF"
    WINCH = "WINCH"
    TOW = "TOW"
    FREE_FLIGHT = "FREE_FLIGHT"
    ROLLING_LANDING = "ROLLING_LANDING"
    PUSHING_GROUND = "PUSHING_GROUND"


class FlightStateClassification(pydantic.BaseModel):
    """Result of state-machine classification & trimming (pipeline §5).

    Scalar fields ``time_start`` and ``time_end`` define the useful time
    window after trimming extended ``PRE_FLIGHT`` / ``PUSHING_GROUND``
    periods.  The full per-timestamp state series is stored as a CSV file
    on disk (columns: ``timestamp_s``, ``state``) and referenced by
    ``states_path``.
    """

    time_start: float
    """Start of the trimmed time window (seconds, data-stream epoch)."""

    time_end: float
    """End of the trimmed time window (seconds, data-stream epoch)."""

    states_path: pathlib.Path
    """Path to a CSV with columns ``timestamp_s`` and ``state``."""


# ---------------------------------------------------------------------------
# GoPro Telemetry model
# ---------------------------------------------------------------------------


class GoProVideoConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    resolution: str | None = None
    frame_rate_hz: float | None = None
    codec: str | None = None


class GoProAudioConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    sample_rate_hz: int | None = None
    channels: int | None = None


class GoProSettings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    field_of_view: str | None = None
    electronic_stabilization_on: bool | None = None


class GoProMetadata(pydantic.BaseModel):
    """Parses the extracted GoPro JSON metadata."""

    model_config = pydantic.ConfigDict(extra="allow")

    device_name: str
    gps_start_utc: datetime.datetime | None = None
    creation_time: datetime.datetime | None = None
    duration_s: float | None = None

    video: GoProVideoConfig = pydantic.Field(default_factory=GoProVideoConfig)
    audio: GoProAudioConfig = pydantic.Field(default_factory=GoProAudioConfig)
    settings: GoProSettings = pydantic.Field(default_factory=GoProSettings)


class GoProTelemetry(pydantic.BaseModel):
    """GoPro telemetry stream."""

    mp4_path: pathlib.Path | None

    metadata: GoProMetadata | None = None

    trajectory: pathlib.Path | None = None
    hf_imu: pathlib.Path | None = None

    info_tag: str | None = None

    origin: WGS84Coordinate | None = None
    classification: FlightStateClassification | None = None


# ---------------------------------------------------------------------------
# IGC Telemetry model
# ---------------------------------------------------------------------------


class IGCMetadata(pydantic.BaseModel):
    """Parses extracted IGC file header metadata."""

    model_config = pydantic.ConfigDict(extra="allow")

    # Core temporal and spatial data
    date: datetime.date | None = None
    gps_start_utc: datetime.datetime | None = None
    duration_s: float | None = None
    num_fixes: int | None = None
    timezone_offset_h: float | None = None

    # Pilot and Aircraft Info
    pilot: str | None = None
    copilot: str | None = None
    glider_type: str | None = None
    glider_id: str | None = None
    competition_id: str | None = None
    competition_class: str | None = None

    # Hardware and Sensors
    fr_type: str | None = None
    fr_id: str | None = None
    firmware: str | None = None
    hardware: str | None = None
    gps_receiver: str | None = None
    pressure_sensor: str | None = None

    # Datums and References
    gps_datum: str | None = None
    altitude_gps_ref: str | None = None
    altitude_pressure_ref: str | None = None


class IGCTelemetry(pydantic.BaseModel):
    """IGC telemetry stream references and origin."""

    igc_path: pathlib.Path | None

    metadata: IGCMetadata | None = None
    trajectory: pathlib.Path | None = None

    info_tag: str | None = None

    origin: WGS84Coordinate | None = None
    classification: FlightStateClassification | None = None


# ---------------------------------------------------------------------------
# Top-level flight record
# ---------------------------------------------------------------------------


class FlightData(pydantic.BaseModel):
    uid: str
    flight_tag: str | None = None

    mp4_telemetry: MP4Telemetry | None = None
    igc_telemetry: IGCTelemetry | None = None
