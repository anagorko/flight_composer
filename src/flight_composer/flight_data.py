"""Flight data models with WGS 84 → local ENU projection support.

Projection uses Azimuthal Equidistant (aeqd) on the WGS 84 ellipsoid,
producing a Local Tangent Plane in East-North-Up coordinates (metres).
"""

from __future__ import annotations

import datetime
import functools
import pathlib

import numpy as np
import pandas as pd
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
# Airfield model
# ---------------------------------------------------------------------------


class Airfield(pydantic.BaseModel):
    icao_code: str = pydantic.Field(
        default="EPBC",
        min_length=4,
        max_length=4,
        pattern=r"^[A-Z]{4}$",
        description="The 4-letter ICAO airport code (e.g., EPBC)",
    )
    name: str = "Warsaw Babice Airport"
    name_pl: str = "Lotnisko Warszawa-Babice"
    origin: WGS84Coordinate = WGS84Coordinate(
        lat=52.2689025181874, lon=20.91072684036565, alt=106.1
    )


# ---------------------------------------------------------------------------
# Glider models
# ---------------------------------------------------------------------------


class PolarCurve(pydantic.BaseModel):
    """Represents a glider polar using the standard 3-point WinPilot format."""

    v1: float = pydantic.Field(description="Speed 1 in m/s (typically min sink speed)")
    w1: float = pydantic.Field(description="Sink rate 1 in m/s")
    v2: float = pydantic.Field(description="Speed 2 in m/s (typically best L/D speed)")
    w2: float = pydantic.Field(description="Sink rate 2 in m/s")
    v3: float = pydantic.Field(
        description="Speed 3 in m/s (typically fast cruise speed)"
    )
    w3: float = pydantic.Field(description="Sink rate 3 in m/s")

    def get_parabola_coeffs(self) -> tuple[float, float, float]:
        """Solves the system of equations to return a, b, c for w = a*v^2 + b*v + c"""
        A = np.array(
            [
                [self.v1**2, self.v1, 1],
                [self.v2**2, self.v2, 1],
                [self.v3**2, self.v3, 1],
            ]
        )
        B = np.array([self.w1, self.w2, self.w3])
        a, b, c = np.linalg.solve(A, B)
        return a, b, c

    def scale_for_weight(self, old_mass: float, new_mass: float) -> "PolarCurve":
        """Returns a new PolarCurve shifted for a different total mass."""
        factor = math.sqrt(new_mass / old_mass)
        return PolarCurve(
            v1=self.v1 * factor,
            w1=self.w1 * factor,
            v2=self.v2 * factor,
            w2=self.w2 * factor,
            v3=self.v3 * factor,
            w3=self.w3 * factor,
        )


class GliderSpecs(pydantic.BaseModel):
    name: str = "SZD-51 Junior"
    mass_kg: float = pydantic.Field(
        default=340.0, description="Dry empty weight (~240kg) + pilot weight (100kg)"
    )
    wing_area_m2: float = pydantic.Field(
        default=12.51, description="Wing area (S) in square meters"
    )
    wingspan_m: float = pydantic.Field(
        default=15.0, description="Wingspan (b) in meters"
    )
    # Updated defaults scaled for 340kg
    polar: PolarCurve = pydantic.Field(
        default_factory=lambda: PolarCurve(
            v1=20.04,
            w1=-0.60,  # ~72 km/h: Min sink
            v2=22.90,
            w2=-0.66,  # ~82 km/h: Best glide
            v3=34.36,
            w3=-1.65,  # ~123 km/h: High speed cruise
        )
    )

    @property
    def aspect_ratio(self) -> float:
        """AR = b^2 / S"""
        return (self.wingspan_m**2) / self.wing_area_m2


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
    lens_projection: str | None = None
    electronic_stabilization_on: bool | None = None
    electronic_stabilization: str | None = None
    digital_zoom_on: bool | None = None


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

    metadata: GoProMetadata

    data_path: pathlib.Path
    hf_imu_path: pathlib.Path

    @functools.cached_property
    def data(self) -> pd.DataFrame | None:
        """Lazily load the GoPro trajectory DataFrame."""
        if self.data_path.exists():
            return pd.read_parquet(self.data_path)
        return None

    @functools.cached_property
    def hf_imu(self) -> pd.DataFrame | None:
        """Lazily load the GoPro high-frequency IMU DataFrame."""
        if self.hf_imu_path.exists():
            return pd.read_parquet(self.hf_imu_path)
        return None


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

    metadata: IGCMetadata | None = None
    data_path: pathlib.Path | None = None

    @functools.cached_property
    def data(self) -> pd.DataFrame | None:
        """Lazily load the IGC trajectory DataFrame."""
        if self.data_path and self.data_path.exists():
            return pd.read_parquet(self.data_path)
        return None


# ---------------------------------------------------------------------------
# Metadata models
# ---------------------------------------------------------------------------


class WindLayer(pydantic.BaseModel):
    """Wind vector at a specific altitude."""

    altitude_m: float = pydantic.Field(
        description="Altitude in meters (AMSL or AGL depending on reference)"
    )
    speed_ms: float = pydantic.Field(description="Wind speed in m/s")
    direction_deg: float = pydantic.Field(
        description="Wind direction in degrees (where the wind is coming from)"
    )


class FlightMetadata(pydantic.BaseModel):
    """User-defined configuration and metadata parsed from YAML."""

    flight_uid: str
    flight_tag: str

    # References to global registries
    glider_id: str = pydantic.Field(default="szd_51_junior")
    airfield_id: str = pydantic.Field(default="epbc")

    # Official timing (from aeroklub logs)
    # Note: Pydantic will parse ISO 8601 strings into datetime objects.
    # It is highly recommended to include timezone info (e.g., +02:00) in the YAML.
    official_takeoff_time: datetime.datetime | None = pydantic.Field(
        default=None, description="Official takeoff time (should be timezone-aware)"
    )
    official_landing_time: datetime.datetime | None = pydantic.Field(
        default=None, description="Official landing time (should be timezone-aware)"
    )

    # Environment
    qnh_hpa: float | None = pydantic.Field(
        default=None, description="Pressure at sea level (hPa) for altitude correction"
    )
    wind_gradient: list[WindLayer] = pydantic.Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level flight record
# ---------------------------------------------------------------------------


class FlightData(pydantic.BaseModel):
    """Top-level unified flight record."""

    # Configuration read from disk
    metadata: FlightMetadata

    # Resolved at runtime using metadata.glider_id and metadata.airfield_id
    airfield: Airfield
    glider_model: GliderSpecs

    gopro: GoProTelemetry | None = None
    gopro_path: pathlib.Path | None = None

    igc: IGCTelemetry | None = None
    igc_path: pathlib.Path | None = None


# ---------------------------------------------------------------------------
# Top-level flight record
# ---------------------------------------------------------------------------


class FlightDataOld(pydantic.BaseModel):
    flight_uid: str
    flight_tag: str

    airfield: Airfield = Airfield()

    gopro: GoProTelemetry | None = None
    gopro_path: pathlib.Path | None = None

    igc: IGCTelemetry | None = None
    igc_path: pathlib.Path | None = None

    glider_model: GliderSpecs = GliderSpecs()  # TODO: choose appropriate, based on IGC or flight metadata prepared by hand; now it defautls to SZD-51 Junior
