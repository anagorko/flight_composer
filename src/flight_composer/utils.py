"""Utility helpers for the flight_composer package."""

from __future__ import annotations

import json
import logging
import math
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flight_composer.processing.flight_data import FlightData

logger = logging.getLogger(__name__)

# ── FOV code → human-readable name ──────────────────────────────────────────
_FOV_NAMES: dict[str, str] = {
    "W": "Wide",
    "S": "SuperView",
    "L": "Linear",
    "N": "Narrow",
    "M": "Max SuperView",
    "X": "HyperView",
    "LH": "Linear + Horizon Lock",
}

# ── Lens projection → human-readable lens type ──────────────────────────────
_LENS_NAMES: dict[str, str] = {
    "GPRO": "Standard",
    "MLNS": "Max Lens Mod",
}

# ── Resolution height → common marketing label ──────────────────────────────
_RES_LABELS: dict[int, str] = {
    720: "720p",
    1080: "1080p",
    1440: "1440p",
    1520: "2.7K",
    2160: "4K",
    2880: "5K",
    3360: "5.3K",
}

# ── EIS method → friendly stabilisation label ───────────────────────────────
_EIS_LABELS: dict[str, str] = {
    "HS High": "HyperSmooth High",
    "HS Boost": "HyperSmooth Boost",
    "HS Standard": "HyperSmooth",
    "HS Off": "Stab Off",
    "HSON": "HyperSmooth On",
    "ON": "EIS On",
    "OFF": "Stab Off",
}


def _format_resolution(resolution: str, frame_rate_hz: float) -> str:
    """Turn ``'1920x1080'`` + ``59.94`` into ``'1080p60'``."""
    try:
        _w, h_str = resolution.split("x")
        h = int(h_str)
    except (ValueError, AttributeError):
        return resolution

    label = _RES_LABELS.get(h, f"{h}p")
    fps = round(frame_rate_hz)
    return f"{label}{fps}"


def _format_fov(fov_code: str | None, lens_proj: str | None) -> str:
    """Return a concise lens / FOV description."""
    # Max Lens Mod overrides the FOV label
    if lens_proj and lens_proj.upper() == "MLNS":
        return "Max Lens Mod"

    if fov_code:
        return _FOV_NAMES.get(fov_code, fov_code)

    return ""


def _format_stabilisation(eis_on: bool | None, eis_method: str | None) -> str:
    """Return a concise stabilisation description."""
    if eis_on is False or (eis_method and eis_method.upper() in ("OFF", "HS OFF")):
        return "Stab Off"

    if eis_method:
        return _EIS_LABELS.get(eis_method, eis_method)

    if eis_on is True:
        return "EIS On"

    return ""


def get_gopro_overlay_text(flight_data: FlightData) -> str | None:
    """Build a short one-line camera info string from GoPro MP4 metadata.

    The returned string is intended to be displayed as a small on-screen
    overlay when the GoPro recording first appears, giving the viewer a
    quick glance at the recording settings.  This is especially useful for
    flagging stabilisation that causes unnatural panning during turns.

    Parameters
    ----------
    flight_data:
        A :class:`FlightData` instance whose ``mp4_telemetry.metadata``
        may point to a ``*_mp4_meta.json`` file produced by
        :func:`extract_mp4_streams.extract_mp4_telemetry`.

    Returns
    -------
    str | None
        A compact info string such as
        ``"GoPro HERO9 Black · 1080p60 · Wide · HyperSmooth High"``
        or *None* when no metadata file is available.
    """
    meta_path = _resolve_metadata_path(flight_data)
    if meta_path is None:
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read GoPro metadata from %s: %s", meta_path, exc)
        return None

    return _build_overlay_text(meta)


# ── internal helpers ─────────────────────────────────────────────────────────


def _resolve_metadata_path(flight_data: FlightData) -> pathlib.Path | None:
    """Return the metadata JSON path if it exists on disk, else *None*."""
    mp4t = flight_data.mp4_telemetry
    if mp4t is None or mp4t.metadata is None:
        return None

    path = pathlib.Path(mp4t.metadata)
    if not path.is_file():
        logger.debug("Metadata file does not exist: %s", path)
        return None

    return path


def get_flight_map_overlay_text(flight_data: FlightData) -> str | None:
    """Build a short one-line flight info string from IGC metadata.

    The returned string is intended to be displayed as a small on-screen
    overlay when the overhead map and flight trajectory first appear,
    giving the viewer immediate context about the flight details.

    Parameters
    ----------
    flight_data:
        A :class:`FlightData` instance whose ``igc_telemetry.metadata``
        may point to a parsed JSON file containing IGC header data.

    Returns
    -------
    str | None
        A compact info string such as
        "Andrzej Nagorko · SZD 51-1 Junior (SP-3303) · 2025-05-08 · Flight Time: 6m 23s"
        or *None* when no metadata file is available.
    """
    meta_path = _resolve_igc_metadata_path(flight_data)
    if meta_path is None:
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read IGC metadata from %s: %s", meta_path, exc)
        return None

    return _build_flight_map_overlay_text(meta)


# ── internal helpers (flight map overlay) ────────────────────────────────────


def _resolve_igc_metadata_path(flight_data: FlightData) -> pathlib.Path | None:
    """Return the IGC metadata JSON path if it exists on disk, else *None*."""
    igc_t = flight_data.igc_telemetry
    if igc_t is None or igc_t.metadata is None:
        return None

    path = pathlib.Path(igc_t.metadata)
    if not path.is_file():
        logger.debug("IGC metadata file does not exist: %s", path)
        return None

    return path


def _format_duration(seconds: float) -> str:
    """Convert *seconds* to a human-readable duration string.

    Examples: ``"6m 23s"``, ``"1h 15m"``, ``"0m 0s"``.
    """
    total = int(math.floor(seconds))
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)

    if h > 0:
        # For flights over an hour, drop the seconds for brevity
        return f"{h}h {m:02d}m" if m else f"{h}h 0m"
    return f"{m}m {s:02d}s" if m else f"0m {s}s"


def _build_flight_map_overlay_text(meta: dict) -> str | None:
    """Compose the flight-map overlay string from the parsed IGC metadata."""
    parts: list[str] = []

    # 1. Pilot name
    pilot = meta.get("pilot")
    if pilot:
        parts.append(pilot)

    # 2. Glider type (with optional registration)
    glider_type = meta.get("glider_type")
    glider_id = meta.get("glider_id")
    if glider_type and glider_id:
        parts.append(f"{glider_type} ({glider_id})")
    elif glider_type:
        parts.append(glider_type)
    elif glider_id:
        parts.append(glider_id)

    # 3. Date
    date = meta.get("date")
    if date:
        parts.append(date)

    # 4. Flight duration
    duration_s = meta.get("duration_s")
    if duration_s is not None:
        parts.append(f"Flight Time: {_format_duration(duration_s)}")

    if not parts:
        return None

    return " · ".join(parts)


def _build_overlay_text(meta: dict) -> str:
    """Compose the overlay string from the parsed metadata dict."""
    parts: list[str] = []

    # 1. Camera model
    device = meta.get("device_name", "")
    if device:
        parts.append(f"GoPro {device}")

    # 2. Resolution + frame-rate
    video = meta.get("video", {})
    res = video.get("resolution")
    fps = video.get("frame_rate_hz")
    if res and fps:
        parts.append(_format_resolution(res, fps))

    # 3. Lens / FOV
    settings = meta.get("settings", {})
    fov_label = _format_fov(
        settings.get("field_of_view"),
        settings.get("lens_projection"),
    )
    if fov_label:
        parts.append(fov_label)

    # 4. Stabilisation
    stab_label = _format_stabilisation(
        settings.get("electronic_stabilization_on"),
        settings.get("electronic_stabilization"),
    )
    if stab_label:
        parts.append(stab_label)

    # 5. Digital Zoom (affects stabilisation behaviour)
    if settings.get("digital_zoom_on", False):
        parts.append("Digital Zoom")

    if not parts:
        return "GoPro (settings unknown)"

    return " · ".join(parts)
