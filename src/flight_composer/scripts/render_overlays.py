"""Render overlay image sequences for all configured widgets.

Reads per-frame telemetry from a sequence CSV and widget definitions from a
companion YAML file, then writes one transparent PNG **per widget per frame**.
Each widget produces its own image sequence (e.g. ``overview_000001.png``,
``speed_000001.png``) that can be layered independently in DaVinci Resolve.

Usage
-----
    pixi run python -m flight_composer.scripts.render_overlays \
        --sequence Seq_EPBC_Overhead_Intro [--actor Glider]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
import pandera.pandas as pa
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

from flight_composer.config import config
from flight_composer.overlay.altitude import AltitudeOverlay
from flight_composer.overlay.overview import OverviewOverlay
from flight_composer.overlay.preprocessing import preprocess_telemetry
from flight_composer.overlay.speed import SpeedOverlay
from flight_composer.overlay.speedup import SpeedupOverlay

FONT_NAME = "IBMPlexMono-Regular.ttf"

# ---------------------------------------------------------------------------
# Pandera schema for the incoming sequence CSV
# ---------------------------------------------------------------------------

SEQUENCE_SCHEMA = pa.DataFrameSchema(
    columns={
        "frame_id": pa.Column(str, pa.Check.str_matches(r"^\d+$")),
        "timestamp_s": pa.Column(float),
        "video_time_s": pa.Column(float),
        "x_m": pa.Column(float),
        "y_m": pa.Column(float),
        "z_m": pa.Column(float),
        "vx_ms": pa.Column(float),
        "vy_ms": pa.Column(float),
        "vz_ms": pa.Column(float),
        "v_mag_ms": pa.Column(float),
        "ax_ms2": pa.Column(float),
        "ay_ms2": pa.Column(float),
        "az_ms2": pa.Column(float),
        "load_g": pa.Column(float),
        "yaw_rad": pa.Column(float),
        "pitch_rad": pa.Column(float),
        "roll_rad": pa.Column(float),
        "phase": pa.Column(str),
    },
    strict=False,
    coerce=True,
)

# Mapping from YAML widget *type* string to the concrete Overlay subclass.
WIDGET_REGISTRY: dict[str, type] = {
    "altitude": AltitudeOverlay,
    "speed": SpeedOverlay,
    "speedup": SpeedupOverlay,
}


# ---------------------------------------------------------------------------
# Main render routine
# ---------------------------------------------------------------------------


def _resolve_actor(sequence: str, actor: str | None) -> str:
    """Return an explicit *actor* or auto-detect it from the CSV files on disk.

    Raises
    ------
    FileNotFoundError
        No CSV matching ``{sequence}_*.csv`` was found.
    ValueError
        Multiple CSVs matched – the user must specify ``--actor`` explicitly.
    """
    if actor is not None:
        return actor

    matches = list(config.DIR.SEQUENCE_DATA.glob(f"{sequence}_*.csv"))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No telemetry CSV found for sequence '{sequence}' in "
            f"{config.DIR.SEQUENCE_DATA}"
        )
    if len(matches) > 1:
        found = ", ".join(m.stem for m in sorted(matches))
        raise ValueError(
            f"Multiple actor CSVs detected for sequence '{sequence}': {found}. "
            f"Please provide --actor explicitly."
        )

    return matches[0].stem[len(sequence) + 1 :]


def render_overlays(
    sequence: str,
    actor: str | None,
    limit: int | None = None,
    widget_names: list[str] | None = None,
) -> None:
    """Generate one transparent PNG per widget per frame for *sequence*."""

    resolved_actor = _resolve_actor(sequence, actor)
    print(f"Actor: {resolved_actor}")

    # -- resolve paths --------------------------------------------------------
    csv_path: pathlib.Path = (
        config.DIR.SEQUENCE_DATA / f"{sequence}_{resolved_actor}.csv"
    )
    yaml_path: pathlib.Path = config.DIR.SEQUENCE_CONFIG / f"{sequence}.yaml"
    output_dir: pathlib.Path = config.DIR.OVERLAY / sequence

    output_dir.mkdir(parents=True, exist_ok=True)

    # -- ingest CSV -----------------------------------------------------------
    print(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"frame_id": str})
    df = SEQUENCE_SCHEMA.validate(df)
    df = preprocess_telemetry(df)

    # -- optional frame limit -------------------------------------------------
    if limit is not None:
        df = df.head(limit)
        print(f"LIMIT APPLIED: Rendering only the first {limit} frames.")

    # -- ingest YAML ----------------------------------------------------------
    print(f"Reading YAML: {yaml_path}")
    with open(yaml_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    global_cfg = cfg.get("global", {})
    resolution: list[int] = global_cfg.get("resolution", [1920, 1080])
    font_path: str = str(config.DIR.FONT_PATH / FONT_NAME)
    global_font_size: int = global_cfg.get("font_size", 24)

    res_x, res_y = resolution[0], resolution[1]

    # -- build the list of active widgets -------------------------------------
    active_widgets = []

    # The overview widget is always present.
    active_widgets.append(
        OverviewOverlay(
            sequence,
            resolved_actor,
            {"name": "overview"},
            (res_x, res_y),
            font_path,
            global_font_size,
        )
    )

    for widget_cfg in cfg.get("widgets", []):
        widget_type = widget_cfg.get("type")
        widget_cls = WIDGET_REGISTRY.get(widget_type)
        if widget_cls is None:
            print(f"WARNING: Unknown widget type '{widget_type}', skipping.")
            continue
        active_widgets.append(
            widget_cls(
                sequence,
                resolved_actor,
                widget_cfg,
                (res_x, res_y),
                font_path,
                global_font_size,
            )
        )

    # -- optional widget filter -----------------------------------------------
    if widget_names is not None:
        active_widgets = [w for w in active_widgets if w.name in widget_names]

    if not active_widgets:
        print("No active widgets after filtering. Nothing to render.")
        sys.exit(0)

    active_names = ", ".join(w.name for w in active_widgets)
    print(f"Active widgets: {active_names}")

    # -- render loop ----------------------------------------------------------
    print(
        f"Rendering {len(df)} frames × {len(active_widgets)} widgets at {res_x}×{res_y} …"
    )

    for widget in tqdm(active_widgets, desc="Overall Progress (Widgets)", position=0):
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Rendering: {widget.name}",
            unit="frame",
            position=1,
            leave=False,
        ):
            frame_id: str = str(row["frame_id"])

            img = Image.new("RGBA", (res_x, res_y), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            widget.render_frame(draw, row)

            out_path = output_dir / f"{widget.name}_{frame_id}.png"
            img.save(out_path, compress_level=1)

    total_files = len(df) * len(active_widgets)
    print(f"\nDone. {total_files} overlay frames written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render overlay PNGs from a sequence CSV and widget config.",
    )
    parser.add_argument(
        "--sequence",
        required=True,
        help="Sequence name (e.g. Seq_EPBC_Overhead_Intro)",
    )
    parser.add_argument(
        "--actor",
        required=False,
        default=None,
        help="Actor name exported from Unreal (e.g. Glider). "
        "Auto-detected when only one CSV matches the sequence.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Render only the first N frames (useful for quick previews).",
    )
    parser.add_argument(
        "--widgets",
        nargs="+",
        default=None,
        help="Render only the listed widget names (e.g. --widgets overview altitude).",
    )
    args = parser.parse_args()
    render_overlays(
        args.sequence, args.actor, limit=args.limit, widget_names=args.widgets
    )


if __name__ == "__main__":
    main()
