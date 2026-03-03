"""Speedup overlay widget.

Renders the current playback-speed ratio (derived from ``speedup_raw``)
as a single text label that can be composited over video.  The value can
optionally be snapped to the nearest power of two for a cleaner display.
"""

from __future__ import annotations

import math
from typing import Tuple

import pandas as pd
from PIL import ImageDraw

from .overlay import Overlay


class SpeedupOverlay(Overlay):
    """Displays the real-time speedup multiplier (e.g. ``2x >>``)."""

    def __init__(
        self,
        sequence_name: str,
        actor_name: str,
        config_dict: dict,
        resolution: Tuple[int, int],
        font_path: str,
        global_font_size: int,
    ):
        super().__init__(
            sequence_name,
            actor_name,
            config_dict,
            resolution,
            font_path,
            global_font_size,
        )

        res_x, res_y = self.resolution
        self.x: int = self._parse_coord(self.config_dict.get("x", 0), res_x)
        self.y: int = self._parse_coord(self.config_dict.get("y", 0), res_y)
        self.color: str = self.config_dict.get("color", "white")
        self.do_round: bool = self.config_dict.get("round", True)

    def render_frame(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """Draw the speedup label for a single frame.

        Parameters
        ----------
        draw:
            A Pillow ``ImageDraw.Draw`` instance bound to the current
            frame's RGBA canvas.
        row:
            A :class:`pandas.Series` for the current frame.  Must contain
            a ``speedup_raw`` column (added by the preprocessing step).
        """
        speedup_raw_val = row["speedup_raw"]

        # Nothing to display when the value is missing.
        try:
            speedup_raw = float(speedup_raw_val)
        except (TypeError, ValueError):
            return
        if math.isnan(speedup_raw):
            return

        if self.do_round:
            # Clamp to a small positive floor to avoid log2(0) errors.
            clamped = max(speedup_raw, 0.001)
            speedup = 2 ** round(math.log2(clamped))
        else:
            speedup = round(speedup_raw, 2)

        # Format with directional arrows.
        if speedup > 1.05:
            text = f"{speedup:g}x >>"
        elif speedup < 0.95:
            text = f"{speedup:g}x <<"
        else:
            text = f"{speedup:g}x"

        self.draw_text_with_outline(
            draw,
            self.x,
            self.y,
            text,
            self.font,
            text_color=self.color,
        )
