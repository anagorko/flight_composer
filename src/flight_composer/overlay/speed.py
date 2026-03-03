"""Speed overlay widget.

Renders the glider's ground speed (``v_mag_ms``) as a two-line display: a
smaller configurable title/unit line (e.g. "km/h") above a much larger
numerical value.  Supports ``km/h`` and ``m/s`` units with automatic
conversion.  Negative speeds are visually clamped to zero.
"""

from __future__ import annotations

import math
from typing import Tuple

import pandas as pd
from PIL import ImageDraw, ImageFont

from .overlay import Overlay


class SpeedOverlay(Overlay):
    """Displays the current ground speed with a title label and large value."""

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

        # -- positional / styling config with sensible defaults ---------------
        self.x: int = self._parse_coord(self.config_dict.get("x", 50), resolution[0])
        self.y: int = self._parse_coord(self.config_dict.get("y", 50), resolution[1])
        self.unit: str = self.config_dict.get("unit", "km/h")
        self.title_text: str = self.config_dict.get("title_text", self.unit)
        self.title_color: str = self.config_dict.get("title_color", "gray")
        self.value_color: str = self.config_dict.get("value_color", "white")

        # -- font sizes -------------------------------------------------------
        self.title_size: int = self.config_dict.get("title_font_size", global_font_size)
        value_size: int = self.config_dict.get("value_font_size", global_font_size * 2)

        try:
            self.title_font = ImageFont.truetype(font_path, self.title_size)
            self.value_font = ImageFont.truetype(font_path, value_size)
        except IOError:
            self.title_font = ImageFont.load_default()
            self.value_font = ImageFont.load_default()

    # --------------------------------------------------------------------- #

    def render_frame(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """Draw the speed title and value for a single frame.

        Parameters
        ----------
        draw:
            A Pillow ``ImageDraw.Draw`` instance bound to the current
            frame's RGBA canvas.
        row:
            A :class:`pandas.Series` for the current frame.  Must contain
            a ``v_mag_ms`` column representing ground speed in m/s.
        """
        speed_raw = row["v_mag_ms"]

        # Skip the frame when the value is missing.
        try:
            speed_float = float(speed_raw)
        except (TypeError, ValueError):
            return
        if math.isnan(speed_float):
            return

        # Convert to the requested unit.
        if self.unit == "km/h":
            speed_val = speed_float * 3.6
        else:
            speed_val = speed_float

        # Clamp so the display never shows a negative speed.
        speed_val = max(0.0, float(speed_val))

        value_text = f"{speed_val:.0f}"

        # -- title line (smaller) ---------------------------------------------
        self.draw_text_with_outline(
            draw,
            self.x,
            self.y,
            self.title_text,
            self.title_font,
            text_color=self.title_color,
        )

        # -- value line (larger, directly below) ------------------------------
        value_y = self.y + self.title_size + 5
        self.draw_text_with_outline(
            draw,
            self.x,
            value_y,
            value_text,
            self.value_font,
            text_color=self.value_color,
        )
