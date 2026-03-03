"""Altitude overlay widget.

Renders the glider's altitude (``z_m``) as a two-line display: a smaller
configurable title line (e.g. "Alt [m]") above a much larger numerical
value.  Negative altitudes are visually clamped to zero.
"""

from __future__ import annotations

import math
from typing import Tuple

import pandas as pd
from PIL import ImageDraw, ImageFont

from .overlay import Overlay


class AltitudeOverlay(Overlay):
    """Displays the current altitude with a title label and large value."""

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
        self.title_text: str = self.config_dict.get("title_text", "Alt [m]")
        self.title_color: str = self.config_dict.get("title_color", "gray")
        self.value_color: str = self.config_dict.get("value_color", "white")

        # -- font sizes -------------------------------------------------------
        title_size: int = self.config_dict.get("title_font_size", global_font_size)
        value_size: int = self.config_dict.get("value_font_size", global_font_size * 2)

        # Keep title_size for vertical-offset calculation in render_frame.
        self.title_size = title_size

        try:
            self.title_font = ImageFont.truetype(font_path, title_size)
            self.value_font = ImageFont.truetype(font_path, value_size)
        except IOError:
            self.title_font = ImageFont.load_default()
            self.value_font = ImageFont.load_default()

    # --------------------------------------------------------------------- #

    def render_frame(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """Draw the altitude title and value for a single frame.

        Parameters
        ----------
        draw:
            A Pillow ``ImageDraw.Draw`` instance bound to the current
            frame's RGBA canvas.
        row:
            A :class:`pandas.Series` for the current frame.  Must contain
            a ``z_m`` column representing altitude in metres.
        """
        alt_raw = row["z_m"]

        # Skip the frame when the value is missing.
        try:
            alt_float = float(alt_raw)
        except (TypeError, ValueError):
            return
        if math.isnan(alt_float):
            return

        # Clamp so the display never shows a negative altitude.
        alt = max(0.0, alt_float)

        value_text = f"{alt:.0f}"

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
