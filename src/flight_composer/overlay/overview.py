"""Overview overlay widget.

Renders every column value from a telemetry row as a vertical text list,
producing a transparent "master‑debug" overlay frame that can be composited
in DaVinci Resolve or similar tools.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from PIL import ImageDraw

from .overlay import Overlay


class OverviewOverlay(Overlay):
    """Full‑dump debug overlay – one line per telemetry column."""

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
        self.name = "overview"

    def render_frame(self, draw: ImageDraw.ImageDraw, row: pd.Series) -> None:
        """Draw every column value for a single frame.

        Parameters
        ----------
        draw:
            A Pillow ``ImageDraw.Draw`` instance bound to the current frame's
            RGBA canvas.
        row:
            A :class:`pandas.Series` representing one row of sequence
            telemetry.  Must contain at least a ``frame_id`` field.
        """
        text_x = 50
        text_y = 50
        line_spacing = self.font_size + 4

        # -- title line -------------------------------------------------------
        frame_id = row["frame_id"]
        title = f"{self.sequence_name} | Frame: {frame_id}"
        self.draw_text_with_outline(
            draw,
            text_x,
            text_y,
            title,
            self.font,
            text_color="yellow",
        )
        text_y = int(text_y + line_spacing)

        # -- actor line -------------------------------------------------------
        self.draw_text_with_outline(
            draw, text_x, int(text_y), f"actor: {self.actor_name}", self.font
        )
        text_y = int(text_y + line_spacing)

        # -- data columns -----------------------------------------------------
        for col in row.index:
            if col == "frame_id":
                continue

            value = row[col]
            if isinstance(value, float):
                formatted = f"{col}: {value:.3f}"
            else:
                formatted = f"{col}: {value}"

            self.draw_text_with_outline(draw, text_x, int(text_y), formatted, self.font)
            text_y = int(text_y + line_spacing)
