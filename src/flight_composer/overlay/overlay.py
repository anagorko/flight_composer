from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

from PIL import ImageFont


class Overlay(ABC):
    def __init__(
        self,
        sequence_name: str,
        actor_name: str,
        config_dict: dict,
        resolution: Tuple[int, int],
        font_path: str,
        global_font_size: int,
    ):
        """
        Base class for all telemetry rendering widgets.

        :param sequence_name: Name of the active MRQ sequence (e.g., 'Seq_EPBC_Overhead_Intro')
        :param actor_name: Name of the actor exported from Unreal (e.g., 'GliderPlaybackFlight07')
        :param config_dict: Dictionary containing widget-specific configuration from YAML
        :param resolution: Tuple of (width, height) in pixels
        :param font_path: Filesystem path to a TrueType font file
        :param global_font_size: Default font size from the global YAML config
        """
        self.sequence_name = sequence_name
        self.actor_name = actor_name
        self.config_dict = config_dict
        self.resolution = resolution
        self.name: str = self.config_dict.get("name", "unnamed")

        # Per-widget font: honour a widget-level override, fall back to global.
        self.font_size: int = self.config_dict.get("font_size", global_font_size)

        try:
            self.font = ImageFont.truetype(font_path, self.font_size)
        except IOError:
            self.font = ImageFont.load_default()

    def _parse_coord(self, val: Union[str, int, float], max_val: int) -> int:
        """
        Parses a coordinate that could be an exact pixel value or a percentage.
        Example: '50%' of a 1920 max_val -> 960.
        """
        if isinstance(val, str) and val.endswith("%"):
            percentage = float(val.strip("%")) / 100.0
            return int(percentage * max_val)
        return int(val)

    @staticmethod
    def draw_text_with_outline(
        draw,
        x: int,
        y: int,
        text: str,
        font: Any,
        text_color: str = "white",
        outline_color: str = "black",
        stroke_width: int = 2,
        **kwargs,
    ):
        """
        Helper to draw legible text against any video background.
        Accepts **kwargs to allow passing arguments like `anchor="rs"`.
        """
        draw.text(
            (x, y),
            str(text),
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=outline_color,
            **kwargs,
        )

    @abstractmethod
    def render_frame(self, draw, row):
        """
        Renders the widget's visual representation for a single frame.

        :param draw: PIL ImageDraw.Draw instance connected to the current frame's canvas
        :param row: Pandas Series representing the telemetry data for this exact frame
        """
        pass
