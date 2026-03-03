import pathlib
import typing

import pydantic
import yaml

# Adjust this import based on exactly where you put these models!
from flight_composer.flight_data import Airfield, GliderSpecs

# Hardcode the absolute root once
PROJECT_ROOT = pathlib.Path(__file__).parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class FlightComposerDirs(pydantic.BaseModel):
    """Directory configuration, loaded from settings.yaml."""

    PROJECT_ROOT: pathlib.Path = PROJECT_ROOT
    CONFIG: pathlib.Path = CONFIG_DIR
    FLIGHTS: pathlib.Path = CONFIG_DIR / "flights"

    MP4: pathlib.Path
    IGC: pathlib.Path
    PROCESSED_DATA: pathlib.Path
    GPX: pathlib.Path
    TELEMETRY: pathlib.Path
    TRAJECTORY: pathlib.Path
    ACTOR_DATA: pathlib.Path
    GEOJSON: pathlib.Path
    SUBTITLES: pathlib.Path
    PLOTS: pathlib.Path
    UNREAL_ROOT: pathlib.Path
    SEQUENCE_DATA: pathlib.Path
    OVERLAY: pathlib.Path
    FONT_PATH: pathlib.Path

    @pydantic.model_validator(mode="after")
    def resolve_paths(self) -> typing.Self:
        """Automatically convert relative paths into absolute paths based on PROJECT_ROOT."""
        for field_name in self.__class__.model_fields:
            val = getattr(self, field_name)
            if isinstance(val, pathlib.Path) and not val.is_absolute():
                setattr(self, field_name, self.PROJECT_ROOT / val)
        return self


# --- Sequence Config Models ---


class SequenceGlobalConfig(pydantic.BaseModel):
    resolution: list[int] = [1920, 1080]
    font_size: int = 24


class SequenceConfig(pydantic.BaseModel):
    """Represents a loaded sequence overlay configuration."""

    global_opts: SequenceGlobalConfig = pydantic.Field(alias="global")
    # We use a generic dict for widgets since the exact keys vary by type
    widgets: list[dict[str, typing.Any]]


# --- Main Config Registry ---


class FlightComposerConfig(pydantic.BaseModel):
    DIR: FlightComposerDirs

    # Registries
    airfields: dict[str, Airfield] = {}
    gliders: dict[str, GliderSpecs] = {}
    sequences: dict[str, SequenceConfig] = {}


def load_config() -> FlightComposerConfig:
    """Loads all YAML files into the global configuration object."""

    # 1. Load Directories
    settings_path = CONFIG_DIR / "global" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as f:
        settings_data = yaml.safe_load(f) or {}
    dirs = FlightComposerDirs(**settings_data.get("directories", {}))

    # 2. Load Airfields
    with open(CONFIG_DIR / "global" / "airfields.yaml", "r", encoding="utf-8") as f:
        airfields_data = yaml.safe_load(f) or {}
        airfields = {k: Airfield(**v) for k, v in airfields_data.items()}

    # 3. Load Gliders
    with open(CONFIG_DIR / "global" / "gliders.yaml", "r", encoding="utf-8") as f:
        gliders_data = yaml.safe_load(f) or {}
        gliders = {k: GliderSpecs(**v) for k, v in gliders_data.items()}

    # 4. Load Sequence configs dynamically
    sequences = {}
    seq_dir = CONFIG_DIR / "sequences"
    if seq_dir.exists():
        for seq_file in seq_dir.glob("*.yaml"):
            with open(seq_file, "r", encoding="utf-8") as f:
                seq_data = yaml.safe_load(f) or {}
                # stem gives us "Seq_EPBC_Overhead_Intro" without the .yaml
                sequences[seq_file.stem] = SequenceConfig(**seq_data)

    return FlightComposerConfig(
        DIR=dirs, airfields=airfields, gliders=gliders, sequences=sequences
    )


# Instantiate the global config object to be imported by the rest of the app
config = load_config()
