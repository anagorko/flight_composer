import dataclasses
import json
import logging
import os
import pathlib
import subprocess

from invoke.context import Context
from invoke.exceptions import Exit
from invoke.runners import Result
from invoke.tasks import task


@dataclasses.dataclass
class InvokeConfig:
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
    CONDA_ENV_NAME = "flight_composer"
    LOGGING_LEVEL = logging.INFO
    CONDA_REQUIREMENTS_FILE = PROJECT_ROOT / "environment.yml"


@task
def _setup_logging(_: Context) -> None:
    """
    Configure logging. Install rich if not available.
    """

    try:
        from rich.console import Console
        from rich.logging import RichHandler
    except ImportError:
        inject_result = subprocess.run(["pipx", "inject", "invoke", "rich"])

        if inject_result.returncode != 0:
            raise Exit(
                "Failed to install required dependencies. Run `pipx inject invoke rich` manually."
            )

        # noinspection PyUnresolvedReferences
        from rich.console import Console

        # noinspection PyUnresolvedReferences
        from rich.logging import RichHandler

    log_format = "\\[[bold]%(name)s[/bold]] %(message)s"
    logging.basicConfig(
        level=InvokeConfig.LOGGING_LEVEL,
        format=log_format,
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(color_system="auto"),
                show_level=True,
                show_path=False,
                enable_link_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
            )
        ],
    )


#
# Conda setup
#

condabin = None


@task(_setup_logging)
def _detect_conda(c: Context, exit_if_not_found: bool = True) -> None:
    """
    Detects the conda installation and sets the global variable `condabin`.
    """

    global condabin

    log = logging.getLogger("detect-conda")

    candidates = [("conda", "Conda"), ("mamba", "Mamba")]

    for bin_name, name in candidates:
        try:
            res = c.run(f"{bin_name} --version", hide=True, warn=True)
            if res and res.return_code == 0:
                condabin = pathlib.Path(bin_name)
                log.info(f"Using global [bold]{name}[/bold] ({res.stdout.strip()})")
                return
        except (FileNotFoundError, Exception):
            continue

    if exit_if_not_found:
        raise Exit("Neither 'mamba' nor 'conda' found. Please install Miniforge.")


def _conda_env_exists(env_name: str) -> bool:
    """
    Checks if a Conda environment exists by parsing the JSON output
    from `conda info --json`.

    Args:
        env_name: The name of the environment to check for.

    Returns:
        True if the environment exists, False otherwise.
    """
    try:
        # Get all conda info in JSON format
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True,
            text=True,
            check=True,  # Raise an exception if conda command fails
        )

        conda_info = json.loads(result.stdout)

        # 'envs' is a list of full paths to the environments
        env_paths = conda_info.get("envs", [])

        # An environment exists if its path is in the list of envs.
        # The path is usually .../envs/{env_name}
        # We check if any of the known environment paths end with our env name.
        # This is more robust than checking the base name of the path.
        for path in env_paths:
            if os.path.basename(path) == env_name:
                return True

        return False

    except (subprocess.CalledProcessError, FileNotFoundError):
        # CalledProcessError means the conda command failed.
        # FileNotFoundError means the conda command was not found.
        print("Conda command not found or failed to execute.")
        return False
    except json.JSONDecodeError:
        print("Failed to parse Conda's JSON output.")
        return False


def conda_run(
    c: Context,
    cmd: str,
    warn: bool = False,
    echo: bool = False,
    pty: bool = False,
    hide: bool = False,
    env: dict[str, str] | None = None,
) -> Result:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    result = c.run(
        f"{condabin} run --no-capture-output -n {InvokeConfig.CONDA_ENV_NAME} {cmd}",
        warn=warn,
        echo=echo,
        pty=pty,
        hide=hide,
        env=run_env,
    )
    assert result is not None
    return result


@task(_setup_logging, _detect_conda)
def create_environment(c: Context) -> None:
    log = logging.getLogger("create-environment")

    requirements_file = InvokeConfig.CONDA_REQUIREMENTS_FILE

    try:
        env_list_result = c.run(f"{condabin} env list --json", hide=True)
        assert env_list_result is not None
        data = json.loads(env_list_result.stdout)
        env_paths = data.get("envs", [])
        installed_envs = [os.path.basename(p) for p in env_paths]
    except json.JSONDecodeError:
        raise Exit(f"Error: Could not parse JSON from '{condabin} env list --json'")
    except Exception as e:
        raise Exit(f"An unexpected error occurred: {e}")

    if InvokeConfig.CONDA_ENV_NAME not in installed_envs:
        log.info(f"Creating Conda env '{InvokeConfig.CONDA_ENV_NAME}'.")
        c.run(
            f"{condabin} create -n {InvokeConfig.CONDA_ENV_NAME} --yes",
            hide=False,
            warn=True,
        )

    if not os.path.exists(requirements_file):
        raise Exit(f"Error: Requirements file not found at '{requirements_file}'")

    log.info(
        f"Installing packages from {requirements_file} into {InvokeConfig.CONDA_ENV_NAME}."
    )
    c.run(f"{condabin} env update -f {requirements_file} --prune")


@task(_detect_conda, _setup_logging)
def install(c: Context) -> None:
    """Install everything from scratch."""

    log = logging.getLogger("install")

    if not _conda_env_exists(InvokeConfig.CONDA_ENV_NAME):
        log.error(
            f"[bold]{InvokeConfig.CONDA_ENV_NAME}[/bold] not found, consult README.md for installation instructions."
        )
        raise Exit(
            f"{InvokeConfig.CONDA_ENV_NAME} not found, consult README.md for installation instructions.."
        )

    with c.cd(InvokeConfig.PROJECT_ROOT):
        # Install GPMF parsing library for telemetry extraction
        conda_run(
            c,
            "python -m pip install gpmf ffmpeg-python",
            pty=True,
        )

        conda_run(
            c,
            "python -m pip install -v -e .",
            pty=True,
        )

    log.info("Installing pre-commit hooks.")
    conda_run(c, "pre-commit install")


@task(_detect_conda, _setup_logging)
def extract_telemetry_csv(
    c: Context, flight_number: str = None, refresh: bool = False
) -> None:
    """
    Extract telemetry data from GoPro MP4 files to CSV format using GPMF parser.

    Args:
        flight_number: Specific flight number to process (optional, processes all if not provided)
        refresh: Force regeneration even if CSV file already exists and is newer
    """
    log = logging.getLogger("extract-telemetry-csv")

    # Build command arguments for the GPMF-based helper script
    script_path = (
        InvokeConfig.PROJECT_ROOT
        / "src"
        / "flight_composer"
        / "scripts"
        / "extract_telemetry_gpmf.py"
    )

    cmd_args = [f"python {script_path}"]

    if flight_number:
        cmd_args.append(f"--flight-number {flight_number}")

    if refresh:
        cmd_args.append("--refresh")

    # Always use verbose mode to match invoke's logging
    cmd_args.append("--verbose")

    command = " ".join(cmd_args)

    try:
        log.info("Running GPMF-based telemetry extraction...")
        result = conda_run(c, command, hide=False)

        if result.return_code != 0:
            raise Exit(
                f"Telemetry extraction failed with exit code {result.return_code}"
            )

    except Exception as e:
        log.error(f"Error running telemetry extraction: {e}")
        raise Exit(f"Telemetry extraction failed: {e}")


@task(_detect_conda, _setup_logging)
def normalize_telemetry(
    c: Context, flight_number: str = None, refresh: bool = False
) -> None:
    """
    Normalize telemetry data through Phase I processing pipeline.

    Args:
        flight_number: Specific flight number to process (optional, processes all if not provided)
        refresh: Force regeneration even if JSON file already exists and is newer
    """
    log = logging.getLogger("normalize-telemetry")

    # Build command arguments for the helper script
    script_path = (
        InvokeConfig.PROJECT_ROOT
        / "src"
        / "flight_composer"
        / "scripts"
        / "normalize_telemetry.py"
    )

    cmd_args = [f"python {script_path}"]

    if flight_number:
        cmd_args.append(f"--flight-number {flight_number}")

    if refresh:
        cmd_args.append("--refresh")

    # Always use verbose mode to match invoke's logging
    cmd_args.append("--verbose")

    command = " ".join(cmd_args)

    try:
        log.info("Running telemetry normalization...")
        result = conda_run(c, command, hide=False)

        if result.return_code != 0:
            raise Exit(
                f"Telemetry normalization failed with exit code {result.return_code}"
            )

    except Exception as e:
        log.error(f"Error running telemetry normalization: {e}")
        raise Exit(f"Telemetry normalization failed: {e}")
