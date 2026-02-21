"""
Flight File Finder Module

This module provides utilities for finding flight files by flight number using
the standardized naming convention: <flight_number>_<descriptive_name>.<extension>

The module searches across multiple directories and handles different flight number
formats (7, 07, 007) to locate the appropriate files.
"""

import re
from pathlib import Path
from typing import List, Optional, Union

from flight_composer import config


def normalize_flight_number(flight_number: Union[str, int]) -> str:
    """
    Normalize flight number to string format.

    Args:
        flight_number: Flight number as string or integer

    Returns:
        str: Flight number as string
    """
    return str(flight_number)


def generate_flight_number_patterns(flight_number: Union[str, int]) -> List[str]:
    """
    Generate different flight number patterns to search for.

    For flight number "7", generates patterns: ["7", "07", "007"]
    For flight number "42", generates patterns: ["42", "042"]

    Args:
        flight_number: Flight number as string or integer

    Returns:
        List[str]: List of flight number patterns to search for
    """
    flight_num_str = normalize_flight_number(flight_number)
    flight_num_int = int(flight_num_str)

    patterns = []

    # Add the original number
    patterns.append(flight_num_str)

    # Add zero-padded versions (2 and 3 digits)
    if flight_num_int < 100:
        patterns.append(f"{flight_num_int:02d}")  # 2-digit zero-padded
    if flight_num_int < 1000:
        patterns.append(f"{flight_num_int:03d}")  # 3-digit zero-padded

    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)

    return unique_patterns


def find_flight_file(
    flight_number: Union[str, int],
    extension: str,
    search_dirs: Optional[List[Path]] = None,
) -> Optional[Path]:
    """
    Find a flight file by flight number and extension.

    Searches for files matching the pattern:
    <flight_number>_<descriptive_name>.<extension>

    Args:
        flight_number: Flight number (e.g., 7, "07", 142)
        extension: File extension without dot (e.g., "gpx", "igc", "mp4")
        search_dirs: Optional list of directories to search.
                    Defaults to config.FLIGHT_SEARCH_DIRS

    Returns:
        Optional[Path]: Path to the found file, or None if not found

    Example:
        >>> find_flight_file(7, "gpx")
        PosixPath('/path/to/TrajectoryData/07_niskie_ladowanie.gpx')
        >>> find_flight_file(7, "mp4")  # Will find both .mp4 and .MP4 files
        PosixPath('/path/to/GoPro/07_flight_recording.MP4')
    """
    if search_dirs is None:
        search_dirs = config.FLIGHT_SEARCH_DIRS

    # Generate flight number patterns
    patterns = generate_flight_number_patterns(flight_number)

    # Clean extension (remove leading dot if present)
    clean_extension = extension.lstrip(".")

    # Search in each directory
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Try each flight number pattern
        for pattern in patterns:
            # Create regex pattern: <flight_number>_<anything>.<extension> (case-insensitive)
            regex_pattern = f"^{re.escape(pattern)}_.*\\.{re.escape(clean_extension)}$"

            # Search for matching files (case-insensitive extension matching)
            for file_path in search_dir.iterdir():
                if file_path.is_file() and re.match(
                    regex_pattern, file_path.name, re.IGNORECASE
                ):
                    return file_path

    return None


def find_all_flight_files(
    flight_number: Union[str, int], search_dirs: Optional[List[Path]] = None
) -> List[Path]:
    """
    Find all files for a given flight number across all extensions.

    Args:
        flight_number: Flight number (e.g., 7, "07", 142)
        search_dirs: Optional list of directories to search.
                    Defaults to config.FLIGHT_SEARCH_DIRS

    Returns:
        List[Path]: List of paths to all found files for the flight

    Example:
        >>> find_all_flight_files(7)
        [PosixPath('/.../07_niskie_ladowanie.gpx'),
         PosixPath('/.../07_niskie_ladowanie.igc')]
    """
    if search_dirs is None:
        search_dirs = config.FLIGHT_SEARCH_DIRS

    # Generate flight number patterns
    patterns = generate_flight_number_patterns(flight_number)

    found_files = []

    # Search in each directory
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Try each flight number pattern
        for pattern in patterns:
            # Create regex pattern: <flight_number>_<anything>.<any_extension>
            regex_pattern = f"^{re.escape(pattern)}_.*\\..+$"

            # Search for matching files
            for file_path in search_dir.iterdir():
                if file_path.is_file() and re.match(regex_pattern, file_path.name):
                    if file_path not in found_files:  # Avoid duplicates
                        found_files.append(file_path)

    return sorted(found_files)


def get_flight_number_from_filename(filename: str) -> Optional[str]:
    """
    Extract flight number from a filename following the naming convention.

    Args:
        filename: Filename to parse

    Returns:
        Optional[str]: Flight number if found, None otherwise

    Example:
        >>> get_flight_number_from_filename("07_niskie_ladowanie.gpx")
        "07"
        >>> get_flight_number_from_filename("invalid_name.txt")
        None
    """
    # Pattern: <digits>_<anything>.<extension>
    match = re.match(r"^(\d+)_.*\..+$", filename)
    if match:
        return match.group(1)
    return None


def list_available_flights(search_dirs: Optional[List[Path]] = None) -> List[str]:
    """
    List all available flight numbers in the search directories.

    Args:
        search_dirs: Optional list of directories to search.
                    Defaults to config.FLIGHT_SEARCH_DIRS

    Returns:
        List[str]: Sorted list of unique flight numbers

    Example:
        >>> list_available_flights()
        ['07', '08', '11', '26', '142']
    """
    if search_dirs is None:
        search_dirs = config.FLIGHT_SEARCH_DIRS

    flight_numbers = set()

    # Search in each directory
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Check each file
        for file_path in search_dir.iterdir():
            if file_path.is_file():
                flight_num = get_flight_number_from_filename(file_path.name)
                if flight_num:
                    flight_numbers.add(flight_num)

    return sorted(flight_numbers, key=lambda x: int(x))


def validate_flight_file_name(filename: str) -> bool:
    """
    Validate if a filename follows the flight file naming convention.

    Args:
        filename: Filename to validate

    Returns:
        bool: True if filename follows the convention, False otherwise

    Example:
        >>> validate_flight_file_name("07_niskie_ladowanie.gpx")
        True
        >>> validate_flight_file_name("random_file.txt")
        False
    """
    # Pattern: <digits>_<text>.<extension>
    pattern = r"^\d+_[a-zA-Z0-9_]+\.[a-zA-Z0-9]+$"
    return bool(re.match(pattern, filename))


def suggest_flight_file_name(
    flight_number: Union[str, int], description: str, extension: str
) -> str:
    """
    Suggest a filename following the flight file naming convention.

    Args:
        flight_number: Flight number
        description: Human-readable description
        extension: File extension

    Returns:
        str: Suggested filename

    Example:
        >>> suggest_flight_file_name(7, "Low Landing", "gpx")
        "07_low_landing.gpx"
    """
    # Normalize flight number to 2-digit zero-padded format for consistency
    flight_num_int = int(normalize_flight_number(flight_number))
    padded_flight_num = f"{flight_num_int:02d}"

    # Clean up description: lowercase, replace spaces with underscores,
    # remove special characters except underscores
    clean_description = re.sub(r"[^a-zA-Z0-9\s_]", "", description.lower())
    clean_description = re.sub(r"\s+", "_", clean_description.strip())

    # Clean extension
    clean_extension = extension.lstrip(".")

    return f"{padded_flight_num}_{clean_description}.{clean_extension}"


def find_gopro_file(
    flight_number: Union[str, int], search_dirs: Optional[List[Path]] = None
) -> Optional[Path]:
    """
    Find a GoPro MP4 file by flight number (case-insensitive extension).

    Searches for files matching the pattern:
    <flight_number>_<descriptive_name>.mp4 or .MP4

    Args:
        flight_number: Flight number (e.g., 7, "07", 142)
        search_dirs: Optional list of directories to search.
                    Defaults to [config.GOPRO_DIR] if available, otherwise config.FLIGHT_SEARCH_DIRS

    Returns:
        Optional[Path]: Path to the found MP4 file, or None if not found

    Example:
        >>> find_gopro_file(7)
        PosixPath('/srv/samba/share/GoProFlights/07_flight_recording.MP4')
    """
    if search_dirs is None:
        # Default to GOPRO_DIR if available, otherwise use standard search dirs
        try:
            from pathlib import Path

            gopro_dir = Path(config.GOPRO_DIR)
            search_dirs = (
                [gopro_dir] if gopro_dir.exists() else config.FLIGHT_SEARCH_DIRS
            )
        except (AttributeError, ImportError):
            search_dirs = config.FLIGHT_SEARCH_DIRS

    # Try both mp4 and MP4 extensions
    for ext in ["mp4", "MP4"]:
        result = find_flight_file(flight_number, ext, search_dirs)
        if result:
            return result

    return None
