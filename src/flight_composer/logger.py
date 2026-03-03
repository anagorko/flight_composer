import logging

import rich.console
import rich.logging


def setup_logging(verbose: bool = False):
    log_format = r"\[[bold]%(name)s[/bold]] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=log_format,
        datefmt="[%X]",
        handlers=[
            rich.logging.RichHandler(
                console=rich.console.Console(color_system="auto"),
                show_level=True,
                show_path=False,
                enable_link_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
            )
        ],
    )
