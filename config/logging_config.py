# cognitive-swarm-agents/config/logging_config.py
import logging
import sys
from typing import Optional

from config.settings import settings # Importer les settings pour utiliser DEBUG et PYTHON_ENV

def setup_logging(
    level: Optional[str] = None,
    enable_color: bool = True
) -> None:
    """
    Configures the root logger for the application.

    Args:
        level (Optional[str]): The logging level to set.
                               If None, defaults to "DEBUG" if settings.DEBUG is True,
                               otherwise "INFO".
        enable_color (bool): Whether to enable colored logs. Defaults to True.
    """
    if level is None:
        effective_level_str = "DEBUG" if settings.DEBUG else "INFO"
    else:
        effective_level_str = level.upper()

    try:
        effective_level = getattr(logging, effective_level_str)
    except AttributeError:
        print(f"Invalid log level: {effective_level_str}. Defaulting to INFO.")
        effective_level = logging.INFO

    # Base formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Color formatter for console (optional, but nice for development)
    if enable_color and settings.PYTHON_ENV == "development":
        # Basic ANSI escape codes for colors
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        class ColorFormatter(logging.Formatter):
            FORMATS = {
                logging.DEBUG: grey + log_format + reset,
                logging.INFO: grey + log_format + reset, # Or another color like blue
                logging.WARNING: yellow + log_format + reset,
                logging.ERROR: red + log_format + reset,
                logging.CRITICAL: bold_red + log_format + reset,
            }

            def format(self, record: logging.LogRecord) -> str:
                log_fmt = self.FORMATS.get(record.levelno, log_format)
                formatter = logging.Formatter(log_fmt, datefmt=date_format)
                return formatter.format(record)
        formatter = ColorFormatter()
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)

    # Configure root logger
    # logging.basicConfig(level=effective_level, format=log_format, datefmt=date_format, stream=sys.stdout)
    # Instead of basicConfig, we configure the root logger handler directly for more control

    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Remove any existing handlers to avoid duplicate logs if this function is called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add a stream handler to output to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence overly verbose loggers from libraries if needed
    # logging.getLogger("httpx").setLevel(logging.WARNING) # Example for httpx library
    # logging.getLogger("openai").setLevel(logging.WARNING)

    # Log that logging is configured
    # Cannot use root_logger.info here as it might be too early / handler not fully set
    # print(f"Logging configured with level: {effective_level_str}")

# Call setup_logging by default when this module is imported for the first time,
# or allow explicit setup if preferred.
# For simplicity, we can call it here, or expect main.py / script entry points to call it.
# Let's assume it will be called explicitly from the application entry points.
# Example:
# if __name__ == "__main__":
#     setup_logging(level="DEBUG")
#     logging.debug("This is a debug message.")
#     logging.info("This is an info message.")
#     logging.warning("This is a warning message.")
#     logging.error("This is an error message.")
#     logging.critical("This is a critical message.")

#     # Example of a logger in another module
#     logger_example = logging.getLogger("my_module_example")
#     logger_example.info("Info message from my_module_example")