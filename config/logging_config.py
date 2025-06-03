"""
MAKERS Logging Configuration Module

This module provides centralized logging configuration for the MAKERS system.
It supports:
- Configurable log levels
- Colored console output (in development)
- Standardized log format
- Custom formatters

The logging configuration can be customized through environment variables
and the settings module.
"""

import logging
import sys
from typing import Optional

from config.settings import settings # Importer les settings pour utiliser DEBUG et PYTHON_ENV

class ColorFormatter(logging.Formatter):
    """
    Custom formatter that adds color to log messages based on their level.
    Only used in development environment.
    
    Colors:
    - DEBUG: Grey
    - INFO: Blue
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Bold Red
    """
    
    # ANSI escape codes for colors
    COLORS = {
        logging.DEBUG: "\x1b[90m",      # Grey
        logging.INFO: "\x1b[34m",       # Blue
        logging.WARNING: "\x1b[33m",    # Yellow
        logging.ERROR: "\x1b[31m",      # Red
        logging.CRITICAL: "\x1b[31;1m", # Bold Red
    }
    RESET = "\x1b[0m"  # Reset all attributes

    def __init__(self, fmt: str, datefmt: str):
        """
        Initialize the color formatter.
        
        Args:
            fmt: Log message format string
            datefmt: Date format string
        """
        super().__init__(fmt, datefmt)
        self.FORMATS = {
            level: color + fmt + self.RESET
            for level, color in self.COLORS.items()
        }

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with appropriate color.
        
        Args:
            record: The log record to format
            
        Returns:
            str: The formatted log message
        """
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)

def setup_logging(
    level: Optional[str] = None,
    enable_color: bool = True # Permet de désactiver la couleur si nécessaire via l'appel
) -> None:
    """
    Configure the root logger for the application.
    
    This function:
    1. Sets up the root logger with the specified level
    2. Configures a console handler with appropriate formatting
    3. Optionally enables colored output in development environment
    
    Args:
        level: The logging level to set. If None, uses DEBUG if settings.DEBUG is True,
              otherwise INFO.
        enable_color: Whether to enable colored logs. Only effective in development
                     environment.
    """
    # Determine effective log level
    if level is None:
        effective_level_str = "DEBUG" if settings.DEBUG else "INFO"
    else:
        effective_level_str = level.upper()

    try:
        effective_level = getattr(logging, effective_level_str)
    except AttributeError:
        print(f"Invalid log level: {effective_level_str}. Defaulting to INFO.")
        effective_level = logging.INFO

    # Configure log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Determine if colored output should be used
    use_colored_logs = enable_color and settings.PYTHON_ENV == "development"

    # Create appropriate formatter
    formatter = (
        ColorFormatter(log_format, date_format)
        if use_colored_logs
        else logging.Formatter(log_format, datefmt=date_format)
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    # Clear existing handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Log configuration
    root_logger.debug(
        f"Logging configured with level: {effective_level_str}, "
        f"Color enabled: {use_colored_logs}"
    )

    # Si vous voulez explicitement désactiver la couleur pour certains loggers de bibliothèques
    # qui pourraient avoir leur propre formatage couleur (bien que ce soit rare pour le logging standard)
    # logging.getLogger("some_verbose_library").propagate = False # ou ajouter un handler spécifique