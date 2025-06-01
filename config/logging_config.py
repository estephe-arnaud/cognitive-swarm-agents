# config/logging_config.py
import logging
import sys
from typing import Optional

from config.settings import settings # Importer les settings pour utiliser DEBUG et PYTHON_ENV

def setup_logging(
    level: Optional[str] = None,
    enable_color: bool = True # Permet de désactiver la couleur si nécessaire via l'appel
) -> None:
    """
    Configures the root logger for the application.

    Args:
        level (Optional[str]): The logging level to set.
                                If None, defaults to "DEBUG" if settings.DEBUG is True,
                                otherwise "INFO".
        enable_color (bool): Whether to enable colored logs. Defaults to True.
                             Effective only if PYTHON_ENV is "development".
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

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Color formatter for console (optional, but nice for development)
    # Couleur activée seulement si enable_color=True ET settings.PYTHON_ENV=="development"
    use_colored_logs = enable_color and settings.PYTHON_ENV == "development"

    if use_colored_logs:
        # Standard ANSI escape codes for colors
        # MODIFICATION: Utilisation de codes ANSI plus standards
        grey = "\x1b[90m"      # Grey (souvent affiché comme noir brillant)
        blue = "\x1b[34m"     # Blue (pour INFO, par exemple)
        yellow = "\x1b[33m"    # Yellow
        red = "\x1b[31m"       # Red
        bold_red = "\x1b[31;1m" # Bold Red (ou \x1b[1;31m)
        reset = "\x1b[0m"      # Reset all attributes

        class ColorFormatter(logging.Formatter):
            FORMATS = {
                logging.DEBUG: grey + log_format + reset,
                logging.INFO: blue + log_format + reset, # Changé en bleu pour INFO
                logging.WARNING: yellow + log_format + reset,
                logging.ERROR: red + log_format + reset,
                logging.CRITICAL: bold_red + log_format + reset,
            }

            def format(self, record: logging.LogRecord) -> str:
                log_fmt = self.FORMATS.get(record.levelno, log_format) # Fallback sur log_format si niveau non trouvé
                formatter = logging.Formatter(log_fmt, datefmt=date_format)
                return formatter.format(record)
        formatter = ColorFormatter()
    else:
        formatter = logging.Formatter(log_format, datefmt=date_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(effective_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Si vous voulez explicitement désactiver la couleur pour certains loggers de bibliothèques
    # qui pourraient avoir leur propre formatage couleur (bien que ce soit rare pour le logging standard)
    # logging.getLogger("some_verbose_library").propagate = False # ou ajouter un handler spécifique

    # Log que le logging est configuré (ne sera pas coloré si la config couleur n'est pas encore appliquée)
    # print(f"Logging configured with level: {effective_level_str}, Color enabled: {use_colored_logs}")