import sys
from pathlib import Path
from loguru import logger
import yaml

# Sentinel: prevents handlers from being added more than once across multiple
# calls (e.g., Flask startup + pipeline startup in the same process).
_logging_initialized = False


def setup_logging(config_path: str = "config/config.yaml") -> None:
    """Configures loguru handlers exactly once per process.

    Idempotent: subsequent calls are no-ops, which is safe when multiple
    pipeline entry-points (app.py, main.py) both call this function.
    """
    global _logging_initialized
    if _logging_initialized:
        return

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        log_level = config.get("logging", {}).get("level", "INFO")
        log_file = config.get("logging", {}).get("log_file", "logs/pipeline.log")

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logger.remove()  # Remove the default stderr handler.
        logger.add(sys.stdout, level=log_level, colorize=True)
        logger.add(log_file, level=log_level, rotation="10 MB", retention="30 days")

        _logging_initialized = True
        logger.info("Logging initialized.")

    except FileNotFoundError:
        # Graceful degradation: log to stdout only when config is missing.
        logger.remove()
        logger.add(sys.stdout, level="INFO", colorize=True)
        _logging_initialized = True
        logger.warning(f"Config not found at '{config_path}'. Using default stdout logging.")

    except Exception as e:  # pragma: no cover
        print(f"CRITICAL: Failed to set up logging — {e}")
