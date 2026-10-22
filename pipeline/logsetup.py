#!/usr/bin/env python3
# =============================================================================
#  logsetup.py  -  shared logging configuration
# -----------------------------------------------------------------------------
#  Every script logs to BOTH the console and a file under config.LOGS_DIR, so
#  nothing is lost when a run is unattended. The log file is named after the
#  run id when one is given (so logs line up with the result JSON/CSV of the
#  same run), otherwise after the component + a UTC timestamp.
#
#  logs/ lives under the mounted volume (reachable from the host) and is
#  gitignored. Use get_logger(component, run_id) once at the top of a script.
# =============================================================================
import logging
import os
import sys
from datetime import UTC, datetime

sys.path.insert(0, "/work")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def get_logger(component, run_id=None, level=logging.INFO):
    """Return a logger writing to console and logs/<run_id|component-stamp>.log.

    Also ensures the output and logs folders exist on first run, so a clean
    clone (where these dirs are gitignored and absent) just works.
    """
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # The script name always leads the filename; the run id (when given) is
    # appended for correlation with that run's result files.
    if run_id:
        tag = f"{component}__{run_id}"
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        tag = f"{component}_{stamp}"
    logfile = os.path.join(config.LOGS_DIR, f"{tag}.log")

    logger = logging.getLogger(f"rsei.{component}")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("logging to %s", logfile)
    return logger
