import logging
import sys

_LOG = logging.getLogger("ehr_cd_risk")


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or "ehr_cd_risk")


def configure_logging(level: int = logging.INFO) -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("ehr_cd_risk")
    root.setLevel(level)
    root.addHandler(h)
