"""Utility to download and cache YAGO 4.5 entities as the default knowledge base."""

import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

YAGO_URL = "https://yago-knowledge.org/data/yago4.5/yago-entities.jsonl.zip"
YAGO_DIR = Path("data/yago")
YAGO_PATH = YAGO_DIR / "yago-entities.jsonl"


def ensure_yago_kb() -> str:
    """Return the path to the YAGO entities JSONL, downloading it if needed."""
    if YAGO_PATH.exists():
        logger.info("YAGO KB already on disk at %s", YAGO_PATH)
        return str(YAGO_PATH)

    YAGO_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = YAGO_DIR / "yago-entities.jsonl.zip"

    logger.info("Downloading YAGO 4.5 entities from %s ...", YAGO_URL)
    urllib.request.urlretrieve(YAGO_URL, zip_path)
    logger.info("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("yago-entities.jsonl", YAGO_DIR)

    zip_path.unlink()
    logger.info("YAGO KB ready at %s", YAGO_PATH)
    return str(YAGO_PATH)
