import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Third-party HTTP clients log every request at INFO; keep them quiet.
    for noisy in ("httpx", "httpcore", "huggingface_hub", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
