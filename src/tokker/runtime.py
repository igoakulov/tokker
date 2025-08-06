import os

# Environment defaults for third‑party libraries (only if not already set)
os.environ.setdefault("TRANSFORMERS_NO_TF_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Optional Google default location; external env or gcloud config may override
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")

# Quiet Transformers internal logs unless there are actual errors.
try:
    from transformers.utils import logging as hf_logging  # type: ignore
    try:
        hf_logging.set_verbosity_error()
    except Exception:
        # transformers may be present but logging setup failed; ignore
        pass
except Exception:
    # transformers may not be installed; ignore
    pass
