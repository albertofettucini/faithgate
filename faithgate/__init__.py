"""faithgate — local-first LLM faithfulness regression gate."""

__version__ = "0.1.0"

from .ingest.decorator import capture  # noqa: E402  (re-export: the documented public API)

__all__ = ["capture", "__version__"]
