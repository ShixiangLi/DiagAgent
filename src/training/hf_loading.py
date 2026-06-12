"""HuggingFace loading helpers for cache-first server training."""

from __future__ import annotations

import os
from typing import Any, Optional


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_falsey(value: str) -> bool:
    return value.strip().lower() in {"0", "false", "no", "n", "off"}


def cached_from_pretrained(
    loader: Any,
    name_or_path: str,
    *,
    logger: Optional[Any] = None,
    description: str = "artifact",
    **kwargs: Any,
) -> Any:
    """Load a HuggingFace artifact with local cache priority.

    Transformers can touch the network even when model files are already cached,
    for example while probing auxiliary chat-template files.  Server training
    should prefer the local cache and only fall back to online loading when the
    cache is incomplete.
    """
    path = str(name_or_path)
    if os.path.exists(path) or "local_files_only" in kwargs:
        return loader.from_pretrained(path, **kwargs)

    mode = os.environ.get("DIAGAGENT_HF_CACHE_FIRST", "1")
    if _is_falsey(mode):
        return loader.from_pretrained(path, **kwargs)

    local_kwargs = dict(kwargs)
    local_kwargs["local_files_only"] = True
    try:
        if logger is not None:
            logger.info("Loading %s from local HF cache first: %s", description, path)
        return loader.from_pretrained(path, **local_kwargs)
    except Exception as exc:
        if _is_truthy(os.environ.get("DIAGAGENT_HF_LOCAL_ONLY", "0")):
            raise RuntimeError(
                f"Local-only HF load failed for {description} {path}. "
                "Populate the HuggingFace cache or unset DIAGAGENT_HF_LOCAL_ONLY."
            ) from exc
        if logger is not None:
            logger.warning(
                "Local HF cache load failed for %s %s (%s: %s); retrying online",
                description,
                path,
                type(exc).__name__,
                exc,
            )
        return loader.from_pretrained(path, **kwargs)


def resolve_device_map(
    value: Any = "single",
    *,
    logger: Optional[Any] = None,
    description: str = "model",
) -> Any:
    """Resolve a user-friendly device-map setting for inference/RL loading.

    ``device_map="auto"`` can shard a 7B model over all visible GPUs.  That
    saves memory, but multi-turn agent rollouts generate one short action at a
    time and pay heavy cross-GPU communication cost.  On the target server a
    single visible 96GB GPU can hold Qwen2.5-7B + LoRA comfortably, so the
    default ``single`` mode keeps the whole model on visible CUDA device 0.
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "false", "0"}:
        return None
    if lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return lowered
    if lowered in {"single", "primary", "cuda", "cuda:0", "gpu0"}:
        try:
            import torch

            if torch.cuda.is_available():
                index = int(os.environ.get("DIAGAGENT_PRIMARY_CUDA_DEVICE", "0"))
                device_map = {"": index}
                if logger is not None:
                    logger.info(
                        "Resolved %s device_map=%s from setting %r",
                        description,
                        device_map,
                        value,
                    )
                return device_map
        except Exception:
            pass
        return None
    if lowered.isdigit():
        return {"": int(lowered)}
    return value
