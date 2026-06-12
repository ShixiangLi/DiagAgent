"""Temperature calibration for node-level fault models (EPO, eq:epo_temperature).

EPO projects node-model outputs into candidate-root evidence and fuses them by
reliability. Raw LightGBM softmax probabilities are typically over-confident, so
the closure / conflict signals derived from them are miscalibrated. Following the
paper, each node model gets a single positive temperature ``T*`` fit on a held-out
validation split by minimizing the negative log-likelihood of the temperature-
scaled softmax:

    T* = argmin_{T>0}  - Σ_j log softmax(l_j / T)_{y_j}
    ỹ  = softmax(l / T*)

The fitted temperature is stored in the model metadata as ``calibration_temperature``
and applied at inference time by the prediction tools and evidence fusion.

This module is dependency-light (numpy + scipy) and deterministic so it can run in
the same offline pipeline as model training.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable temperature-scaled softmax over the last axis."""
    t = max(float(temperature), 1e-6)
    scaled = logits / t
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def negative_log_likelihood(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    """Mean NLL of the temperature-scaled softmax over (logits, labels)."""
    probs = _softmax_with_temperature(logits, temperature)
    n = logits.shape[0]
    idx = np.arange(n)
    true_probs = np.clip(probs[idx, labels], 1e-12, 1.0)
    return float(-np.mean(np.log(true_probs)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard ECE for diagnostics/reporting (confidence vs accuracy)."""
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    accuracies = (predictions == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple = (0.05, 10.0),
) -> dict:
    """Fit a single temperature ``T*`` by minimizing validation NLL.

    Args:
        logits: ``(n_samples, n_classes)`` raw model margins (pre-softmax).
        labels: ``(n_samples,)`` integer class labels.
        bounds: Search interval for the scalar temperature.

    Returns:
        Dict with ``temperature``, before/after NLL and ECE, and ``n_samples``.
        Falls back to ``temperature=1.0`` when the input is degenerate.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    result = {
        "temperature": 1.0,
        "nll_before": None,
        "nll_after": None,
        "ece_before": None,
        "ece_after": None,
        "n_samples": int(logits.shape[0]) if logits.ndim == 2 else 0,
        "status": "skipped",
    }

    if logits.ndim != 2 or logits.shape[0] < 8 or logits.shape[1] < 2:
        return result

    # Drop labels outside the class range (defensive against remap drift).
    valid = (labels >= 0) & (labels < logits.shape[1])
    if not valid.all():
        logits = logits[valid]
        labels = labels[valid]
    if logits.shape[0] < 8:
        return result

    nll_before = negative_log_likelihood(logits, labels, 1.0)
    ece_before = expected_calibration_error(
        _softmax_with_temperature(logits, 1.0), labels
    )

    # Already well-calibrated models (tiny NLL and ECE on the validation split)
    # have nothing meaningful to calibrate. Fitting a temperature here only lets
    # the optimizer chase a near-separable boundary (e.g. T -> lower bound),
    # which makes the model MORE overconfident — the opposite of what EPO needs
    # for reliable evidence. Keep the identity transform in that regime.
    if nll_before < 0.02 and ece_before < 0.01:
        result.update({
            "temperature": 1.0,
            "nll_before": round(float(nll_before), 6),
            "nll_after": round(float(nll_before), 6),
            "ece_before": round(float(ece_before), 6),
            "ece_after": round(float(ece_before), 6),
            "n_samples": int(logits.shape[0]),
            "status": "already_calibrated",
        })
        return result

    try:
        from scipy.optimize import minimize_scalar

        opt = minimize_scalar(
            lambda t: negative_log_likelihood(logits, labels, t),
            bounds=bounds,
            method="bounded",
        )
        temperature = float(opt.x)
        if not math.isfinite(temperature) or temperature <= 0:
            temperature = 1.0
    except Exception:
        # Coarse grid fallback if scipy optimization fails.
        grid = np.linspace(bounds[0], bounds[1], 100)
        nlls = [negative_log_likelihood(logits, labels, t) for t in grid]
        temperature = float(grid[int(np.argmin(nlls))])

    # Reject boundary-hit temperatures: hitting either optimizer bound signals a
    # degenerate objective (near-separable val set) rather than a genuine
    # calibration optimum. Collapse to identity.
    lo, hi = float(bounds[0]), float(bounds[1])
    if temperature <= lo * 1.001 or temperature >= hi * 0.999:
        temperature = 1.0

    nll_after = negative_log_likelihood(logits, labels, temperature)
    ece_after = expected_calibration_error(
        _softmax_with_temperature(logits, temperature), labels
    )

    # Reject degenerate fits: non-finite NLL/ECE, or any fit that does not
    # strictly improve validation NLL. ``not (a <= b)`` also catches NaN, which
    # a plain ``a > b`` comparison would silently pass.
    improved = (
        math.isfinite(nll_after)
        and math.isfinite(ece_after)
        and nll_after <= nll_before + 1e-9
    )
    if not improved:
        temperature = 1.0
        nll_after = nll_before
        ece_after = ece_before

    result.update({
        "temperature": round(float(temperature), 4),
        "nll_before": round(float(nll_before), 6),
        "nll_after": round(float(nll_after), 6),
        "ece_before": round(float(ece_before), 6),
        "ece_after": round(float(ece_after), 6),
        "n_samples": int(logits.shape[0]),
        "status": "fitted",
    })
    return result


def apply_temperature_to_probabilities(
    probabilities: dict,
    temperature: float,
) -> dict:
    """Re-temperature an existing probability dict (label -> prob).

    LightGBM gives softmax probabilities, not logits, at inference. We recover
    pseudo-logits via ``log(p)`` (sufficient up to an additive constant that the
    softmax cancels) and re-apply the calibrated temperature. With ``T==1.0``
    this is an identity transform.
    """
    if not probabilities or abs(float(temperature) - 1.0) < 1e-6:
        return dict(probabilities)
    labels = list(probabilities.keys())
    probs = np.array([max(float(probabilities[k]), 1e-12) for k in labels], dtype=np.float64)
    pseudo_logits = np.log(probs)
    recalced = _softmax_with_temperature(pseudo_logits.reshape(1, -1), temperature)[0]
    return {label: float(p) for label, p in zip(labels, recalced)}
