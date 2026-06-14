from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictionSummary:
    label: str
    confidence: float
    runner_up_confidence: float
    margin: float


class ProbabilitySmoother:
    """Simple EMA smoother over class probabilities for steadier live decisions."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self._state: torch.Tensor | None = None

    def reset(self):
        self._state = None

    def update(self, probs: torch.Tensor) -> torch.Tensor:
        x = probs.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
        if self._state is None or self.alpha >= 0.999:
            self._state = x.clone()
        else:
            alpha = max(0.0, min(self.alpha, 1.0))
            self._state = alpha * x + (1.0 - alpha) * self._state

        total = float(self._state.sum().item())
        if total > 0.0:
            self._state = self._state / total
        return self._state.clone()


def summarize_probabilities(probs: torch.Tensor, classes: list[str]) -> PredictionSummary:
    x = probs.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if x.numel() != len(classes):
        raise RuntimeError(
            "Probability/class count mismatch. Re-run training or refresh model artifacts."
        )

    topk = x.topk(k=min(2, x.numel()))
    idx = int(topk.indices[0].item())
    confidence = float(topk.values[0].item())
    runner_up = float(topk.values[1].item()) if topk.values.numel() > 1 else 0.0
    return PredictionSummary(
        label=classes[idx],
        confidence=confidence,
        runner_up_confidence=runner_up,
        margin=confidence - runner_up,
    )
