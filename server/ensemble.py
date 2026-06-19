"""
Recycling Lab Tycoon — Ensemble Voting Algorithm
=================================================
Implements Majority Voting across 5 model predictions with confidence-based
fallback when votes are tied or inconclusive.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Structured result from the ensemble voting process."""

    detected_item: str
    """Winning class label (e.g. ``"plastic"``)."""

    confidence: float
    """Confidence score for the winning prediction (0–1)."""

    all_votes: List[str]
    """Individual votes from each model, in order."""

    vote_counts: Dict[str, int] = field(default_factory=dict)
    """How many votes each class received."""

    method: str = ""
    """Which strategy was used: ``"majority"`` or ``"fallback"``."""

    def to_json(self) -> dict:
        """Serialise to the JSON packet format expected by Unity."""
        return {
            "detected_item": self.detected_item,
            "confidence": round(self.confidence, 4),
            "all_votes": self.all_votes,
        }


def majority_vote(
    predictions: List[Tuple[str, float]],
) -> EnsembleResult:
    """
    Majority Voting Algorithm
    -------------------------

    Parameters
    ----------
    predictions : list of (class_name, confidence)
        One entry per model, in order.

    Returns
    -------
    EnsembleResult

    Algorithm
    ---------
    1. **Scenario 1 — Consensus**:
       The class with the highest vote count wins if it has ≥ 2 votes AND
       is not tied with another class at the same count.

    2. **Scenario 2 — Tie / Disagreement**:
       When no class has a clear majority, fall back to the single prediction
       with the highest confidence score across all models.
    """
    votes = [cls for cls, _ in predictions]
    counter = Counter(votes)
    vote_counts = dict(counter.most_common())

    # ── Scenario 1: clear majority ────────────────────────────────────
    top_count = counter.most_common(1)[0][1]
    top_classes = [cls for cls, cnt in counter.items() if cnt == top_count]

    if top_count >= 2 and len(top_classes) == 1:
        winner = top_classes[0]
        # Average confidence of models that voted for the winner
        winner_confs = [conf for cls, conf in predictions if cls == winner]
        avg_conf = sum(winner_confs) / len(winner_confs)

        return EnsembleResult(
            detected_item=winner,
            confidence=avg_conf,
            all_votes=votes,
            vote_counts=vote_counts,
            method="majority",
        )

    # ── Scenario 2: tie or total disagreement — fallback to max conf ──
    best_idx = max(range(len(predictions)), key=lambda i: predictions[i][1])
    best_cls, best_conf = predictions[best_idx]

    return EnsembleResult(
        detected_item=best_cls,
        confidence=best_conf,
        all_votes=votes,
        vote_counts=vote_counts,
        method="fallback",
    )
