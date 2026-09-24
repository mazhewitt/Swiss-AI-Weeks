"""The Pseudo-Label fidelity check: does the Pseudo-Label task at a Shifted Cutoff resemble the real one?

On train, two comparisons for each minimum-payments setting of the labeller:

- the Pseudo-Label `none` share against the real train `none` share;
- the milestone-2 rule's macro-F1 against Pseudo-Labels at the Shifted Cutoff (inputs from the
  transactions before it only) against its macro-F1 against the real labels at the real Cutoff.

A setting passes when both gaps are within their tolerances. The check chooses the setting whose
worse gap, relative to its tolerance, is smallest (ties to the smaller setting), and the check
passes when the chosen setting does.
"""

from dataclasses import dataclass

import pandas as pd

from .config import CUTOFF, LABELS, NONE_LABEL
from .evaluation import score
from .rules import DEFAULT_NONE_GATE, RulesModel

NONE_SHARE_TOLERANCE = 0.05
RULE_F1_TOLERANCE = 0.05
# the labeller's minimum-payments settings the check chooses among
FIDELITY_CANDIDATES = tuple(range(2, 11))
# the setting the fidelity check chose on the real train data (the slow real-data test records it);
# `pseudo-labels` uses it unless told otherwise
PSEUDO_MIN_PAYMENTS = 4


def milestone2_rule(cutoff: pd.Timestamp = CUTOFF) -> RulesModel:
    """The rule the check measures the task with: E1 plus the `none`-gate, at `cutoff`."""
    return RulesModel(none_gate=DEFAULT_NONE_GATE, cutoff=cutoff)


@dataclass(frozen=True)
class Setting:
    """One minimum-payments setting's figures."""

    min_payments: int
    none_share: float
    rule_scores: dict[str, float]  # the rule against the Pseudo-Labels: macro-F1 and per-label F1
    none_gap: float  # Pseudo-Label none share - real none share
    rule_gap: float  # rule macro-F1 against Pseudo-Labels - against real labels

    @property
    def rule_macro_f1(self) -> float:
        return self.rule_scores["macro_f1"]

    @property
    def passed(self) -> bool:
        return abs(self.none_gap) <= NONE_SHARE_TOLERANCE and abs(self.rule_gap) <= RULE_F1_TOLERANCE

    @property
    def worse_gap(self) -> float:
        return max(abs(self.none_gap) / NONE_SHARE_TOLERANCE, abs(self.rule_gap) / RULE_F1_TOLERANCE)


@dataclass(frozen=True)
class Fidelity:
    real_none_share: float
    real_rule_macro_f1: float
    settings: tuple[Setting, ...]
    chosen: Setting

    @property
    def passed(self) -> bool:
        return self.chosen.passed


def fidelity_check(
    real_labels: pd.Series,
    real_rule: pd.Series,
    pseudo_labels: dict[int, pd.Series],
    shifted_rule: pd.Series,
) -> Fidelity:
    """Compare the Pseudo-Label task with the real one over the Clients of `real_labels`.

    `real_rule` is the rule's prediction at the real Cutoff and `shifted_rule` at the Shifted Cutoff;
    `pseudo_labels` maps each minimum-payments setting to its Pseudo-Labels. A Client missing from a
    Pseudo-Label table (no transactions at all) is `none`.
    """
    clients = real_labels.index
    real = real_labels.astype(str)
    real_none_share = float((real == NONE_LABEL).mean())
    real_f1 = score(real, real_rule.reindex(clients).astype(str))["macro_f1"]
    shifted = shifted_rule.reindex(clients).astype(str)
    settings = []
    for m, labels in sorted(pseudo_labels.items()):
        pseudo = labels.reindex(clients).astype(object).fillna(NONE_LABEL).astype(str)
        unknown = set(pseudo) - set(LABELS)
        if unknown:
            raise ValueError(f"Pseudo-Labels outside the allowed set: {sorted(unknown)}")
        none_share = float((pseudo == NONE_LABEL).mean())
        scores = score(pseudo, shifted)
        settings.append(Setting(m, none_share, scores, none_share - real_none_share, scores["macro_f1"] - real_f1))
    if not settings:
        raise ValueError("the fidelity check needs at least one minimum-payments setting")
    chosen = min(settings, key=lambda s: (s.worse_gap, s.min_payments))
    return Fidelity(real_none_share, real_f1, tuple(settings), chosen)
