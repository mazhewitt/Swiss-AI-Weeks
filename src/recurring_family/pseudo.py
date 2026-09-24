"""The Pseudo-Label fidelity check: does the Pseudo-Label task at a Shifted Cutoff resemble the real one?

On train, two comparisons for each labeller setting (minimum payments in all, minimum payments before
the Shifted Cutoff, Client churn at the Shifted Cutoff; see `streams.LabellerParams`):

- the Pseudo-Label `none` share against the real train `none` share;
- the milestone-2 rule's macro-F1 against Pseudo-Labels at the Shifted Cutoff (inputs from the
  transactions before it only) against its macro-F1 against the real labels at the real Cutoff.

A setting passes when both gaps are within their tolerances. The check chooses the setting whose
worse gap, relative to its tolerance, is smallest (ties to the simplest: less churn, then fewer
payments before the Shifted Cutoff, then fewer payments), and the check passes when the chosen
setting does.
"""

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .config import CUTOFF, LABELS, NONE_LABEL
from .evaluation import score
from .rules import DEFAULT_NONE_GATE, RulesModel
from .streams import LabellerParams

NONE_SHARE_TOLERANCE = 0.05
RULE_F1_TOLERANCE = 0.05
# the labeller settings the check chooses among: every combination of these, bar the ones that
# repeat another (a minimum in all at or below the minimum before the Shifted Cutoff never binds)
FIDELITY_CANDIDATES = tuple(range(2, 11))  # min_payments
FIDELITY_BEFORE_CANDIDATES = (0, 1, 2, 3)  # min_payments_before
FIDELITY_CHURN_CANDIDATES = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3)  # churn
# the settings `pseudo-labels` and `--pseudo` use unless told otherwise. min_payments 4 was the check's
# choice among min_payments alone (ticket 04). The check now passes on real train at min_payments 3 and
# churn 0.2, but the ranker trained on those Pseudo-Labels only ties this setup, so the defaults stay (ticket 08)
PSEUDO_MIN_PAYMENTS = 4
PSEUDO_MIN_PAYMENTS_BEFORE = 0
PSEUDO_CHURN = 0.0


def milestone2_rule(cutoff: pd.Timestamp = CUTOFF) -> RulesModel:
    """The rule the check measures the task with: E1 plus the `none`-gate, at `cutoff`."""
    return RulesModel(none_gate=DEFAULT_NONE_GATE, cutoff=cutoff)


def search_space(
    min_payments: tuple[int, ...] = FIDELITY_CANDIDATES,
    min_payments_before: tuple[int, ...] = FIDELITY_BEFORE_CANDIDATES,
    churn: tuple[float, ...] = FIDELITY_CHURN_CANDIDATES,
) -> tuple[LabellerParams, ...]:
    """Every combination, less those whose minimum in all never binds (`min_payments` at or below
    `min_payments_before`: they repeat the setting with `min_payments_before + 1`), unless nothing else is left."""
    every = [
        LabellerParams(m, b, c)
        for c, b, m in product(sorted(churn), sorted(min_payments_before), sorted(min_payments))
    ]
    binding = [s for s in every if s.min_payments > s.min_payments_before]
    return tuple(binding or every)


@dataclass(frozen=True)
class Setting:
    """One labeller setting's figures."""

    labeller: LabellerParams
    none_share: float
    rule_scores: dict[str, float]  # the rule against the Pseudo-Labels: macro-F1 and per-label F1
    none_gap: float  # Pseudo-Label none share - real none share
    rule_gap: float  # rule macro-F1 against Pseudo-Labels - against real labels

    @property
    def min_payments(self) -> int:
        return self.labeller.min_payments

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
    pseudo_labels: dict[LabellerParams, pd.Series],
    shifted_rule: pd.Series,
) -> Fidelity:
    """Compare the Pseudo-Label task with the real one over the Clients of `real_labels`.

    `real_rule` is the rule's prediction at the real Cutoff and `shifted_rule` at the Shifted Cutoff;
    `pseudo_labels` maps each labeller setting to its Pseudo-Labels. A Client missing from a
    Pseudo-Label table (no transactions at all) is `none`.
    """
    clients = real_labels.index
    real = real_labels.astype(str)
    real_none_share = float((real == NONE_LABEL).mean())
    real_f1 = score(real, real_rule.reindex(clients).astype(str))["macro_f1"]
    shifted = shifted_rule.reindex(clients).astype(str)
    settings = []
    for labeller, labels in sorted(pseudo_labels.items(), key=lambda item: _simplicity(item[0])):
        pseudo = labels.reindex(clients).astype(object).fillna(NONE_LABEL).astype(str)
        unknown = set(pseudo) - set(LABELS)
        if unknown:
            raise ValueError(f"Pseudo-Labels outside the allowed set: {sorted(unknown)}")
        none_share = float((pseudo == NONE_LABEL).mean())
        scores = score(pseudo, shifted)
        settings.append(Setting(labeller, none_share, scores, none_share - real_none_share, scores["macro_f1"] - real_f1))
    if not settings:
        raise ValueError("the fidelity check needs at least one labeller setting")
    chosen = min(settings, key=lambda s: (s.worse_gap, _simplicity(s.labeller)))
    return Fidelity(real_none_share, real_f1, tuple(settings), chosen)


def _simplicity(labeller: LabellerParams) -> tuple:
    """Less churn first, then fewer payments before the Shifted Cutoff, then fewer payments in all."""
    return labeller.churn, labeller.min_payments_before, labeller.min_payments
