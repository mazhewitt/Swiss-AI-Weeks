"""The blend (`--model blend`): a weighted average of the gated rule's and the Stream Ranker's probabilities.

The rule is one-hot, so its weight is the head start it gives its chosen label; the ranker spreads the
rest across the Client's candidate families and `none`. With the tuned decision layer (E3) on top, the
blend can follow the ranker where it is confident and fall back on the rule where it is not. The rule
weight is chosen on train out-of-fold probabilities (`scripts/blend_weight_sweep.py`), never on valid.
"""

import json
from pathlib import Path

import pandas as pd

from .config import LABELS
from .ranker import RankerModel
from .rules import RulesModel

DEFAULT_RULE_WEIGHT = 0.5


def _part_path(path: Path, part: str) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.{part}{path.suffix}")


class BlendModel:
    name = "blend"

    def __init__(
        self,
        rules: RulesModel | None = None,
        ranker: RankerModel | None = None,
        rule_weight: float = DEFAULT_RULE_WEIGHT,
    ):
        if isinstance(rule_weight, bool) or not 0.0 <= float(rule_weight) <= 1.0:
            raise ValueError(f"rule_weight must be between 0 and 1, not {rule_weight!r}")
        self.rules = RulesModel() if rules is None else rules
        self.ranker = RankerModel() if ranker is None else ranker
        self.rule_weight = float(rule_weight)

    @property
    def variant(self) -> str:
        """What the experiment log adds to the model name: the rule's variant and the rule weight."""
        return f"{self.rules.variant}+w{self.rule_weight:g}"

    def fit(self, transactions: pd.DataFrame, labels: pd.Series) -> "BlendModel":
        self.rules.fit(transactions, labels)
        self.ranker.fit(transactions, labels)
        return self

    def predict_proba(self, transactions: pd.DataFrame, clients: pd.Index) -> pd.DataFrame:
        index = pd.Index(clients, name="client_id")
        rule = self.rules.predict_proba(transactions, index)[list(LABELS)].reindex(index)
        ranker = self.ranker.predict_proba(transactions, index)[list(LABELS)].reindex(index)
        blended = self.rule_weight * rule + (1.0 - self.rule_weight) * ranker
        return blended.div(blended.sum(axis=1), axis=0)

    def save(self, path: Path) -> None:
        """The blend's own file names its parts, saved beside it as `<stem>.rules.json` and `<stem>.ranker.json`."""
        self.rules.save(_part_path(path, "rules"))
        self.ranker.save(_part_path(path, "ranker"))
        payload = {
            "model": self.name,
            "rule_weight": self.rule_weight,
            "rules": _part_path(path, "rules").name,
            "ranker": _part_path(path, "ranker").name,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "BlendModel":
        path = Path(path)
        payload = json.loads(path.read_text())
        rules = RulesModel.load(path.with_name(payload["rules"]))
        ranker = RankerModel.load(path.with_name(payload["ranker"]))
        return cls(rules, ranker, payload["rule_weight"])
