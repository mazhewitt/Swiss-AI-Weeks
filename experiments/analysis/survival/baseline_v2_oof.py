"""Train out-of-fold macro-F1 of v2's setup (ranker + none model A+B+ranker_none, Pseudo-Labels) on the
current detector, with the folds of `rf cv`. The baseline for ticket 13. Reads no valid labels."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from none_model_variants import fold_parts, out_of_fold, macro_f1s  # noqa: E402
from recurring_family.cli import Paths
from recurring_family.none_model import NONE_FEATURES

paths = Paths(Path("."))
parts, labels = fold_parts(paths, paths.artifacts / "survival" / "baseline_folds.pkl")
oof = out_of_fold(parts, labels, NONE_FEATURES)
out = Path("experiments/analysis/survival")
oof.to_csv(out / "oof_baseline_v2.csv", index_label="client_id")
res = {"variant": "baseline v2 (ranker+none+pseudo)", **macro_f1s(oof, labels)}
res_r = {"variant": "ranker+pseudo (no none model)", **macro_f1s(out_of_fold(parts, labels), labels)}
print(json.dumps([res, res_r], indent=1))
(out / "baseline.json").write_text(json.dumps([res, res_r], indent=1))
