"""Choosing a milestone candidate among logged runs by paired bootstrap.

Candidates are given simplest first. Every pair is compared on the same Clients with the paired
bootstrap and the usual verdict (deltas under the tie margin are ties; otherwise the interval must
exclude 0). The winner is the simplest candidate that no other candidate beats, so a more complex
candidate wins only by a clear improvement, never by a tie.
"""

from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from .evaluation import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, TIE_MARGIN, bootstrap_interval, paired_bootstrap, verdict


@dataclass(frozen=True)
class Candidate:
    run_id: str
    model: str
    change: str
    predicted: pd.Series  # per Client, indexed by client_id


@dataclass(frozen=True)
class Scored:
    candidate: Candidate
    macro_f1: float
    low: float
    high: float


@dataclass(frozen=True)
class Pair:
    """`later` (the more complex candidate) against `earlier` (the simpler one)."""

    later: Candidate
    earlier: Candidate
    delta: float
    low: float
    high: float
    verdict: str


@dataclass
class Comparison:
    split: str
    n_clients: int
    scored: list[Scored]
    pairs: list[Pair]
    winner: Candidate
    beaten_by: dict[str, list[str]] = field(default_factory=dict)  # run_id -> run_ids that beat it


def compare_candidates(truth: pd.Series, candidates: list[Candidate], split: str) -> Comparison:
    """Score each candidate against `truth` (indexed by client_id), compare every pair and pick the
    simplest candidate no other candidate beats (the highest-scoring one if every candidate is beaten)."""
    scored = []
    for c in candidates:
        s = bootstrap_interval(truth, c.predicted.reindex(truth.index))
        scored.append(Scored(c, s["macro_f1"], s["low"], s["high"]))
    pairs = []
    beaten_by: dict[str, list[str]] = {c.run_id: [] for c in candidates}
    for earlier, later in combinations(candidates, 2):
        result = paired_bootstrap(truth, later.predicted.reindex(truth.index), earlier.predicted.reindex(truth.index))
        pair = Pair(later, earlier, result["delta"], result["low"], result["high"], verdict(result))
        pairs.append(pair)
        if pair.verdict == "improvement":
            beaten_by[earlier.run_id].append(later.run_id)
        elif pair.verdict == "worse":
            beaten_by[later.run_id].append(earlier.run_id)
    unbeaten = [c for c in candidates if not beaten_by[c.run_id]]
    winner = unbeaten[0] if unbeaten else max(scored, key=lambda s: s.macro_f1).candidate
    return Comparison(split, len(truth), scored, pairs, winner, beaten_by)


def _reason(comparison: Comparison) -> str:
    unbeaten = [c for c, by in comparison.beaten_by.items() if not by]
    if not unbeaten:
        return "every candidate is beaten by another, so the highest macro-F1 wins"
    ranked = {s.candidate.run_id: s.macro_f1 for s in comparison.scored}
    best = max(ranked, key=ranked.get)
    text = (
        "it is the simplest candidate that no other candidate beats"
        if len(unbeaten) > 1 else "no other candidate beats it, and it beats or ties every other"
    )
    if best != comparison.winner.run_id:
        text += (
            f"; {best} scores higher ({ranked[best]:.4f} vs {ranked[comparison.winner.run_id]:.4f}) "
            "but only by a tie, so the simpler candidate is kept"
        )
    return text


def summary_lines(comparison: Comparison) -> list[str]:
    """The comparison as plain text lines for the terminal."""
    lines = [f"compare: {len(comparison.scored)} candidates on {comparison.split} ({comparison.n_clients} Clients)"]
    for i, s in enumerate(comparison.scored, 1):
        lines.append(
            f"  {i}. {s.candidate.run_id} {s.candidate.model:<22} macro-F1 {s.macro_f1:.4f} "
            f"(95% {s.low:.4f} .. {s.high:.4f})"
        )
    for p in comparison.pairs:
        lines.append(
            f"  {p.later.model} vs {p.earlier.model}: delta {p.delta:+.4f} "
            f"(95% {p.low:+.4f} .. {p.high:+.4f}) -> {p.verdict}"
        )
    lines.append(f"  winner: {comparison.winner.run_id} ({comparison.winner.model}): {_reason(comparison)}")
    return lines


def to_markdown(comparison: Comparison, title: str) -> str:
    set_name = "selection set" if comparison.split == "selection" else f"{comparison.split} split"
    out = [
        f"# {title}",
        "",
        f"Scored on the {comparison.n_clients} {set_name} Clients from each run's committed predictions "
        "(`experiments/runs/<run>.csv`). Paired bootstrap over Clients: "
        f"{BOOTSTRAP_SAMPLES} resamples (seed {BOOTSTRAP_SEED}), the same resamples for every candidate. "
        f"A delta under {TIE_MARGIN} is a tie; otherwise the paired 95% interval must exclude 0. "
        "Candidates are listed simplest first; on a tie the simpler candidate is preferred.",
        "",
        "| # | Candidate | Run | Macro-F1 | 95% interval |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(comparison.scored, 1):
        out.append(
            f"| {i} | {s.candidate.model}: {s.candidate.change} | {s.candidate.run_id} | {s.macro_f1:.4f} "
            f"| {s.low:.4f} .. {s.high:.4f} |"
        )
    out += [
        "",
        "Paired comparisons (the more complex candidate minus the simpler one):",
        "",
        "| Candidate | Against | Delta | 95% interval | Verdict |",
        "|---|---|---|---|---|",
    ]
    for p in comparison.pairs:
        out.append(
            f"| {p.later.model} | {p.earlier.model} | {p.delta:+.4f} | {p.low:+.4f} .. {p.high:+.4f} | {p.verdict} |"
        )
    w = comparison.winner
    out += ["", f"**Winner: {w.run_id} ({w.model})**: {_reason(comparison)}.", ""]
    return "\n".join(out)
