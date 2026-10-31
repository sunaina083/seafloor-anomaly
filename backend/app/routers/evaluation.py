"""
routers/evaluation.py

GET /evaluation -> precision/recall at several review depths, and "review effort
saved", for each scoring method. Computed from the ranked index, which already
holds each test tile's true label and all three scores.
"""

from fastapi import APIRouter

from . import tiles as tiles_router  # reuse the loaded ranked index

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

REVIEW_FRACTIONS = [0.05, 0.10, 0.20, 0.30]   # review the top 5/10/20/30% of tiles
RECALL_TARGETS = [0.50, 0.80, 0.90]           # share of all debris we want to find

# (key used by the frontend, display label, field in the ranked index)
METHODS = [
    ("ensemble", "Ensemble", "ensemble_score"),
    ("ae", "Autoencoder", "ae_score"),
    ("baseline", "Brightness baseline", "baseline_score"),
]


def is_debris(value) -> bool:
    """The label may be stored as 1/0, True/False or text; accept all of them."""
    return str(value).strip().lower() in ("1", "1.0", "true", "anomalous")


def evaluate_method(records: list[dict], score_field: str) -> dict:
    # Rank tiles from highest to lowest score, exactly as an analyst would see them.
    ranked = sorted(records, key=lambda r: r[score_field], reverse=True)
    labels = [is_debris(r["true_label"]) for r in ranked]
    n = len(labels)
    total_debris = sum(labels)

    # cumulative[i] = how many debris tiles are among the top (i + 1) tiles
    cumulative = []
    running = 0
    for is_d in labels:
        running += is_d
        cumulative.append(running)

    at_depth = []
    for fraction in REVIEW_FRACTIONS:
        k = max(1, round(n * fraction))
        found = cumulative[k - 1]
        at_depth.append({
            "fraction": fraction,
            "tiles_reviewed": k,
            "debris_found": found,
            "precision": found / k,
            "recall": found / total_debris if total_debris else 0.0,
        })

    # Review effort saved: how few tiles must be reviewed to reach each recall target?
    effort = []
    for target in RECALL_TARGETS:
        needed = target * total_debris
        k = next((i + 1 for i, c in enumerate(cumulative) if c >= needed), n)
        effort.append({
            "recall_target": target,
            "tiles_reviewed": k,
            "fraction_reviewed": k / n,
            "effort_saved": 1 - k / n,
        })

    return {"at_depth": at_depth, "effort": effort}


@router.get("")
def get_evaluation():
    records = tiles_router._RANKED_TILES
    n = len(records)
    n_debris = sum(is_debris(r["true_label"]) for r in records)

    return {
        "n_tiles": n,
        "n_debris": n_debris,
        "debris_rate": n_debris / n if n else 0.0,
        "methods": [
            {"key": key, "label": label, **evaluate_method(records, field)}
            for key, label, field in METHODS
        ],
    }