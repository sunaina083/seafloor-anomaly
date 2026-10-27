"""
evaluate.py

Compares the autoencoder's ranking against the naive intensity baseline
using three metrics, computed at several "review fraction" cutoffs
(e.g. top 5%, 10%, 20% of tiles by score):

1. Precision@K: of the tiles reviewed, what fraction actually contain debris?
2. Recall within reviewed fraction: of all true debris tiles, what fraction
   got caught within the reviewed slice?
3. Review effort saved: framed plainly -- "reviewing only X% of tiles still
   catches Y% of debris" vs. having to review 100% exhaustively.

Usage:
    python evaluate.py
"""

import csv
from pathlib import Path

SCORES_ROOT = Path("../data/scores")
REVIEW_FRACTIONS = [0.05, 0.10, 0.20, 0.30]  # top 5%, 10%, 20%, 30% of tiles by score


def load_scores(csv_path, score_column):
    """Load (true_label, score) pairs from a scores CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["true_label"]), float(row[score_column])))
    return rows


def evaluate_ranking(rows, fraction):
    """
    Given (true_label, score) pairs, sort by score descending (most
    anomalous first), take the top `fraction` of tiles, and compute
    precision and recall against the true labels.
    """
    sorted_rows = sorted(rows, key=lambda r: r[1], reverse=True)
    n_total = len(sorted_rows)
    n_reviewed = max(1, int(n_total * fraction))
    reviewed = sorted_rows[:n_reviewed]

    total_debris = sum(1 for label, _ in sorted_rows if label == 1)
    debris_found = sum(1 for label, _ in reviewed if label == 1)

    precision = debris_found / n_reviewed
    recall = debris_found / total_debris if total_debris > 0 else 0.0

    return precision, recall, n_reviewed, debris_found, total_debris


def print_report(method_name, rows):
    print(f"\n=== {method_name} ===")
    for fraction in REVIEW_FRACTIONS:
        precision, recall, n_reviewed, debris_found, total_debris = evaluate_ranking(rows, fraction)
        print(
            f"  Review top {fraction:.0%} ({n_reviewed} tiles): "
            f"precision={precision:.1%}, recall={recall:.1%} "
            f"({debris_found}/{total_debris} debris tiles found)"
        )


def normalize(rows):
    """Min-max normalize scores to 0-1 range, so AE and baseline scores
    (which live on totally different scales) can be fairly averaged."""
    scores = [s for _, s in rows]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0
    return [(label, (s - lo) / span) for label, s in rows]


def main():
    ae_rows = load_scores(SCORES_ROOT / "test_ae_scores.csv", "ae_score")
    baseline_rows = load_scores(SCORES_ROOT / "test_baseline_scores.csv", "baseline_score")

    print_report("Autoencoder (ML method)", ae_rows)
    print_report("Baseline (intensity thresholding)", baseline_rows)

    # ---- Ensemble: average normalized AE + baseline scores ----
    # The two methods appear to catch different things (AE stronger at the
    # very top of the ranking, baseline stronger at broader recall), so
    # combining them may outperform either alone.
    ae_norm = normalize(ae_rows)
    base_norm = normalize(baseline_rows)
    ensemble_rows = [
        (label_a, (score_a + score_b) / 2)
        for (label_a, score_a), (label_b, score_b) in zip(ae_norm, base_norm)
    ]
    print_report("Ensemble (AE + baseline averaged)", ensemble_rows)

       # ---- Plain-English summary using the 10% cutoff as the headline number ----
    ae_precision, ae_recall, _, _, total_debris = evaluate_ranking(ae_rows, 0.10)
    base_precision, base_recall, _, _, _ = evaluate_ranking(baseline_rows, 0.10)
    ens_precision, ens_recall, _, _, _ = evaluate_ranking(ensemble_rows, 0.10)

    print("\n=== Plain-English Summary (reviewing top 10% of tiles) ===")
    print(
        f"An analyst reviewing only the top 10% of tiles, ranked by the "
        f"ensemble method, would catch {ens_recall:.0%} of all debris "
        f"(vs. {ae_recall:.0%} using the autoencoder alone and {base_recall:.0%} "
        f"using the naive brightness baseline alone) -- while skipping 90% "
        f"of the survey entirely."
    )


if __name__ == "__main__":
    main()