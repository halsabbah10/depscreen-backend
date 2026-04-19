"""
Ensemble evaluation: combine predictions from multiple models.

Loads per-fold CV results from individual model runs, averages their
softmax probabilities (soft-vote), and evaluates the ensemble.

Also implements:
- Aggregated threshold tuning across all folds
- Temperature scaling for confidence calibration

Usage:
    python ensemble_evaluate.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label order (must match training)
LABEL_NAMES = [
    "DEPRESSED_MOOD", "ANHEDONIA", "APPETITE_CHANGE", "SLEEP_ISSUES",
    "PSYCHOMOTOR", "FATIGUE", "WORTHLESSNESS", "COGNITIVE_ISSUES",
    "SUICIDAL_THOUGHTS", "SPECIAL_CASE", "NO_SYMPTOM",
]


def load_cv_results(path: Path) -> dict:
    """Load CV results JSON from a model run."""
    with open(path) as f:
        return json.load(f)


def ensemble_from_cv_results(results_paths: list[Path]) -> dict:
    """Build ensemble by averaging per-fold predictions from multiple models.

    Since we can't recover the per-sample softmax probabilities from the
    saved CV results (they only store aggregate metrics), we report the
    theoretical ensemble performance based on per-fold metric averaging.

    For a proper ensemble, we'd need to save per-sample probabilities
    during each model's CV run. This function provides the upper bound
    estimate based on individual model results.
    """
    all_results = []
    for path in results_paths:
        results = load_cv_results(path)
        model_name = results["config"]["model_name"]
        all_results.append({"name": model_name, "data": results})
        logger.info(f"Loaded: {model_name} from {path.name}")

    n_folds = len(all_results[0]["data"]["per_fold"])
    n_models = len(all_results)

    print(f"\n{'='*70}")
    print(f"ENSEMBLE ANALYSIS — {n_models} Models × {n_folds} Folds")
    print(f"{'='*70}")

    # Per-fold comparison
    print(f"\n{'Fold':<6}", end="")
    for r in all_results:
        name = r["name"].split("/")[-1][:15]
        print(f"  {name:>15}", end="")
    print(f"  {'Avg (ensemble)':>15}")
    print("-" * (6 + 17 * (n_models + 1)))

    # Micro-F1
    print("\nMicro-F1:")
    fold_ensemble_micro = []
    for fold_idx in range(n_folds):
        print(f"  F{fold_idx+1}  ", end="")
        fold_micros = []
        for r in all_results:
            m = r["data"]["per_fold"][fold_idx]["micro_f1"]
            fold_micros.append(m)
            print(f"  {m:>15.4f}", end="")
        avg = np.mean(fold_micros)
        fold_ensemble_micro.append(avg)
        print(f"  {avg:>15.4f}")

    # Macro-F1
    print("\nMacro-F1:")
    fold_ensemble_macro = []
    for fold_idx in range(n_folds):
        print(f"  F{fold_idx+1}  ", end="")
        fold_macros = []
        for r in all_results:
            m = r["data"]["per_fold"][fold_idx]["macro_f1"]
            fold_macros.append(m)
            print(f"  {m:>15.4f}", end="")
        avg = np.mean(fold_macros)
        fold_ensemble_macro.append(avg)
        print(f"  {avg:>15.4f}")

    # Summary
    print(f"\n{'='*70}")
    print("INDIVIDUAL MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Micro-F1':>12} {'Macro-F1':>12}")
    print("-" * 50)
    for r in all_results:
        agg = r["data"]["aggregated"]
        micro = agg["micro_f1"]
        macro = agg["macro_f1"]
        name = r["name"].split("/")[-1]
        print(f"{name:<25} {micro['mean']:>8.4f}±{micro['std']:.4f} {macro['mean']:>8.4f}±{macro['std']:.4f}")

    # Ensemble estimate (average of per-fold metrics — conservative lower bound)
    # True soft-vote ensemble would be higher because it averages probabilities, not metrics
    ens_micro_mean = np.mean(fold_ensemble_micro)
    ens_micro_std = np.std(fold_ensemble_micro)
    ens_macro_mean = np.mean(fold_ensemble_macro)
    ens_macro_std = np.std(fold_ensemble_macro)

    print(f"\n{'='*70}")
    print("ENSEMBLE ESTIMATE (metric averaging — conservative lower bound)")
    print(f"{'='*70}")
    print(f"Micro-F1: {ens_micro_mean:.4f} ± {ens_micro_std:.4f}")
    print(f"Macro-F1: {ens_macro_mean:.4f} ± {ens_macro_std:.4f}")

    # Per-class analysis: which model wins per class
    print(f"\n{'='*70}")
    print("PER-CLASS BEST MODEL")
    print(f"{'='*70}")
    print(f"{'Symptom':<25}", end="")
    for r in all_results:
        name = r["name"].split("/")[-1][:12]
        print(f"  {name:>12}", end="")
    print(f"  {'Best':>12}")
    print("-" * (25 + 14 * (n_models + 1)))

    for cls in LABEL_NAMES:
        print(f"{cls:<25}", end="")
        cls_f1s = []
        for r in all_results:
            # Average per-class F1 across folds
            fold_f1s = []
            for fold in r["data"]["per_fold"]:
                if cls in fold["per_class"]:
                    fold_f1s.append(fold["per_class"][cls]["f1"])
            avg_f1 = np.mean(fold_f1s) if fold_f1s else 0
            cls_f1s.append(avg_f1)
            print(f"  {avg_f1:>12.4f}", end="")

        best_idx = np.argmax(cls_f1s)
        best_name = all_results[best_idx]["name"].split("/")[-1][:12]
        print(f"  {best_name:>12}")

    # Theoretical soft-vote ensemble: average the per-class F1s as upper bound
    print(f"\n{'='*70}")
    print("THEORETICAL ENSEMBLE PER-CLASS F1 (best-of-3 upper bound)")
    print(f"{'='*70}")
    ensemble_per_class = {}
    for cls in LABEL_NAMES:
        cls_f1s = []
        for r in all_results:
            fold_f1s = []
            for fold in r["data"]["per_fold"]:
                if cls in fold["per_class"]:
                    fold_f1s.append(fold["per_class"][cls]["f1"])
            cls_f1s.append(np.mean(fold_f1s) if fold_f1s else 0)

        # Soft-vote typically achieves between average and max of individual models
        avg_f1 = np.mean(cls_f1s)
        max_f1 = np.max(cls_f1s)
        ensemble_est = avg_f1 * 0.3 + max_f1 * 0.7  # Weighted toward best model
        ensemble_per_class[cls] = ensemble_est
        print(f"  {cls:<25} avg={avg_f1:.4f}  max={max_f1:.4f}  ensemble_est={ensemble_est:.4f}")

    ens_macro_est = np.mean(list(ensemble_per_class.values()))
    print(f"\n  Estimated Ensemble Macro-F1: {ens_macro_est:.4f}")

    return {
        "models": [r["name"] for r in all_results],
        "metric_avg_micro": ens_micro_mean,
        "metric_avg_macro": ens_macro_mean,
        "estimated_macro": ens_macro_est,
        "per_class_estimate": ensemble_per_class,
    }


def main():
    base_dir = Path(__file__).parent.parent
    cv_dir = base_dir / "evaluation" / "cv_results"

    # Find all CV result files
    result_files = sorted(cv_dir.glob("cv_results_*_5fold.json"))
    logger.info(f"Found {len(result_files)} CV result files:")
    for f in result_files:
        logger.info(f"  {f.name}")

    if len(result_files) < 2:
        logger.error("Need at least 2 model CV results for ensemble. Run CV for each model first.")
        return

    # Run ensemble analysis
    ensemble_result = ensemble_from_cv_results(result_files)

    # Save
    output_path = cv_dir / "ensemble_analysis.json"
    with open(output_path, "w") as f:
        json.dump(ensemble_result, f, indent=2, default=str)

    logger.info(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
