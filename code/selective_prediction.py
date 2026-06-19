#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selective Prediction with Accepted Error Control.

This script implements Algorithm 1 from the SafeGround paper. It repeatedly
shuffles ``total.json`` into independent calibration and test subsets,
calibrates an uncertainty threshold on the calibration subset, and evaluates
the accepted error rate on the test subset.

Examples:
    python selective_prediction.py \
        --total_file /path/to/total.json \
        --output_dir /path/to/output \
        --delta_cal 0.05 \
        --target_error_rate 0.30 \
        --n_splits 100 \
        --test_ratio 0.6

    python selective_prediction.py \
        --total_file /path/to/total.json \
        --output_dir /path/to/output \
        --delta_cal 0.05 \
        --delta_split 0.05 \
        --target_error_range 0.30:0.50:0.04 \
        --n_splits 1000 \
        --test_ratio 0.6 \
        --seed 42
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_total_data(total_file: str) -> Tuple[List[Dict], List[str]]:
    """Load valid samples and uncertainty method names from ``total.json``."""
    with open(total_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    results = [result for result in data.get("results", []) if "error" not in result]
    if not results:
        raise ValueError(f"No valid samples found in {total_file}")

    config = data.get("config", {})
    methods = config.get("methods", []) or config.get("uncertainty_methods", [])
    if not methods:
        uncertainty = results[0].get(
            "uncertainties", results[0].get("uncertainty", {})
        )
        methods = (
            list(uncertainty.keys())
            if isinstance(uncertainty, dict)
            else ["uncertainty"]
        )
    if not methods:
        raise ValueError("No uncertainty methods found in the input data")

    logger.info("Loaded %d valid samples", len(results))
    logger.info("Uncertainty methods: %s", methods)
    return results, methods


def get_uncertainties_by_method(results: Sequence[Dict], method: str) -> List[float]:
    """Extract one finite uncertainty value per sample for a method."""
    uncertainties: List[float] = []
    for index, result in enumerate(results):
        uncertainty = result.get("uncertainties", result.get("uncertainty", {}))
        if isinstance(uncertainty, dict):
            if method not in uncertainty:
                raise KeyError(f"Sample {index} has no uncertainty value for {method!r}")
            value = uncertainty[method]
        else:
            value = uncertainty
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(
                f"Sample {index} has an invalid uncertainty for {method!r}: {value!r}"
            )
        uncertainties.append(float(value))
    return uncertainties


def get_hits(results: Sequence[Dict]) -> List[bool]:
    """Extract correctness labels (``correct`` or the legacy ``hit`` field)."""
    hits: List[bool] = []
    for index, result in enumerate(results):
        if "correct" in result:
            value = result["correct"]
        elif "hit" in result:
            value = result["hit"]
        else:
            raise KeyError(f"Sample {index} has neither 'correct' nor 'hit'")
        hits.append(bool(value))
    return hits


def clopper_pearson_upper_bound(errors: int, accepted: int, delta: float) -> float:
    """Return the one-sided ``1-delta`` binomial upper confidence bound."""
    if accepted < 0 or errors < 0 or errors > accepted:
        raise ValueError("Require 0 <= errors <= accepted")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between 0 and 1")
    if accepted == 0 or errors == accepted:
        return 1.0
    return float(stats.beta.ppf(1.0 - delta, errors + 1, accepted - errors))


def calibrate_threshold(
    uncertainties: Sequence[float],
    hits: Sequence[bool],
    delta_cal: float,
    target_error_rate: float,
) -> Dict:
    """Calibrate a threshold using Algorithm 1's ordered testing rule.

    Candidate thresholds are traversed in increasing uncertainty order. The
    search stops at the first candidate whose Clopper--Pearson bound exceeds
    the target. This fixed-sequence condition is essential: selecting any
    feasible candidate after a failed candidate is not Algorithm 1.
    """
    if len(uncertainties) != len(hits):
        raise ValueError("uncertainties and hits must have equal lengths")
    if not uncertainties:
        raise ValueError("Calibration data must not be empty")
    if not 0.0 <= target_error_rate <= 1.0:
        raise ValueError("target_error_rate must be in [0, 1]")

    ordered = sorted(zip(uncertainties, hits), key=lambda item: item[0])
    accepted = 0
    errors = 0
    best: Optional[Dict] = None
    cursor = 0

    # A threshold accepts all tied scores, so ties are tested as one candidate.
    while cursor < len(ordered):
        threshold = float(ordered[cursor][0])
        while cursor < len(ordered) and ordered[cursor][0] == threshold:
            accepted += 1
            errors += int(not ordered[cursor][1])
            cursor += 1

        upper_bound = clopper_pearson_upper_bound(errors, accepted, delta_cal)
        if upper_bound > target_error_rate:
            break

        best = {
            "threshold": threshold,
            "accepted_samples": accepted,
            "errors": errors,
            "empirical_accepted_error_rate": errors / accepted,
            "upper_bound": upper_bound,
            "coverage": accepted / len(ordered),
            "abstention_rate": 1.0 - accepted / len(ordered),
            "attainable": True,
        }

    if best is None:
        return {
            "threshold": None,
            "accepted_samples": 0,
            "errors": 0,
            "empirical_accepted_error_rate": 0.0,
            "upper_bound": 1.0,
            "coverage": 0.0,
            "abstention_rate": 1.0,
            "attainable": False,
        }
    return best


def evaluate_split(
    cal_results: Sequence[Dict],
    test_results: Sequence[Dict],
    method: str,
    delta_cal: float,
    target_error_rate: float,
) -> Dict:
    """Calibrate on one subset and evaluate selective prediction on another."""
    calibration = calibrate_threshold(
        get_uncertainties_by_method(cal_results, method),
        get_hits(cal_results),
        delta_cal,
        target_error_rate,
    )
    test_uncertainties = get_uncertainties_by_method(test_results, method)
    test_hits = get_hits(test_results)
    threshold = calibration["threshold"]

    selected = [
        index
        for index, uncertainty in enumerate(test_uncertainties)
        if threshold is not None and uncertainty <= threshold
    ]
    accepted = len(selected)
    errors = sum(not test_hits[index] for index in selected)
    correct_total = sum(test_hits)
    correct_accepted = accepted - errors
    n_test = len(test_results)
    accepted_error_rate = errors / accepted if accepted else 0.0

    return {
        "threshold": threshold,
        "attainable": calibration["attainable"],
        "coverage": accepted / n_test,
        "abstention_rate": 1.0 - accepted / n_test,
        "power": correct_accepted / correct_total if correct_total else 0.0,
        "accepted_accuracy": correct_accepted / accepted if accepted else 0.0,
        "accepted_error_rate": accepted_error_rate,
        "calibration_accepted_error_rate": calibration[
            "empirical_accepted_error_rate"
        ],
        "calibration_upper_bound": calibration["upper_bound"],
        "accepted": accepted,
        "accepted_errors": errors,
        "n_cal": len(cal_results),
        "n_test": n_test,
        "guarantee_satisfied": accepted_error_rate <= target_error_rate,
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def run_guarantee_check(
    results: Sequence[Dict],
    methods: Sequence[str],
    delta_cal: float,
    target_error_rates: Sequence[float],
    n_splits: int,
    test_ratio: float,
    delta_split: float,
    seed: int = 42,
) -> Tuple[Dict, int, int]:
    """Estimate test behavior over repeated random calibration/test splits."""
    if n_splits <= 0:
        raise ValueError("n_splits must be positive")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be strictly between 0 and 1")
    if not 0.0 < delta_split < 1.0:
        raise ValueError("delta_split must be strictly between 0 and 1")

    n_total = len(results)
    n_test = int(n_total * test_ratio)
    n_cal = n_total - n_test
    if n_test == 0 or n_cal == 0:
        raise ValueError("test_ratio must leave at least one calibration and test sample")

    rng = np.random.default_rng(seed)
    records = {
        method: {rate: [] for rate in target_error_rates} for method in methods
    }

    for split in range(n_splits):
        indices = rng.permutation(n_total)
        test_results = [results[index] for index in indices[:n_test]]
        cal_results = [results[index] for index in indices[n_test:]]
        if split % 20 == 0:
            logger.info("Split %d/%d", split + 1, n_splits)

        for method in methods:
            for rate in target_error_rates:
                records[method][rate].append(
                    evaluate_split(cal_results, test_results, method, delta_cal, rate)
                )

    summary: Dict = {}
    metric_names = (
        "coverage",
        "abstention_rate",
        "power",
        "accepted_accuracy",
        "accepted_error_rate",
        "calibration_accepted_error_rate",
        "calibration_upper_bound",
    )
    for method in methods:
        summary[method] = {}
        for rate in target_error_rates:
            split_records = records[method][rate]
            finite_thresholds = [
                record["threshold"]
                for record in split_records
                if record["threshold"] is not None
            ]
            accepted_total = sum(record["accepted"] for record in split_records)
            errors_total = sum(record["accepted_errors"] for record in split_records)
            satisfaction_rate = float(
                np.mean([record["guarantee_satisfied"] for record in split_records])
            )

            metrics = {
                name: _mean_std([record[name] for record in split_records])
                for name in metric_names
            }
            metrics.update(
                {
                    "threshold": (
                        _mean_std(finite_thresholds)
                        if finite_thresholds
                        else {"mean": None, "std": None}
                    ),
                    "attainable_rate": float(
                        np.mean([record["attainable"] for record in split_records])
                    ),
                    "aggregate_accepted_error_rate": (
                        errors_total / accepted_total if accepted_total else 0.0
                    ),
                    "guarantee_satisfied_rate": satisfaction_rate,
                    "delta_split_satisfied": satisfaction_rate >= 1.0 - delta_split,
                }
            )
            summary[method][rate] = metrics
    return summary, n_cal, n_test


def save_summary(
    summary: Dict,
    n_cal: int,
    n_test: int,
    output_dir: str,
    target_error_rates: Sequence[float],
    delta_cal: float,
    delta_split: float,
    n_splits: int,
) -> None:
    """Write a readable report and a machine-readable JSON result."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_file = output_path / "accepted_error_control_summary.txt"

    with summary_file.open("w", encoding="utf-8") as file:
        file.write("=" * 100 + "\n")
        file.write("SELECTIVE PREDICTION WITH ACCEPTED ERROR CONTROL\n")
        file.write("=" * 100 + "\n\n")
        file.write(f"Calibration/test samples per split: {n_cal}/{n_test}\n")
        file.write(f"Calibration failure probability (delta_cal): {delta_cal}\n")
        file.write(f"Empirical split-check delta (delta_split): {delta_split}\n")
        file.write(f"Number of random splits: {n_splits}\n")
        file.write(f"Target accepted error rates: {list(target_error_rates)}\n")
        file.write("Threshold rule: Algorithm 1 ordered testing; stop at first uncertified threshold\n\n")

        for method, rate_results in summary.items():
            file.write(f"Method: {method}\n")
            file.write("-" * 100 + "\n")
            file.write(
                f"{'Target':<10} {'Accepted error mean +/- std':<31} "
                f"{'Aggregate':<12} {'Coverage mean +/- std':<28} "
                f"{'P[error<=target]':<18} {'Delta OK':<8}\n"
            )
            for rate in target_error_rates:
                metrics = rate_results[rate]
                error = metrics["accepted_error_rate"]
                coverage = metrics["coverage"]
                file.write(
                    f"{rate:<10.3f} "
                    f"{error['mean']:.4f} +/- {error['std']:.4f}          "
                    f"{metrics['aggregate_accepted_error_rate']:<12.4f} "
                    f"{coverage['mean']:.4f} +/- {coverage['std']:.4f}       "
                    f"{metrics['guarantee_satisfied_rate'] * 100:<17.1f}% "
                    f"{'YES' if metrics['delta_split_satisfied'] else 'NO':<8}\n"
                )
            file.write("\n")

        file.write("Notes:\n")
        file.write("  - Accepted error rate is errors / accepted predictions (0 when none are accepted).\n")
        file.write("  - Power is the fraction of all correct test predictions that remain accepted.\n")
        file.write("  - delta_split is an empirical repeated-split check, not part of Algorithm 1's theorem.\n")

    results_file = output_path / "accepted_error_control_results.json"
    with results_file.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": {
                    "n_cal": n_cal,
                    "n_test": n_test,
                    "delta_cal": delta_cal,
                    "delta_split": delta_split,
                    "n_splits": n_splits,
                    "target_error_rates": list(target_error_rates),
                },
                "summary": summary,
            },
            file,
            indent=2,
            allow_nan=False,
        )
    logger.info("Summary saved to %s", summary_file)
    logger.info("Results saved to %s", results_file)


def parse_target_error_rates(single: Optional[float], range_spec: Optional[str]) -> List[float]:
    """Parse either one target rate or an inclusive ``start:end:step`` range."""
    if single is not None and range_spec is not None:
        raise ValueError("Use only one of --target_error_rate and --target_error_range")
    if range_spec is None:
        rates = [single] if single is not None else list(np.arange(0.30, 0.50 + 0.02, 0.04))
    else:
        parts = range_spec.split(":")
        if len(parts) != 3:
            raise ValueError("target_error_range must have format start:end:step")
        start, end, step = map(float, parts)
        if step <= 0 or end < start:
            raise ValueError("Require step > 0 and end >= start")
        rates = list(np.arange(start, end + step / 2.0, step))

    rounded = [round(float(rate), 10) for rate in rates]
    if any(rate < 0.0 or rate > 1.0 for rate in rounded):
        raise ValueError("Target error rates must be in [0, 1]")
    return rounded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selective Prediction with Accepted Error Control"
    )
    parser.add_argument("--total_file", required=True, help="Path to total.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--delta_cal", type=float, default=0.05)
    parser.add_argument("--delta_split", type=float, default=0.05)
    parser.add_argument("--target_error_rate", type=float)
    parser.add_argument(
        "--target_error_range",
        help="Inclusive target range in start:end:step format",
    )
    parser.add_argument("--n_splits", type=int, default=1000)
    parser.add_argument("--test_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target_error_rates = parse_target_error_rates(
        args.target_error_rate, args.target_error_range
    )
    results, methods = load_total_data(args.total_file)
    logger.info("Running %d random splits", args.n_splits)
    logger.info("Target accepted error rates: %s", target_error_rates)

    summary, n_cal, n_test = run_guarantee_check(
        results,
        methods,
        args.delta_cal,
        target_error_rates,
        args.n_splits,
        args.test_ratio,
        args.delta_split,
        args.seed,
    )
    save_summary(
        summary,
        n_cal,
        n_test,
        args.output_dir,
        target_error_rates,
        args.delta_cal,
        args.delta_split,
        args.n_splits,
    )
    logger.info("Accepted-error-control check completed")


if __name__ == "__main__":
    main()
