"""
Zeno - Compact Story Demo for Classical ML
==========================================

A smaller, judge-friendly demo focused on three things:
  1. real-world domain shift,
  2. carbon-aware transfer,
  3. safety against negative transfer.

Stories:
  - Economy: housing affordability transfer (North -> South California)
  - Health: tumor screening transfer (small -> large tumors)
  - Safety: synthetic negative-transfer case where the gate says SKIP

Usage:
    python -m tests.run_story_ml_demo
    python -m tests.run_story_ml_demo --scenario housing --seeds 5
    python -m tests.run_story_ml_demo --target-frac 0.25
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from libraries.metrics import set_seed, mse, r2_score, accuracy_from_logits
from libraries.train_core import fit_linear_sgd, fit_logistic_sgd
from libraries.transfer import (
    regularized_transfer_linear,
    bayesian_transfer_linear,
    bayesian_transfer_logistic,
)
from libraries.negative_transfer import should_transfer
from libraries.carbon import CarbonTracker
from tests.real_datasets import (
    load_california_housing_linear,
    load_breast_cancer_logistic,
    load_diabetes_linear,
)


def to_torch(X, y):
    return torch.from_numpy(X), torch.from_numpy(y)


def take_fraction(X, y, frac, seed=0):
    if frac >= 1.0:
        return X, y
    rng = np.random.RandomState(seed)
    n = len(X)
    k = max(16, int(round(n * frac)))
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    return X[idx], y[idx]


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = 1.96 * std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, std, ci95


def fmt(mean, std, ci95, pct=False):
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    return f"{mean * scale:.2f}{suffix} +/- {std * scale:.2f}{suffix} (95% CI +/- {ci95 * scale:.2f}{suffix})"


def tracker_run(label, fn):
    tracker = CarbonTracker(label)
    tracker.start()
    out = fn()
    carbon = tracker.stop()
    return out, carbon


def run_regression_story(seed, target_frac):
    dataset_name = "california_housing"
    title = "Housing affordability under regional shift"
    subtitle = "Source = Northern California, target = Southern California"
    fallback_note = None

    try:
        (Xs_tr, ys_tr, _, _), (Xt_tr, yt_tr, Xt_te, yt_te) = load_california_housing_linear(seed=seed)
    except Exception:
        dataset_name = "diabetes_progression"
        title = "Disease progression under population shift"
        subtitle = "Offline fallback: source = lower-BMI patients, target = higher-BMI patients"
        fallback_note = (
            "California Housing was unavailable on this machine, so the demo fell back to "
            "the built-in diabetes progression dataset."
        )
        (Xs_tr, ys_tr, _, _), (Xt_tr, yt_tr, Xt_te, yt_te) = load_diabetes_linear(seed=seed)

    Xt_tr, yt_tr = take_fraction(Xt_tr, yt_tr, target_frac, seed=seed + 17)

    decision = should_transfer(Xs_tr, Xt_tr)

    Xs_t, ys_t = to_torch(Xs_tr, ys_tr)
    Xt_t, yt_t = to_torch(Xt_tr, yt_tr)
    Xte_t, yte_t = to_torch(Xt_te, yt_te)

    d = Xs_t.shape[1]
    w0, b0 = torch.zeros(d), torch.zeros(1)
    batch_size = min(64, max(16, len(Xt_tr) // 4))

    label_prefix = dataset_name

    (w_src, b_src), source_carbon = tracker_run(
        f"{label_prefix}_source_pretrain",
        lambda: fit_linear_sgd(
            Xs_t, ys_t, w0, b0,
            epochs=30, lr=0.01, batch_size=min(64, len(Xs_tr)),
        ),
    )

    (scratch_wb, scratch_carbon) = tracker_run(
        f"{label_prefix}_scratch",
        lambda: fit_linear_sgd(
            Xt_t, yt_t, w0, b0,
            epochs=30, lr=0.01, batch_size=batch_size,
        ),
    )
    w_scratch, b_scratch = scratch_wb
    scratch_r2 = r2_score(Xte_t @ w_scratch + b_scratch, yte_t)

    (reg_wb, reg_carbon) = tracker_run(
        f"{label_prefix}_regularized",
        lambda: regularized_transfer_linear(Xt_t, yt_t, w_src, b_src, lam=1.0),
    )
    w_reg, b_reg = reg_wb
    reg_r2 = r2_score(Xte_t @ w_reg + b_reg, yte_t)

    (bayes_wb, bayes_carbon) = tracker_run(
        f"{label_prefix}_bayesian",
        lambda: bayesian_transfer_linear(Xt_t, yt_t, w_src, b_src),
    )
    w_bayes, b_bayes = bayes_wb
    bayes_r2 = r2_score(Xte_t @ w_bayes + b_bayes, yte_t)

    return {
        "dataset_name": dataset_name,
        "title": title,
        "subtitle": subtitle,
        "fallback_note": fallback_note,
        "decision": decision,
        "source_pretrain": source_carbon,
        "scratch": {"score": float(scratch_r2), "carbon": scratch_carbon},
        "regularized": {"score": float(reg_r2), "carbon": reg_carbon},
        "bayesian": {"score": float(bayes_r2), "carbon": bayes_carbon},
        "n_target": int(len(Xt_tr)),
    }


def run_breast_cancer(seed, target_frac):
    (Xs_tr, ys_tr, _, _), (Xt_tr, yt_tr, Xt_te, yt_te) = load_breast_cancer_logistic(seed=seed)
    Xt_tr, yt_tr = take_fraction(Xt_tr, yt_tr, target_frac, seed=seed + 23)

    decision = should_transfer(Xs_tr, Xt_tr)

    Xs_t, ys_t = to_torch(Xs_tr, ys_tr)
    Xt_t, yt_t = to_torch(Xt_tr, yt_tr)
    Xte_t, yte_t = to_torch(Xt_te, yt_te)

    d = Xs_t.shape[1]
    w0, b0 = torch.zeros(d), torch.zeros(1)
    batch_size = min(len(Xt_tr), max(16, len(Xt_tr) // 3))

    (w_src, b_src), source_carbon = tracker_run(
        "cancer_source_pretrain",
        lambda: fit_logistic_sgd(
            Xs_t, ys_t, w0, b0,
            epochs=80, lr=0.01, batch_size=min(32, len(Xs_tr)),
        ),
    )

    (scratch_wb, scratch_carbon) = tracker_run(
        "cancer_scratch",
        lambda: fit_logistic_sgd(
            Xt_t, yt_t, w0, b0,
            epochs=80, lr=0.01, batch_size=batch_size,
        ),
    )
    w_scratch, b_scratch = scratch_wb
    scratch_acc = accuracy_from_logits(Xte_t @ w_scratch + b_scratch, yte_t)

    (bayes_wb, bayes_carbon) = tracker_run(
        "cancer_bayesian",
        lambda: bayesian_transfer_logistic(
            Xt_t, yt_t, w_src, b_src,
            source_precision=1.0,
            epochs=40, lr=0.01, batch_size=batch_size,
        ),
    )
    w_bayes, b_bayes = bayes_wb
    bayes_acc = accuracy_from_logits(Xte_t @ w_bayes + b_bayes, yte_t)

    return {
        "decision": decision,
        "source_pretrain": source_carbon,
        "scratch": {"score": float(scratch_acc), "carbon": scratch_carbon},
        "bayesian": {"score": float(bayes_acc), "carbon": bayes_carbon},
        "n_target": int(len(Xt_tr)),
    }


def run_negative_transfer(seed):
    rng = np.random.RandomState(seed)
    d = 8
    w_true = rng.randn(d).astype(np.float32)

    Xs = (rng.randn(400, d) + 1.0).astype(np.float32)
    ys = (Xs @ w_true + 0.1 * rng.randn(400)).astype(np.float32)

    Xt = (rng.randn(120, d) - 1.0).astype(np.float32)
    yt = (-(Xt @ w_true) + 0.1 * rng.randn(120)).astype(np.float32)

    decision = should_transfer(Xs, Xt, verbose=False)

    Xs_t, ys_t = to_torch(Xs, ys)
    Xt_t, yt_t = to_torch(Xt, yt)
    d = Xs_t.shape[1]
    w0, b0 = torch.zeros(d), torch.zeros(1)

    w_src, b_src = fit_linear_sgd(Xs_t, ys_t, w0, b0, epochs=20, lr=0.05, batch_size=64)
    w_scratch, b_scratch = fit_linear_sgd(Xt_t, yt_t, w0, b0, epochs=20, lr=0.05, batch_size=32)
    w_naive, b_naive = fit_linear_sgd(Xt_t, yt_t, w_src.clone(), b_src.clone(), epochs=3, lr=0.05, batch_size=32)
    w_safe, b_safe = regularized_transfer_linear(Xt_t, yt_t, w_src, b_src, lam=0.05)

    scratch_mse = mse(Xt_t @ w_scratch + b_scratch, yt_t)
    naive_mse = mse(Xt_t @ w_naive + b_naive, yt_t)
    safe_mse = mse(Xt_t @ w_safe + b_safe, yt_t)

    return {
        "decision": decision,
        "scratch_mse": float(scratch_mse),
        "naive_mse": float(naive_mse),
        "safe_mse": float(safe_mse),
    }


def aggregate_story(results, method_names):
    summary = {}
    for method in method_names:
        scores = [r[method]["score"] for r in results]
        co2 = [r[method]["carbon"]["co2_kg"] for r in results]
        time_s = [r[method]["carbon"]["time_s"] for r in results]
        summary[method] = {
            "score": summarize(scores),
            "co2": summarize(co2),
            "time": summarize(time_s),
        }
    return summary


def print_story_header(title, subtitle):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print(f"  {subtitle}")
    print("=" * 76)


def print_metric_block(method, metric_name, stats, co2_stats, time_stats, pct=False):
    print(f"  {method:<16} {metric_name}: {fmt(*stats, pct=pct)}")
    print(f"{'':<18} CO2:   {co2_stats[0]:.2e} kg +/- {co2_stats[1]:.2e}")
    print(f"{'':<18} Time:  {time_stats[0]:.4f}s +/- {time_stats[1]:.4f}s")


def main():
    parser = argparse.ArgumentParser(description="Compact classical transfer-learning story demo")
    parser.add_argument("--scenario", choices=["all", "housing", "health", "safety"], default="all")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--target-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 76)
    print("  Zeno Story Demo - Classical ML")
    print("  \"Smaller demo, stronger story: real shifts, low carbon, safe transfer\"")
    print("=" * 76)
    print(f"  seeds={args.seeds}  target_frac={args.target_frac:.0%}  base_seed={args.seed}")

    if args.scenario in ("all", "housing"):
        set_seed(args.seed)
        housing_runs = [run_regression_story(args.seed + i, args.target_frac) for i in range(args.seeds)]
        housing_summary = aggregate_story(housing_runs, ["scratch", "regularized", "bayesian"])
        decision = housing_runs[0]["decision"]
        print_story_header(
            f"Story 1: {housing_runs[0]['title']}",
            housing_runs[0]["subtitle"],
        )
        if housing_runs[0].get("fallback_note"):
            print(f"  Note: {housing_runs[0]['fallback_note']}")
        print(f"  Target examples used per run: {housing_runs[0]['n_target']}")
        print(f"  Transfer gate: {'TRANSFER' if decision['recommend'] else 'SKIP'}  "
              f"(MMD^2={decision['mmd']:.4f}, PAD={decision['pad']:.4f}, "
              f"KS-shift={decision['ks_fraction']:.0%})")
        print_metric_block("Scratch", "R^2", housing_summary["scratch"]["score"], housing_summary["scratch"]["co2"], housing_summary["scratch"]["time"])
        print_metric_block("Regularized", "R^2", housing_summary["regularized"]["score"], housing_summary["regularized"]["co2"], housing_summary["regularized"]["time"])
        print_metric_block("Bayesian", "R^2", housing_summary["bayesian"]["score"], housing_summary["bayesian"]["co2"], housing_summary["bayesian"]["time"])

        scratch_co2 = housing_summary["scratch"]["co2"][0]
        bayes_co2 = housing_summary["bayesian"]["co2"][0]
        if scratch_co2 > 0:
            saved = 100.0 * (scratch_co2 - bayes_co2) / scratch_co2
            gain = housing_summary["bayesian"]["score"][0] - housing_summary["scratch"]["score"][0]
            print(f"\n  Takeaway: Bayesian transfer changes the story from \"train from zero\" to")
            print(f"  \"reuse a source model\" — here that is {gain:+.4f} R^2 with {saved:.1f}% less target-stage CO2.")

    if args.scenario in ("all", "health"):
        set_seed(args.seed)
        health_runs = [run_breast_cancer(args.seed + 100 + i, args.target_frac) for i in range(args.seeds)]
        health_summary = aggregate_story(health_runs, ["scratch", "bayesian"])
        decision = health_runs[0]["decision"]
        print_story_header(
            "Story 2: Health screening under covariate shift",
            "Source = smaller tumors, target = larger tumors",
        )
        print(f"  Target examples used per run: {health_runs[0]['n_target']}")
        print(f"  Transfer gate: {'TRANSFER' if decision['recommend'] else 'SKIP'}  "
              f"(MMD^2={decision['mmd']:.4f}, PAD={decision['pad']:.4f}, "
              f"KS-shift={decision['ks_fraction']:.0%})")
        print_metric_block("Scratch", "Accuracy", health_summary["scratch"]["score"], health_summary["scratch"]["co2"], health_summary["scratch"]["time"], pct=True)
        print_metric_block("Bayesian", "Accuracy", health_summary["bayesian"]["score"], health_summary["bayesian"]["co2"], health_summary["bayesian"]["time"], pct=True)

        scratch_co2 = health_summary["scratch"]["co2"][0]
        bayes_co2 = health_summary["bayesian"]["co2"][0]
        if scratch_co2 > 0:
            saved = 100.0 * (scratch_co2 - bayes_co2) / scratch_co2
            gain = 100.0 * (health_summary["bayesian"]["score"][0] - health_summary["scratch"]["score"][0])
            print(f"\n  Takeaway: when target labels are scarce, a source-informed prior can buy")
            print(f"  {gain:+.2f} percentage points with {saved:.1f}% less target-stage CO2.")

    if args.scenario in ("all", "safety"):
        safety = run_negative_transfer(args.seed)
        print_story_header(
            "Story 3: Safety first - negative transfer guardrail",
            "Synthetic case with shifted features and reversed target relationship",
        )
        print(f"  Transfer gate: {'TRANSFER' if safety['decision']['recommend'] else 'SKIP'}  "
              f"(MMD^2={safety['decision']['mmd']:.4f}, PAD={safety['decision']['pad']:.4f}, "
              f"KS-shift={safety['decision']['ks_fraction']:.0%})")
        print(f"  Scratch MSE:       {safety['scratch_mse']:.4f}")
        print(f"  Naive transfer:    {safety['naive_mse']:.4f}")
        print(f"  Safe regularized:  {safety['safe_mse']:.4f}")
        if safety['scratch_mse'] > 0:
            ratio = safety['naive_mse'] / safety['scratch_mse']
            print(f"\n  Takeaway: naive reuse is {ratio:.1f}x worse than scratch here, so the")
            print("  guardrail is not a side feature - it is part of the product story.")

    print("\n" + "=" * 76)
    print("  Why this version is better for judges")
    print("=" * 76)
    print("  1. It starts with real problems, not methods.")
    print("  2. It reports mean/std across seeds instead of a single lucky run.")
    print("  3. It treats carbon and negative transfer as first-class outputs.")


if __name__ == "__main__":
    main()
