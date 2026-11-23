"""Export the carrier-calibrated F4 forecast selected in the two-window comparison."""

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import CensoredData, ecdf

import matched
import solution as core


MIN_AGE_DAYS = 14
ONTIME_QUANTILE = .908
TAIL_QUANTILE = .981
BUFFER_HOURS = 20


def read_batches(path):
    with np.load(path) as saved:
        ref, evaluation = (
            {key[len(prefix):]: saved[key] for key in saved if key.startswith(prefix)}
            for prefix in ("ref_", "eval_")
        )
        return ref, evaluation, int(saved["training_rows"].item())


def base_forecast(batch):
    return np.maximum(batch["floor"], np.maximum(batch["q90"], batch["q97"] - 24))


def residual_quantile(curve, observed, probability):
    rank = np.searchsorted(curve.cdf.probabilities, probability - 1e-12)
    if rank < len(curve.cdf.quantiles):
        return float(curve.cdf.quantiles[rank])
    # A censored upper tail can leave a requested rank unidentified.
    last_event = np.flatnonzero(np.diff(curve.cdf.probabilities, prepend=0) > 0)[-1]
    identified = min(last_event, np.searchsorted(curve.cdf.probabilities, min(.975, probability)))
    return max(float(curve.cdf.quantiles[identified]), core.empirical_quantile(observed, probability))


def constraint_shift(base, outcome, known, observation):
    observed = outcome[known] - base[known]
    if not observed.size:
        raise ValueError("Calibration has no observed delivery times")
    censored = observation[~known] - base[~known]
    sample = CensoredData(uncensored=observed, right=censored) if censored.size else observed
    curve = ecdf(sample)
    ontime = residual_quantile(curve, observed, ONTIME_QUANTILE)
    severe = residual_quantile(curve, observed, TAIL_QUANTILE) - 24
    return max(ontime, severe)


def shifts(reference, cutoff, base=None):
    if cutoff is None:
        available = np.ones(len(reference["y"]), dtype=bool)
        known = available
        observation = np.zeros(len(known))
    else:
        available = reference["pay_day"] < cutoff - MIN_AGE_DAYS
        known = reference["sign_day"][available] < cutoff
        observation = cutoff * 24 - reference["pay_hour"][available]
    carrier = reference["carrier"][available]
    base = (base_forecast(reference) if base is None else base)[available]
    outcomes = reference["y"][available]
    pooled = constraint_shift(base, outcomes, known, observation)
    by_carrier = {}
    for group in np.unique(carrier):
        mask = carrier == group
        if (known & mask).sum() < 200:
            continue
        shift = constraint_shift(base[mask], outcomes[mask], known[mask], observation[mask])
        size = int(mask.sum())
        weight = size / (size + 120)
        by_carrier[int(group)] = weight * shift + (1 - weight) * pooled
    return pooled, by_carrier


def promises(reference, target, cutoff):
    pooled, by_carrier = shifts(reference, cutoff)
    base = base_forecast(target)
    prediction = base + pooled
    for carrier, shift in by_carrier.items():
        mask = target["carrier"] == carrier
        prediction[mask] = base[mask] + shift
    return core.finalize(prediction, target["floor"]) + BUFFER_HOURS


def verify_window(db, cache, diagnostics, name, dates):
    ref, target, training_n = read_batches(cache / f"{name}.npz")
    prediction = promises(ref, target, matched.day(dates[0]))
    truth = core.range_frame(db, dates)
    if not np.array_equal(target["y"], truth.y) or not np.array_equal(target["carrier"], truth.carrier):
        raise ValueError("Validation records and cached forecasts are out of order")
    score, groups = core.evaluate(truth.y, prediction, truth.carrier_text)
    groups.to_csv(diagnostics / f"{name}_carriers.csv", index=False, float_format="%.8f")
    core.export_predictions(diagnostics / f"{name}_prediction.csv", truth.ids, prediction)
    core.atomic_csv(diagnostics / f"{name}_truth.csv", ["pkg_no", "lgs_company_nm", "y_total_hours"],
                    zip(truth.ids, truth.carrier_text, truth.y))
    core.log(f"{name}: {score}")
    return dict(dates=list(dates), training_n=training_n, **score)


def run(db, cache, output, diagnostics):
    output.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    windows = {
        "earlier": verify_window(db, cache, diagnostics, "stress", (matched.STRESS_START, matched.STRESS_END)),
        "later": verify_window(db, cache, diagnostics, "hold", (matched.HOLD_START, matched.HOLD_END)),
    }
    reference, test, training_n = read_batches(cache / "test.npz")
    prediction = promises(reference, test, None)
    ids = db.execute("SELECT pkg_no FROM test ORDER BY rid").fetchnumpy()["pkg_no"]
    path = output / "prediction.csv"
    core.export_predictions(path, ids, prediction)
    verification = core.check_submission(db, path)
    summary = dict(
        windows=windows, submission=verification, production_training_n=training_n,
        production_calibration_n=len(reference["y"]),
        settings=dict(min_age_days=MIN_AGE_DAYS, ontime_quantile=ONTIME_QUANTILE,
                      tail_quantile=TAIL_QUANTILE, buffer_hours=BUFFER_HOURS),
    )
    (diagnostics / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    core.log(f"Verified submission: {verification}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    with duckdb.connect(str(args.db), read_only=True, config={"threads": 6, "memory_limit": "2GB"}) as db:
        run(db, args.cache, args.output, args.diagnostics)


if __name__ == "__main__":
    main()
