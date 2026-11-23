"""Tune carrier-level delivery choices on signed outcomes from chronological backtests."""

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path

import duckdb
import numpy as np

import adaptive
import matched
import release as v2
import solution as core


WINDOWS = {"stress": (matched.STRESS_START, matched.STRESS_END), "hold": (matched.HOLD_START, matched.HOLD_END)}
TARGET_ONTIME = .907
TARGET_TAIL = .973
TAIL_SCALES = (.55, 1, 1.4, 1.8, 2.35)


@dataclass(frozen=True)
class Candidate:
    history_weight: float
    tail_scale: float
    rule: str


def aligned(base, advanced, weight):
    for name in ("carrier", "floor", "y", "pay_day", "sign_day"):
        if name in base and not np.array_equal(base[name], advanced[name]):
            raise ValueError(f"Historical and original forecasts differ in source rows for {name}")
    return {**base, **{name: (1 - weight) * base[name] + weight * advanced[name] for name in ("q90", "q97")}}


def forecasts(directory, advanced, name):
    reference, target, rows = v2.read_batches(directory / f"{name}.npz")
    if advanced is None:
        return {0.: (reference, target)}, rows
    new_reference, new_target, new_rows = v2.read_batches(advanced / f"{name}.npz")
    if rows != new_rows:
        raise ValueError("The feature variants must use identical training-side eligibility")
    blend = (aligned(reference, new_reference, .5), aligned(target, new_target, .5))
    return {0.: (reference, target), .5: blend, 1.: (new_reference, new_target)}, rows


def shape(batch, scale):
    tail = np.maximum(0, batch["q97"] - batch["q90"] - 24)
    return np.maximum(batch["floor"], batch["q90"] + scale * tail)


def carrier_base(reference, target, cutoff, method):
    if method.rule == "allocation":
        policies = adaptive.train(reference, cutoff, 17)
        fallback = v2.promises(reference, target, cutoff)
        return adaptive.prediction(reference, target, cutoff, policies, (.072, .018), fallback=fallback)[0]
    target_shape = shape(target, method.tail_scale)
    reference_shape = shape(reference, method.tail_scale)
    pooled, shifts = v2.shifts(reference, cutoff, reference_shape)
    result = target_shape + pooled
    for group, offset in shifts.items():
        rows = target["carrier"] == group
        result[rows] = target_shape[rows] + offset
    return core.finalize(result, target["floor"])


def required_buffer(actual, forecast, sample_guard):
    residual = actual - forecast
    p_tail = TARGET_TAIL
    if sample_guard:
        # Smaller groups need more headroom when scored on one fourth of the cohort.
        audit_size = max(len(residual) / 3, 1)
        p_tail += min(.008, .9 * np.sqrt(.027 * .973 / audit_size))
    needed = max(core.empirical_quantile(residual, TARGET_ONTIME),
                 core.empirical_quantile(residual, p_tail) - 24)
    # Round upward to match the four decimal places accepted by the result file.
    return float(np.ceil(needed * 10000 + 1e-6) / 10000)


def historical_policy(cache, advanced, audit_rows=None):
    scenarios = {}
    for name, dates in WINDOWS.items():
        variants, _ = forecasts(cache, advanced, name)
        cutoff = matched.day(dates[0])
        candidates = [Candidate(weight, scale, "anchored") for weight, scale in product(variants.keys(), TAIL_SCALES)]
        candidates.extend(Candidate(weight, 1., "allocation") for weight in variants)
        scenarios[name] = {}
        for candidate in candidates:
            ref, target = variants[candidate.history_weight]
            scenarios[name][candidate] = carrier_base(ref, target, cutoff, candidate)
            core.log(f"{name}: {candidate}")
    targets = {name: v2.read_batches(cache / f"{name}.npz")[1] for name in WINDOWS}
    selected = {}
    evaluated = []
    for carrier in np.unique(targets["hold"]["carrier"]):
        masks = {name: (target["carrier"] == carrier) if audit_rows is None else
                 (target["carrier"] == carrier) & ~audit_rows[name] for name, target in targets.items()}
        results = []
        for candidate in candidates:
            minimums = {
                name: required_buffer(target["y"][masks[name]], scenarios[name][candidate][masks[name]], audit_rows is not None)
                for name, target in targets.items()
            }
            buffer = max(-48., *minimums.values())
            means = []
            for name, target in targets.items():
                core_prediction = scenarios[name][candidate][masks[name]] + buffer
                promised = core.finalize(core_prediction, target["floor"][masks[name]])
                means.append(float(np.mean(promised)))
            cost = float(np.mean(means))
            results.append((cost, candidate, buffer))
            evaluated.append(dict(carrier=int(carrier), **asdict(candidate), cost=cost, buffer=buffer, **minimums))
        selected[int(carrier)] = min(results, key=lambda row: row[0])[1:]
    promoted = {}
    for name, target in targets.items():
        final = np.empty(len(target["carrier"]))
        for group, (candidate, buffer) in selected.items():
            mask = target["carrier"] == group
            final[mask] = scenarios[name][candidate][mask] + buffer
        final = core.finalize(final, target["floor"])
        promoted[name] = (target, final)
        core.log(f"{name}: {adaptive.fast_score(target, final)}")
    return selected, evaluated, promoted


def deploy(cache, advanced, selected):
    variants, _ = forecasts(cache, advanced, "test")
    _, target = variants[0.]
    ref_main = variants[0.][0]
    result = v2.promises(ref_main, target, None)
    for candidate in set(candidate for candidate, _ in selected.values()):
        ref, variant = variants[candidate.history_weight]
        proposal = carrier_base(ref, variant, None, candidate)
        for group, (method, padding) in selected.items():
            if method == candidate:
                same_carrier = target["carrier"] == group
                result[same_carrier] = proposal[same_carrier] + padding
    return core.finalize(result, target["floor"])


def audit_masks(db):
    return {
        name: db.execute("""
            SELECT hash(pkg_no) % 4 = 1 AS audit
            FROM train
            WHERE pay_ts >= CAST(? AS TIMESTAMP) AND pay_ts < CAST(? AS TIMESTAMP)
            ORDER BY rid
        """, dates).fetchnumpy()["audit"]
        for name, dates in WINDOWS.items()
    }


def save_score(folder, name, truth, prediction, mask=None):
    if mask is None:
        mask = slice(None)
    y, hours = truth.y[mask], prediction[mask]
    ids, carriers = truth.ids[mask], truth.carrier_text[mask]
    score, detail = core.evaluate(y, hours, carriers)
    detail.to_csv(folder / f"{name}_carriers.csv", index=False, float_format="%.8f")
    core.export_predictions(folder / f"{name}_prediction.csv", ids, hours)
    core.atomic_csv(folder / f"{name}_truth.csv", ["pkg_no", "lgs_company_nm", "y_total_hours"], zip(ids, carriers, y))
    core.log(f"{name}: {score}")
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--advanced", type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.db is None:
        parser.error("--output requires --db to verify package IDs")
    args.diagnostics.mkdir(parents=True, exist_ok=True)
    audit = None
    if args.db is not None:
        with duckdb.connect(str(args.db), read_only=True) as db:
            audit = audit_masks(db)
    selected, evaluated, promoted = historical_policy(args.cache, args.advanced, audit)
    summary = {"by_carrier": {str(group): dict(**asdict(rule), buffer=buffer)
                              for group, (rule, buffer) in selected.items()}, "windows": {}, "audit": {}}
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(args.db), read_only=True, config={"threads": 6, "memory_limit": "2GB"}) as db:
            for name, (target, prediction) in promoted.items():
                truth = core.range_frame(db, WINDOWS[name])
                if not np.array_equal(target["y"], truth.y) or not np.array_equal(target["carrier"], truth.carrier):
                    raise ValueError("Outcome mismatch when exporting historical validation")
                summary["windows"][name] = save_score(args.diagnostics, name, truth, prediction)
                summary["audit"][name] = save_score(args.diagnostics, f"{name}_audit", truth, prediction, audit[name])
            prediction = deploy(args.cache, args.advanced, selected)
            keys = db.execute("SELECT pkg_no FROM test ORDER BY rid").fetchnumpy()["pkg_no"]
            path = args.output / "prediction.csv"
            core.export_predictions(path, keys, prediction)
            summary["submission"] = core.check_submission(db, path)
            core.log(f"Verified {summary['submission']}")
    else:
        summary["windows"] = {name: adaptive.fast_score(target, pred) for name, (target, pred) in promoted.items()}
    (args.diagnostics / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.diagnostics / "carrier_candidates.json").write_text(json.dumps(evaluated, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
