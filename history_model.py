"""Fit point-in-time history features for the F4 quantile model."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import duckdb
import numpy as np

import matched
import solution as core


ENTITY_NAMES = ("carrier", "channel", "carrier_state")
ENTITY_SQL = (
    "lgs_company_nm::BIGINT", "shpp_chnl::BIGINT",
    "lgs_company_nm::BIGINT * 4294967296 + (shpp_prvn_nm::BIGINT + 1)",
)
METRIC_NAMES = ("past_issue_support", "cohort_done_8d", "cohort_done_12d", "recent_signed_hours", "recent_inflow_ratio")
FULL_FEATURES = core.FEATURES + tuple(f"hist_{group}_{metric}" for group in ENTITY_NAMES for metric in METRIC_NAMES)


@dataclass
class Daily:
    codes: np.ndarray
    dates: np.ndarray
    count: np.ndarray
    values: np.ndarray


def load_series(db, key):
    source = f"train WHERE NOT ({matched.CALIBRATION_MEMBERS}) AND pay_ts <= issue_ts AND issue_ts <= sign_ts"
    issue_day = matched.ORDINAL_SQL.format("issue_ts")
    sign_day = matched.ORDINAL_SQL.format("sign_ts")
    cohorts = db.execute(f"""
        SELECT {key} AS entity, {issue_day} AS day, count(*) AS n,
               count(*) FILTER (WHERE sign_ts <= issue_ts + INTERVAL 8 DAY) AS done8,
               count(*) FILTER (WHERE sign_ts <= issue_ts + INTERVAL 12 DAY) AS done12
        FROM {source} GROUP BY 1, 2
    """).fetchnumpy()
    signs = db.execute(f"""
        SELECT {key} AS entity, {sign_day} AS day, count(*) AS n,
               sum(y) AS total_hours
        FROM {source} GROUP BY 1, 2
    """).fetchnumpy()
    return (
        Daily(cohorts["entity"], cohorts["day"], cohorts["n"], np.column_stack((cohorts["done8"], cohorts["done12"]))),
        Daily(signs["entity"], signs["day"], signs["n"], signs["total_hours"][:, None]),
    )


def rolling_sum(data, window, maturity):
    """Statistics for cohorts strictly before the anchor, with full maturity."""
    prefix = np.pad(np.cumsum(data, axis=1), ((0, 0), (1, 0), (0, 0)))
    end = np.maximum(0, np.arange(data.shape[1]) - maturity)
    start = np.maximum(0, end - window)
    return prefix[:, end] - prefix[:, start]


def division(value, count, min_support):
    return np.divide(value, count, out=np.full_like(value, np.nan, dtype=float), where=count >= min_support)


def build_snapshot(cohorts, signs):
    codes = np.union1d(cohorts.codes, signs.codes)
    first = min(0, cohorts.dates.min(), signs.dates.min())
    window_size = max(128, int(max(cohorts.dates.max(), signs.dates.max()) - first + 2))

    def materialize(rows):
        dense = np.zeros((len(codes), window_size, 1 + rows.values.shape[1]))
        dense[np.searchsorted(codes, rows.codes), rows.dates - first] = np.column_stack((rows.count, rows.values))
        return dense

    issue_stats = materialize(cohorts)
    last8 = rolling_sum(issue_stats, 14, 8)
    last12 = rolling_sum(issue_stats, 18, 12)
    recent_signed = rolling_sum(materialize(signs), 9, 0)
    volume3 = rolling_sum(issue_stats, 3, 0)[:, :, 0]
    volume14 = rolling_sum(issue_stats, 14, 0)[:, :, 0]
    result = np.stack((
        np.log1p(last8[:, :, 0]), division(last8[:, :, 1], last8[:, :, 0], 35),
        division(last12[:, :, 2], last12[:, :, 0], 35),
        division(recent_signed[:, :, 1], recent_signed[:, :, 0], 25),
        division(14 / 3 * volume3, volume14, 35),
    ), axis=-1).astype(np.float32)
    return codes, int(first), result


def augment(db, frame, table, where, params, dates, cutoff):
    day = matched.ORDINAL_SQL.format("issue_ts")
    issue_days = db.execute(f"SELECT {day} FROM {table} WHERE {where} ORDER BY rid", params).fetchnumpy()
    anchor = next(iter(issue_days.values())).astype(np.int32)
    if cutoff is not None:
        anchor = np.minimum(anchor, matched.day(cutoff))
    carrier = frame.x[:, 0].astype(np.int64)
    channels = frame.x[:, 1].astype(np.int64)
    geography = carrier * 4294967296 + (frame.x[:, 2].astype(np.int64) + 1)
    out = np.empty((len(frame.x), len(FULL_FEATURES)), dtype=np.float32)
    out[:, :len(core.FEATURES)] = frame.x
    for index, (values, snapshot) in enumerate(zip((carrier, channels, geography), dates, strict=True)):
        codes, first, matrix = snapshot
        indices = np.searchsorted(codes, values)
        positions = np.minimum(len(codes) - 1, indices)
        valid = (codes[positions] == values) & (anchor >= first) & (anchor - first < matrix.shape[1])
        column = len(core.FEATURES) + index * len(METRIC_NAMES)
        features = out[:, column:column + len(METRIC_NAMES)]
        features[:] = np.nan
        features[valid] = matrix[positions[valid], anchor[valid] - first]
    return core.Frame(out, frame.y)


def run_split(db, cache, name, start, end, snapshots, threads, rounds):
    path = cache / f"{name}.npz"
    if path.exists():
        with np.load(path) as saved:
            if saved["rounds"].item() == rounds:
                core.log(f"Reusing historical features and forecasts for {name}")
                return
    train_where = f"NOT ({matched.CALIBRATION_MEMBERS})"
    ref_where = f"({matched.CALIBRATION_MEMBERS})"
    if start is not None:
        train_where += " AND pay_ts < CAST(? AS TIMESTAMP) AND sign_ts < CAST(? AS TIMESTAMP)"
        ref_where += " AND pay_ts < CAST(? AS TIMESTAMP) AND issue_ts < CAST(? AS TIMESTAMP)"
        train_params = ref_params = (start, start)
        target_table = "train"
        target_where = "pay_ts >= CAST(? AS TIMESTAMP) AND pay_ts < CAST(? AS TIMESTAMP)"
        target_params = (start, end)
        target_dates = matched.timestamps(db, target_where, target_params)
    else:
        train_params = target_params = ()
        ref_where += " AND pay_ts >= CAST(? AS TIMESTAMP)"
        ref_params = ("2026-08-27",)
        target_table, target_where, target_dates = "test", "true", {}
    train = augment(db, core.read_frame(db, "train", train_where, train_params), "train", train_where, train_params, snapshots, start)
    ref = augment(db, core.read_frame(db, "train", ref_where, ref_params), "train", ref_where, ref_params, snapshots, start)
    test = augment(db, core.read_frame(db, target_table, target_where, target_params), target_table, target_where, target_params, snapshots, start)
    ref_dates = matched.timestamps(db, ref_where, ref_params)
    core.log(f"{name}: history features, {len(train.y):,} fit rows; {len(ref.x):,} calibration rows; {len(test.x):,} forecast rows")
    both = core.Frame(np.concatenate([ref.x, test.x]))
    q90, q97 = core.fit_quantiles(train, both, threads, rounds, FULL_FEATURES)
    split = len(ref.x)
    archive = {}
    for prefix, frame, fields, p90, p97 in (("ref", ref, ref_dates, q90[:split], q97[:split]),
                                              ("eval", test, target_dates, q90[split:], q97[split:])):
        series = dict(q90=p90, q97=p97, floor=frame.duration_floor, carrier=frame.carrier, **fields)
        if frame.y is not None:
            series["y"] = frame.y
        archive.update({f"{prefix}_{key}": value for key, value in series.items()})
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **archive, training_rows=len(train.y), rounds=rounds)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "benchmark", "test"), default="benchmark")
    parser.add_argument("--rounds", type=int, default=240)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with duckdb.connect(str(args.db), read_only=True, config={"threads": args.threads, "memory_limit": "2GB"}) as db:
        snapshots = [build_snapshot(*load_series(db, key)) for key in ENTITY_SQL]
        if args.phase in ("all", "benchmark"):
            run_split(db, args.output, "stress", matched.STRESS_START, matched.STRESS_END, snapshots, args.threads, args.rounds)
            run_split(db, args.output, "hold", matched.HOLD_START, matched.HOLD_END, snapshots, args.threads, args.rounds)
        if args.phase in ("all", "test"):
            run_split(db, args.output, "test", None, None, snapshots, args.threads, args.rounds)
    core.log(f"Historical forecast fitting completed in {time.monotonic() - started:.1f} s")


if __name__ == "__main__":
    main()
