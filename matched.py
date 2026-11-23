"""Fit each calibration set and evaluation batch with the same base model."""

import argparse
from pathlib import Path
import time

import duckdb
import numpy as np

import solution as core


DEV_START = "2026-08-27"
DEV_END = "2026-09-04"
HOLD_START, HOLD_END = core.HOLDOUT
STRESS_START, STRESS_END = DEV_END, HOLD_START
CALIBRATION_MEMBERS = "hash(pkg_no) % 8 = 0"
ORDINAL_SQL = "date_diff('day', DATE '2026-08-01', {})::SMALLINT"


def day(date):
    return int((np.datetime64(date) - np.datetime64("2026-08-01")) / np.timedelta64(1, "D"))


def timestamps(db, where, params):
    pay = ORDINAL_SQL.format("pay_ts")
    sign = ORDINAL_SQL.format("sign_ts")
    return db.execute(f"""
        SELECT {pay} AS pay_day, {sign} AS sign_day,
               epoch(pay_ts - TIMESTAMP '2026-08-01') / 3600 AS pay_hour
        FROM train WHERE {where} ORDER BY rid
    """, params).fetchnumpy()


def fit_split(db, output, name, start, end, threads, rounds):
    out_path = output / f"{name}.npz"
    if out_path.exists():
        with np.load(out_path) as past:
            if past["rounds"].item() == rounds:
                core.log(f"Using cached matched fit for {name}")
                return
    if start is not None:
        known = "pay_ts < CAST(? AS TIMESTAMP) AND sign_ts < CAST(? AS TIMESTAMP)"
        train = core.read_frame(db, "train", f"{known} AND NOT ({CALIBRATION_MEMBERS})", (start, start))
        reference_where = f"({CALIBRATION_MEMBERS}) AND pay_ts < CAST(? AS TIMESTAMP) AND issue_ts < CAST(? AS TIMESTAMP)"
        reference_params = (start, start)
        valid_where = "pay_ts >= CAST(? AS TIMESTAMP) AND pay_ts < CAST(? AS TIMESTAMP)"
        valid = core.read_frame(db, "train", valid_where, (start, end))
        valid_dates = timestamps(db, valid_where, (start, end))
    else:
        train = core.read_frame(db, "train", f"NOT ({CALIBRATION_MEMBERS})")
        reference_where = f"({CALIBRATION_MEMBERS}) AND pay_ts >= CAST(? AS TIMESTAMP)"
        reference_params = ("2026-08-27",)
        valid = core.read_frame(db, "test")
        valid_dates = {}
    ref = core.read_frame(db, "train", reference_where, reference_params)
    ref_dates = timestamps(db, reference_where, reference_params)
    core.log(f"Matched {name}: {len(train.y):,} training, {len(ref.y):,} reference, {len(valid.x):,} forecast rows")
    combined = core.Frame(np.concatenate([ref.x, valid.x]))
    q90, q97 = core.fit_quantiles(train, combined, threads, rounds)
    cut = len(ref.x)
    record = {}
    for label, frame, dates, qs in (("ref", ref, ref_dates, (q90[:cut], q97[:cut])),
                                     ("eval", valid, valid_dates, (q90[cut:], q97[cut:]))):
        data = dict(q90=qs[0], q97=qs[1], floor=frame.duration_floor, carrier=frame.carrier, **dates)
        if frame.y is not None:
            data["y"] = frame.y
        record.update({f"{label}_{key}": value for key, value in data.items()})
    temporary = out_path.with_suffix(".partial.npz")
    np.savez(temporary, **record, rounds=rounds, training_rows=len(train.y))
    temporary.replace(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=240)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with duckdb.connect(str(args.db), read_only=True, config={"threads": args.threads, "memory_limit": "2GB"}) as db:
        splits = (("hold", HOLD_START, HOLD_END), ("stress", STRESS_START, STRESS_END), ("test", None, None))
        for name, start, end in splits:
            fit_split(db, args.output, name, start, end, args.threads, args.rounds)
    core.log(f"Matched forecasts available after {time.monotonic() - started:.1f} s")


if __name__ == "__main__":
    main()
