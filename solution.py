"""Train and validate an F4 forecast, then export the contest submission."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


TRAIN_FILE = "train_202608_0920.csv"
TEST_FILE = "test_202609.csv"
EXPECTED = {"train": 5_267_074, "test": 911_415}
DEVELOPMENT = ("2026-08-22", "2026-09-01")
HOLDOUT = ("2026-09-12", "2026-09-21")
SEED = 1729
FLOOR = 0.01
CATEGORICAL = (
    "lgs_company_nm", "shpp_chnl", "shpp_prvn_nm", "shpp_city_nm",
    "stk_cate_tp_nm", "start_airport_cd", "end_airport_cd", "mode_category",
    "combine_flag", "combine_type", "business_tp", "operation_type",
    "route_cat", "sale_area",
)
NUMERIC = (
    "pkg_goods_cnt", "weight", "pay_to_issue", "pay_dow", "pay_hour",
    "issue_dow", "issue_hour", "send_country_known",
)
FEATURES = CATEGORICAL + NUMERIC
CAT_INDICES = list(range(len(CATEGORICAL)))
POLICIES = {
    "calibrated": (0.900, 0.970),
    "buffered": (0.911, 0.978),
    "conservative": (0.925, 0.986),
}


def log(message):
    print(message, flush=True)


def prepare(db, data_dir):
    codes = ",\n".join(
        f"coalesce(try_cast(regexp_extract({field}, '[0-9]+$') as INTEGER), -1) as {field}"
        for field in CATEGORICAL
    )
    for table, filename in (("train", TRAIN_FILE), ("test", TEST_FILE)):
        extra = "", ""
        if table == "train":
            extra = (", cast(sign_time as TIMESTAMP) as sign_ts, cast(y_total_hours as DOUBLE) as y", ", sign_ts, y")
        sql = f"""
            CREATE OR REPLACE TABLE {table} AS
            WITH raw AS (
                SELECT *, cast(pay_time as TIMESTAMP) as pay_ts,
                       cast(issue_time as TIMESTAMP) as issue_ts {extra[0]}
                FROM read_csv(?, header=true, all_varchar=true, strict_mode=true)
            )
            SELECT row_number() over () as rid, pkg_no, lgs_company_nm as carrier,
                   pay_ts, issue_ts {extra[1]}, {codes},
                   try_cast(pkg_goods_cnt as FLOAT) as pkg_goods_cnt,
                   try_cast(weight as FLOAT) as weight,
                   cast(greatest(0, epoch(issue_ts - pay_ts)/3600.0) as FLOAT) as pay_to_issue,
                   cast(dayofweek(pay_ts) as FLOAT) as pay_dow,
                   cast(hour(pay_ts) as FLOAT) as pay_hour,
                   cast(dayofweek(issue_ts) as FLOAT) as issue_dow,
                   cast(hour(issue_ts) as FLOAT) as issue_hour,
                   cast(send_country_nm is not null as FLOAT) as send_country_known
            FROM raw
        """
        db.execute(sql, [str(data_dir / filename)])
        bad_label = " OR sign_ts IS NULL OR y IS NULL OR NOT isfinite(y) OR y <= 0 OR y >= 2160" if table == "train" else ""
        size, unique, bad = db.execute(f"""
            SELECT count(*), count(distinct pkg_no),
                   count(*) FILTER (WHERE pay_ts IS NULL OR issue_ts IS NULL OR
                       carrier IS NULL OR lgs_company_nm < 0 {bad_label})
            FROM {table}
        """).fetchone()
        if size != EXPECTED[table] or size != unique or bad:
            raise ValueError(f"Invalid {table} input: rows={size}, unique IDs={unique}, invalid rows={bad}")
        raw_groups, encoded_groups, first, last = db.execute(f"""
            SELECT count(distinct carrier), count(distinct lgs_company_nm),
                   min(pay_ts)::DATE, max(pay_ts)::DATE FROM {table}
        """).fetchone()
        if raw_groups != encoded_groups:
            raise ValueError(f"Carrier encoding collision in {table}")
        log(f"{table}: {size:,} packages, {raw_groups} carriers, paid {first} to {last}")
    db.execute("CHECKPOINT")


@dataclass
class Frame:
    x: np.ndarray
    y: np.ndarray | None = None
    known: np.ndarray | None = None
    ids: np.ndarray | None = None
    carrier_text: np.ndarray | None = None

    @property
    def carrier(self):
        return self.x[:, 0].astype(np.int32)

    @property
    def duration_floor(self):
        return np.maximum(FLOOR, self.x[:, FEATURES.index("pay_to_issue")])


def read_frame(db, table, where="true", params=(), known_before=None, keep_ids=False):
    columns = list(FEATURES)
    if table == "train":
        columns.append("y")
    if known_before:
        columns.append("sign_ts < CAST(? AS TIMESTAMP) AS known")
        params = (known_before, *params)
    if keep_ids:
        columns.extend(("pkg_no", "carrier"))
    rows = db.execute(
        f"SELECT {', '.join(columns)} FROM {table} WHERE {where} ORDER BY rid", params
    ).fetchnumpy()
    matrix = np.empty((len(rows[FEATURES[0]]), len(FEATURES)), dtype=np.float32)
    for index, column in enumerate(FEATURES):
        values = rows[column]
        if np.ma.isMaskedArray(values):
            values = values.filled(np.nan)
        matrix[:, index] = values
    if not len(matrix):
        raise ValueError(f"No observations in {table}: {where}")
    return Frame(matrix, rows.get("y"), rows.get("known"), rows.get("pkg_no"), rows.get("carrier"))


def training_frame(db, cutoff=None):
    if cutoff:
        return read_frame(db, "train", "pay_ts < CAST(? AS TIMESTAMP) AND sign_ts < CAST(? AS TIMESTAMP)", (cutoff, cutoff))
    return read_frame(db, "train")


def range_frame(db, dates, known_before=None):
    return read_frame(db, "train", "pay_ts >= CAST(? AS TIMESTAMP) AND pay_ts < CAST(? AS TIMESTAMP)", dates, known_before, True)


def empirical_quantile(values, probability):
    return float(np.quantile(values, probability, method="inverted_cdf"))


def historical_baselines(train, valid):
    key_fields = ["carrier", "channel", "state"]
    history = pd.DataFrame(train.x[:, :3].astype(np.int32), columns=key_fields)
    history["hours"] = train.y
    future = pd.DataFrame(valid.x[:, :3].astype(np.int32), columns=key_fields)
    global_q = empirical_quantile(train.y, 0.90)
    by_route = np.full(len(future), global_q)
    for grouping in (["carrier"], ["carrier", "channel"], key_fields):
        groups = history.groupby(grouping, sort=False)["hours"]
        quantile = groups.quantile(0.90).where(groups.size() >= 150)
        joined = future.join(quantile, on=grouping)["hours"].to_numpy()
        by_route = np.where(np.isfinite(joined), joined, by_route)
    return np.full(len(future), global_q), by_route


def fit_quantiles(train, valid, threads, rounds, feature_names=FEATURES):
    predictions = valid.x.copy()
    for index in CAT_INDICES:
        present = np.unique(train.x[:, index])
        unrecognized = ~np.isin(predictions[:, index], present)
        predictions[unrecognized, index] = -1
    dataset = lgb.Dataset(train.x, label=train.y, feature_name=list(feature_names), categorical_feature=CAT_INDICES)
    quantiles = []
    for percentile in (0.90, 0.97):
        start = time.monotonic()
        parameters = {
            "objective": "quantile", "alpha": percentile, "learning_rate": 0.065,
            "num_leaves": 47, "min_data_in_leaf": 140, "max_bin": 127,
            "min_data_per_group": 100, "cat_smooth": 45, "feature_fraction": 0.9,
            "bagging_fraction": 0.8, "bagging_freq": 1, "seed": SEED,
            "num_threads": threads, "verbosity": -1, "deterministic": True,
            "force_col_wise": True,
        }
        booster = lgb.train(parameters, dataset, num_boost_round=rounds)
        quantiles.append(booster.predict(predictions, num_threads=threads))
        log(f"  quantile={percentile:.2f}, fit and forecast {time.monotonic() - start:.1f} s")
        del booster
    return tuple(quantiles)


def finalize(hours, floor):
    return np.round(np.maximum(hours, floor), 4)


def calibrate(y, p90, p97, carrier, mask, targets):
    """Estimate the needed shift for each constraint, with partial pooling."""
    residuals = y - p90, y - p97
    defaults = tuple(empirical_quantile(r[mask], q) for r, q in zip(residuals, targets))
    shifts = {}
    for group in np.unique(carrier[mask]):
        group_mask = mask & (carrier == group)
        size = int(group_mask.sum())
        if size < 40:
            continue
        weight = size / (size + 180)
        shifts[int(group)] = tuple(
            weight * empirical_quantile(r[group_mask], q) + (1 - weight) * typical
            for r, q, typical in zip(residuals, targets, defaults)
        )
    return defaults, shifts


def apply_calibration(p90, p97, carrier, offsets):
    defaults, shifts = offsets
    ontime = p90 + defaults[0]
    severe = p97 + defaults[1] - 24.0
    for group, (first, second) in shifts.items():
        mask = carrier == group
        ontime[mask] = p90[mask] + first
        severe[mask] = p97[mask] + second - 24.0
    return np.maximum(ontime, severe)


def evaluate(y, prediction, carrier_text):
    gap = y - prediction
    data = pd.DataFrame({"carrier": carrier_text, "ontime": gap <= 0, "severe": gap > 24})
    groups = data.groupby("carrier").agg(n=("ontime", "size"), ontime=("ontime", "mean"), severe=("severe", "mean"))
    groups["pen_ontime"] = 500 * (0.90 - groups["ontime"]).clip(lower=0)
    groups["pen_severe"] = 500 * (groups["severe"] - 0.03).clip(lower=0)
    me = abs(float(np.mean(gap)))
    on_pen = float(groups["pen_ontime"].sum())
    sev_pen = float(groups["pen_severe"].sum())
    metrics = dict(
        n=len(y), ME=me, A=me + on_pen + sev_pen, B=me + on_pen,
        pen_ontime=on_pen, pen_severe=sev_pen, overall_ontime=float(np.mean(gap <= 0)),
        overall_severe=float(np.mean(gap > 24)), promise_hours=float(np.mean(prediction)),
        actual_hours=float(np.mean(y)), carriers=int(len(groups)),
        carriers_at_90=int((groups["ontime"] >= 0.90).sum()),
        carriers_at_3=int((groups["severe"] <= 0.03).sum()),
    )
    return metrics, groups.reset_index()


def atomic_csv(path, headers, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)
    tmp.replace(path)


def export_predictions(path, ids, hours):
    if len(ids) != len(hours) or not (np.isfinite(hours) & (hours > 0)).all():
        raise ValueError("Prediction count or positive finite hours check failed")
    atomic_csv(path, ["pkg_no", "pred_hours"], zip(ids, (f"{value:.4f}" for value in hours)))


def check_submission(db, path):
    columns = db.execute("DESCRIBE SELECT * FROM read_csv(?, header=true, all_varchar=true)", [str(path)]).fetchall()
    if [column[0] for column in columns] != ["pkg_no", "pred_hours"]:
        raise ValueError("Submission has unexpected columns")
    counts = db.execute("""
        WITH predictions AS (
            SELECT pkg_no, try_cast(pred_hours AS DOUBLE) AS p
            FROM read_csv(?, header=true, all_varchar=true)
        )
        SELECT count(predictions.pkg_no), count(distinct predictions.pkg_no),
               count(*) FILTER (WHERE test.pkg_no IS NULL),
               count(*) FILTER (WHERE predictions.pkg_no IS NULL),
               count(*) FILTER (WHERE p IS NULL OR NOT isfinite(p) OR p <= 0), min(p), max(p)
        FROM predictions FULL OUTER JOIN test USING (pkg_no)
    """, [str(path)]).fetchone()
    got, unique, extra, missing, invalid, minimum, maximum = counts
    if (got, unique, extra, missing, invalid) != (EXPECTED["test"], EXPECTED["test"], 0, 0, 0):
        raise ValueError(f"Submission validation failed: {counts}")
    return dict(rows=got, unique=unique, extra=extra, missing=missing, invalid=invalid,
                min_hours=minimum, max_hours=maximum)


def solve(db, output, diagnostics, threads, rounds):
    start = time.monotonic()
    log("Development fold")
    dev_train = training_frame(db, DEVELOPMENT[0])
    dev = range_frame(db, DEVELOPMENT, known_before=HOLDOUT[0])
    log(f"  {len(dev_train.y):,} eligible training rows; {len(dev.y):,} development rows")
    dev90, dev97 = fit_quantiles(dev_train, dev, threads, rounds)
    del dev_train
    # Tune and compare margins on separate rows, all signed before the final holdout.
    known_portion = np.flatnonzero(dev.known)
    if len(known_portion) < 1000:
        raise ValueError("Too few signed development labels as of the holdout cutoff")
    tune = dev.known & (np.arange(len(dev.y)) % 2 == 0)
    measure = dev.known & ~tune
    dev_scores = {}
    for name, targets in POLICIES.items():
        calibration = calibrate(dev.y, dev90, dev97, dev.carrier, tune, targets)
        hours = finalize(apply_calibration(dev90, dev97, dev.carrier, calibration), dev.duration_floor)
        dev_scores[name] = evaluate(dev.y[measure], hours[measure], dev.carrier_text[measure])[0]
    policy = min(dev_scores, key=lambda name: dev_scores[name]["A"])
    for name, score in dev_scores.items():
        log(f"  {name}: development A={score['A']:.5f}")
    targets = POLICIES[policy]
    offsets = calibrate(dev.y, dev90, dev97, dev.carrier, dev.known, targets)
    log(f"  chosen policy={policy}, targets={targets}")

    log("Chronological holdout")
    hold_train = training_frame(db, HOLDOUT[0])
    hold = range_frame(db, HOLDOUT)
    holdout_train_size = len(hold_train.y)
    log(f"  {holdout_train_size:,} eligible training rows; {len(hold.y):,} holdout rows")
    global90, grouped90 = historical_baselines(hold_train, hold)
    h90, h97 = fit_quantiles(hold_train, hold, threads, rounds)
    del hold_train
    calibrated = apply_calibration(h90, h97, hold.carrier, offsets)
    candidates = {
        "global_p90": global90, "group_p90": grouped90, "model_p90": h90,
        "dual_quantile": np.maximum(h90, h97 - 24.0), "dual_calibrated": calibrated,
    }
    results = {}
    for name, raw in candidates.items():
        hours = finalize(raw, hold.duration_floor)
        metrics, groups = evaluate(hold.y, hours, hold.carrier_text)
        results[name] = metrics
        groups.to_csv(diagnostics / f"carriers_{name}.csv", index=False, float_format="%.8f")
        log(f"  {name}: A={metrics['A']:.5f}, B={metrics['B']:.5f}, ME={metrics['ME']:.5f}")
    export_predictions(diagnostics / "holdout_prediction.csv", hold.ids, finalize(calibrated, hold.duration_floor))
    atomic_csv(diagnostics / "holdout_truth.csv", ["pkg_no", "lgs_company_nm", "y_total_hours"],
               zip(hold.ids, hold.carrier_text, hold.y))
    pd.DataFrame.from_dict(results, orient="index").rename_axis("method").to_csv(diagnostics / "holdout_metrics.csv")

    log("Final full-data training")
    full = training_frame(db)
    test = read_frame(db, "test", keep_ids=True)
    t90, t97 = fit_quantiles(full, test, threads, rounds)
    del full
    # The rules permit all provided labels for the final fit. Calibrate using
    # out-of-time predictions on the closest labeled payment cohort.
    final_offsets = calibrate(hold.y, h90, h97, hold.carrier, np.ones(len(hold.y), dtype=bool), targets)
    final = finalize(apply_calibration(t90, t97, test.carrier, final_offsets), test.duration_floor)
    result_file = output / "prediction.csv"
    export_predictions(result_file, test.ids, final)
    verification = check_submission(db, result_file)
    log(f"  verified: {verification}")
    info = dict(
        development=list(DEVELOPMENT), holdout=list(HOLDOUT), eligible_holdout_training=holdout_train_size,
        development_known=int(len(known_portion)), development_total=len(dev.y), policy=policy,
        policy_targets=list(targets), development_scores=dev_scores, final_shifts=final_offsets[1],
        final_default_shift=list(final_offsets[0]), holdout_results=results, submission=verification,
        submission_mean=float(np.mean(final)), seed=SEED, threads=threads, rounds=rounds,
        seconds=round(time.monotonic() - start, 1),
    )
    (diagnostics / "run_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory holding the two official CSV files")
    parser.add_argument("--output", type=Path, required=True, help="directory for prediction.csv")
    parser.add_argument("--cache", type=Path, required=True, help="working directory for cache and diagnostics")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=240)
    parser.add_argument("--phase", choices=("all", "prepare", "fit", "check"), default="all")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.cache / "features.duckdb"
    with duckdb.connect(str(db_path), config={"threads": args.threads, "memory_limit": "2GB"}) as db:
        if args.phase in ("all", "prepare"):
            prepare(db, args.data)
        if args.phase in ("all", "fit"):
            solve(db, args.output, args.cache, args.threads, args.rounds)
        if args.phase == "check":
            log(check_submission(db, args.output / "prediction.csv"))


if __name__ == "__main__":
    main()
