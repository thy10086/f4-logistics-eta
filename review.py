"""Score a saved F4 v3 audit window, verify a submission, or reproduce v3 from scratch."""

import argparse
import csv
from gzip import open as gzip_open
from io import TextIOWrapper
import json
import math
import os
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
from zipfile import ZipFile
import zlib

from run_local import set_up_dependencies, sha256


HERE = Path(__file__).resolve().parent
EXPECTED_ROWS = 911_415
EXPECTED_SHA = "248760b9117cc28c47e68ffae163814e46d87b2bd8f03a058a34775a2571f8e9"
SCORER_SHA = "13c47c6e8eafab7822171a1e48e75581a471e208017e504393a850c8254b9a45"


def member(archive, relative):
    items = [entry for entry in archive.infolist() if not entry.is_dir()
             and (entry.filename == relative or entry.filename.endswith("/" + relative))]
    if len(items) != 1:
        raise ValueError(f"Expected exactly one {relative} in the official F4.zip")
    return items[0]


def extract(archive, relative, target):
    entry = member(archive, relative)
    expected = f"{entry.file_size}:{entry.CRC:08x}"
    stamp = target.with_suffix(target.suffix + ".source")
    if (target.is_file() and stamp.is_file() and target.stat().st_size == entry.file_size
            and stamp.read_text(encoding="ascii") == expected):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_suffix(target.suffix + ".partial")
    digest = 0
    with archive.open(entry) as source, stage.open("wb") as destination:
        while data := source.read(2**20):
            digest = zlib.crc32(data, digest)
            destination.write(data)
    if stage.stat().st_size != entry.file_size or digest != entry.CRC:
        raise ValueError(f"CRC or size mismatch for {relative}")
    stage.replace(target)
    stamp.write_text(expected, encoding="ascii")


def validate(kit, prediction):
    expected_ids = set()
    hours_min, hours_max = math.inf, 0.
    with prediction.open(encoding="utf-8-sig", newline="") as file:
        data = csv.reader(file)
        if next(data, None) != ["pkg_no", "pred_hours"]:
            raise ValueError("Prediction header must be pkg_no,pred_hours in that order")
        for row_number, row in enumerate(data, 2):
            if len(row) != 2 or not row[0] or row[0] in expected_ids:
                raise ValueError(f"Missing or duplicate pkg_no, or invalid columns at row {row_number}")
            val = float(row[1])
            if not math.isfinite(val) or val <= 0:
                raise ValueError(f"Invalid pred_hours at row {row_number}")
            expected_ids.add(row[0])
            hours_min, hours_max = min(hours_min, val), max(hours_max, val)
    if len(expected_ids) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} unique predictions; found {len(expected_ids):,}")
    with ZipFile(kit) as archive, archive.open(member(archive, "data/test_202609.csv")) as binary:
        reader = csv.reader(TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
        headers = next(reader)
        index = headers.index("pkg_no")
        test_rows = 0
        for row in reader:
            if len(row) != len(headers) or row[index] not in expected_ids:
                raise ValueError(f"An official test package is duplicated, missing from the predictions, or malformed at row {test_rows + 2}")
            expected_ids.remove(row[index])
            test_rows += 1
    if test_rows != EXPECTED_ROWS or expected_ids:
        raise ValueError("The prediction CSV and official test set differ")
    actual_sha = sha256(prediction)
    if actual_sha != EXPECTED_SHA:
        raise ValueError(f"The predictions cover the test set, but the locked v3 SHA-256 differs: {actual_sha}")
    return dict(rows=test_rows, unique=test_rows, missing=0, extra=0, invalid=0, min_hours=hours_min,
                max_hours=hours_max, test_has_truth="y_total_hours" in headers, sha256=actual_sha)


def quick(kit, work):
    work.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((HERE / "evidence/verification.json").read_text(encoding="utf-8"))
    truth = work / "hold_audit_truth.csv"
    forecast = work / "hold_audit_prediction.csv"
    scorer = work / "score.py"
    dataset = work / "train_202608_0920.csv"
    print("Preparing the official training CSV and scoring script …", flush=True)
    with ZipFile(kit) as archive:
        extract(archive, "tools/score.py", scorer)
        extract(archive, "data/train_202608_0920.csv", dataset)
    if sha256(scorer) != SCORER_SHA:
        raise ValueError("The official scoring script does not match the verified contest version")
    stage = forecast.with_suffix(".partial")
    with gzip_open(HERE / "evidence/hold_audit_prediction.csv.gz", "rb") as source, stage.open("wb") as dest:
        copyfileobj(source, dest)
    if sha256(stage) != evidence["hold_audit_prediction_sha256"]:
        raise ValueError("The saved audit predictions do not match the evidence")
    stage.replace(forecast)
    env = set_up_dependencies(work / "python-deps")
    sys.path.insert(0, str(work / "python-deps"))
    import duckdb

    print("Matching the saved 09/12–09/20 audit IDs to official training truth …", flush=True)
    expected = next(row for row in evidence["checks"] if row["name"] == "hold_audit")
    joined = truth.with_suffix(".partial.csv")
    with duckdb.connect(config={"threads": 4, "memory_limit": "2GB"}) as db:
        matched_rows = db.sql("""
            SELECT t.pkg_no, t.lgs_company_nm, t.y_total_hours
            FROM read_csv(?, header=true, all_varchar=true) AS t
            INNER JOIN read_csv(?, header=true, all_varchar=true) AS p USING (pkg_no)
        """, params=[str(dataset), str(forecast)])
        matched_rows.write_csv(str(joined), header=True)
        actual = db.execute("SELECT count(*), count(DISTINCT pkg_no) FROM read_csv(?, header=true)", [str(joined)]).fetchone()
        if actual != (expected["n"], expected["n"]):
            raise ValueError(f"Expected {expected['n']} complete audit labels, found {actual}")
    joined.replace(truth)
    run = subprocess.run([sys.executable, "-u", str(scorer), "--truth", str(truth), "--pred", str(forecast)],
                         env=env, check=True, capture_output=True, encoding="utf-8")
    (work / "hold_audit_official_score.txt").write_text(run.stdout, encoding="utf-8")
    metrics = {key: float(value.replace(",", "")) for key, value in map(str.split, run.stdout.splitlines())}
    wanted = {key: expected[key] for key in ("ME", "B", "A", "pen_ontime", "pen_severe")}
    if metrics != wanted:
        raise ValueError(f"Original score.py returned {metrics}, expected {wanted}")
    print(f"\nLocal validation | 09/12–09/20 hash audit | {expected['n']:,} labelled training packages")
    print(run.stdout.rstrip())
    print("All five metrics match the frozen evidence.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    for name in ("score", "full", "verify"):
        command = phases.add_parser(name)
        command.add_argument("--kit", type=Path, required=True)
        if name == "verify":
            command.add_argument("--prediction", type=Path, required=True)
            command.add_argument("--json", type=Path)
        else:
            command.add_argument("--work-dir", type=Path, default=Path("f4-score-run" if name == "score" else "f4-full-run"))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use CPython 3.12")
    kit = args.kit.resolve(strict=True)
    if args.phase == "verify":
        result = validate(kit, args.prediction.resolve(strict=True))
        text = json.dumps(result, indent=2) + "\n"
        print(text, end="", flush=True)
        if args.json:
            args.json.write_text(text, encoding="utf-8")
    elif args.phase == "score":
        quick(kit, args.work_dir.resolve())
    else:
        work = args.work_dir.resolve()
        subprocess.run([sys.executable, "-X", "utf8", "-u", str(HERE / "run_local.py"), "--kit",
                        str(kit), "--work-dir", str(work)], check=True)
        result = validate(kit, work / "result/prediction.csv")
        result["interpreter"] = sys.version.split()[0]
        result["kit_sha256"] = sha256(kit)
        report = json.dumps(result, indent=2) + "\n"
        (work / "verified_submission.json").write_text(report, encoding="utf-8")
        print(report, end="", flush=True)


if __name__ == "__main__":
    main()
