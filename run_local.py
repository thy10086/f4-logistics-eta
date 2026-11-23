"""Reproduce F4 v3 from the official ZIP and this source package."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfileobj
import subprocess
import sys
from zipfile import ZipFile


SOURCE = Path(__file__).resolve().parent
KIT_FILES = ("data/train_202608_0920.csv", "data/test_202609.csv", "tools/score.py")
SOURCE_FILES = ("solution.py", "matched.py", "history_model.py", "adaptive.py", "release.py", "carrier_policy.py", "requirements.txt")
SCORES = ("stress", "hold", "stress_audit", "hold_audit")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(label, command, env):
    print(f"\n[{label}]", flush=True)
    subprocess.run([sys.executable, "-u", *map(str, command)], cwd=SOURCE, env=env, check=True)


def extract_kit(path, target):
    with ZipFile(path) as kit:
        for relative in KIT_FILES:
            names = [item for item in kit.infolist() if not item.is_dir() and
                     (item.filename == relative or item.filename.endswith("/" + relative))]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one {relative} in the official F4 ZIP")
            entry = names[0]
            output = target / relative
            if output.is_file() and output.stat().st_size == entry.file_size:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            partial = output.with_suffix(output.suffix + ".partial")
            print(f"Extracting {relative} ({entry.file_size:,} bytes)", flush=True)
            with kit.open(entry) as source, partial.open("wb") as destination:
                copyfileobj(source, destination, length=2**20)
            if partial.stat().st_size != entry.file_size:
                raise ValueError(f"Extracted size differs for {relative}")
            partial.replace(output)


def set_up_dependencies(store):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONUTF8="1")
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(store) if not existing_path else os.pathsep.join((str(store), existing_path))
    expected = dict(line.split("==", maxsplit=1) for line in (SOURCE / "requirements.txt").read_text().splitlines() if line.strip())
    check = (
        "from importlib import import_module; from importlib.metadata import version; import sys; "
        f"expected = {expected!r}; "
        "sys.exit(0 if all(version(name) == pin and import_module(name) for name, pin in expected.items()) else 1)"
    )
    probe = [sys.executable, "-c", check]
    health = subprocess.run(probe, env=env, capture_output=True)
    if health.returncode:
        store.mkdir(parents=True, exist_ok=True)
        run("Install pinned Python dependencies", ["-m", "pip", "install", "--disable-pip-version-check", "--upgrade",
                                                   "--target", store, "-r", SOURCE / "requirements.txt"], env)
        subprocess.run(probe, env=env, check=True)
    return env


def official_scores(kit, checks, env):
    scores = {}
    summary = json.loads((checks / "summary.json").read_text(encoding="utf-8"))
    for name in SCORES:
        process = subprocess.run(
            [sys.executable, str(kit / "tools/score.py"), "--truth", str(checks / f"{name}_truth.csv"),
             "--pred", str(checks / f"{name}_prediction.csv")],
            env=env, check=True, stdout=subprocess.PIPE, encoding="utf-8",
        )
        (checks / f"{name}_official_score.txt").write_text(process.stdout, encoding="utf-8")
        print(f"\n[Official score: {name}]\n{process.stdout.rstrip()}", flush=True)
        values = {key: float(value.replace(",", "")) for key, value in map(str.split, process.stdout.splitlines())}
        kind = "audit" if name.endswith("_audit") else "windows"
        metrics = summary[kind][name.removesuffix("_audit")]
        fields = {"ME", "A", "B", "pen_ontime", "pen_severe"}
        if set(values) != fields or any(abs(values[key] - metrics[key]) > .000051 for key in fields):
            raise ValueError(f"Official and project scores differ for {name}")
        scores[name] = values
    return scores, summary["submission"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True, help="path to the official F4.zip")
    parser.add_argument("--work-dir", type=Path, default=Path("f4-local-run"), help="where to write data, models, and results")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 for this reproduction")
    source_kit = args.kit.resolve()
    if not source_kit.is_file():
        parser.error(f"File not found: {source_kit}")
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    kit_hash = sha256(source_kit)
    code = hashlib.sha256("".join(sha256(SOURCE / name) for name in SOURCE_FILES).encode()).hexdigest()
    cache = work / "cache" / f"{kit_hash[:12]}-{code[:12]}"
    kit = cache / "official"
    extract_kit(source_kit, kit)
    env = set_up_dependencies(work / "python-deps")

    db = cache / "features.duckdb"
    output = work / "result"
    checks = work / "check"
    prepared = cache / "prepared.ok"
    if not (db.is_file() and prepared.is_file()):
        run("Prepare features", [SOURCE / "solution.py", "--data", kit / "data", "--cache", cache,
                                 "--output", output, "--phase", "prepare"], env)
        prepared.touch()
    run("Fit original quantile models", [SOURCE / "matched.py", "--db", db, "--output", cache / "original",
                                         "--threads", "6", "--rounds", "240"], env)
    run("Fit historical-feature models", [SOURCE / "history_model.py", "--db", db, "--output", cache / "history",
                                           "--phase", "all", "--threads", "6", "--rounds", "240"], env)
    run("Calibrate and export predictions", [SOURCE / "carrier_policy.py", "--db", db, "--cache", cache / "original",
                                             "--advanced", cache / "history", "--diagnostics", checks, "--output", output], env)
    scores, submission = official_scores(kit, checks, env)
    prediction = output / "prediction.csv"
    digest = sha256(prediction)
    expected = json.loads((SOURCE / "evidence/verification.json").read_text(encoding="utf-8"))["sha256"]
    check = dict(prediction_sha256=digest, reference_sha256=expected, matches_reference_csv=digest == expected,
                 official_scores=scores, submission=submission)
    (work / "reproduction.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(f"\nPrediction: {prediction}\nSHA-256: {digest}\nMatches reference CSV: {digest == expected}\n", flush=True)


if __name__ == "__main__":
    main()
