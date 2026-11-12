"""
Import the datasets listed in datasets.yaml.

    python import_datasets.py                  # import new or changed entries
    python import_datasets.py go_dom_eb        # only these datasets
    python import_datasets.py --force          # re-import even if up to date
    python import_datasets.py --list           # show what is listed and what is imported

An entry is up to date when data/<dataset_id>/ was imported from the same task ID and
workflow with the current import code (it has every analysis). A failed import leaves
the previous version of that dataset in place.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml

from importer import DATA_DIR, WORKFLOW_FILES, import_task

CONFIG = Path(__file__).parent / "datasets.yaml"
ANALYSES = ["van-krevelen", "carbon-oxidation", "boxplot", "pcoa"]


def load_config(path: Path = CONFIG) -> list[dict]:
    entries = yaml.safe_load(path.read_text()) or []
    if not isinstance(entries, list):
        sys.exit(f"{path}: expected a list of datasets")
    seen = set()
    for i, e in enumerate(entries, 1):
        where = f"{path.name} entry {i}"
        if not isinstance(e, dict):
            sys.exit(f"{where}: expected dataset_id / task_id / workflow")
        e["dataset_id"], e["task_id"] = str(e.get("dataset_id", "")).strip(), str(e.get("task_id", "")).strip()
        e["workflow"] = str(e.get("workflow", "fbmn")).strip().lower()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", e["dataset_id"]):
            sys.exit(f"{where}: dataset_id must use letters, digits, _ or - (got {e['dataset_id']!r})")
        if not re.fullmatch(r"[A-Za-z0-9]{8,64}", e["task_id"]):
            sys.exit(f"{where}: task_id doesn't look like a GNPS2 task ID (got {e['task_id']!r})")
        if e["workflow"] not in WORKFLOW_FILES:
            sys.exit(f"{where}: workflow must be one of {', '.join(WORKFLOW_FILES)} (got {e['workflow']!r})")
        if e["dataset_id"] in seen:
            sys.exit(f"{where}: dataset_id {e['dataset_id']!r} is listed twice")
        seen.add(e["dataset_id"])
    return entries


def current(dataset_id: str) -> dict | None:
    path = DATA_DIR / dataset_id / "info.json"
    return json.loads(path.read_text()) if path.exists() else None


def up_to_date(entry: dict) -> bool:
    info = current(entry["dataset_id"])
    return bool(info and info.get("task_id") == entry["task_id"] and info.get("workflow") == entry["workflow"]
                and set(ANALYSES) <= set(info.get("analyses", [])))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_ids", nargs="*", help="only import these (default: all in datasets.yaml)")
    ap.add_argument("--force", action="store_true", help="re-import even if already up to date")
    ap.add_argument("--list", action="store_true", help="show the datasets and whether they are imported")
    args = ap.parse_args()

    entries = load_config()
    if args.dataset_ids:
        unknown = set(args.dataset_ids) - {e["dataset_id"] for e in entries}
        if unknown:
            sys.exit(f"Not in {CONFIG.name}: {', '.join(sorted(unknown))}. Add them there first.")
        entries = [e for e in entries if e["dataset_id"] in args.dataset_ids]

    if args.list:
        for e in entries:
            info = current(e["dataset_id"])
            state = ("up to date" if up_to_date(e) else "needs import") + (
                f" (imported {info.get('imported_at', '?')})" if info else " (not imported)")
            print(f"{e['dataset_id']:<24} {e['workflow']:<5} {e['task_id']}  {state}")
        listed = {e["dataset_id"] for e in load_config()}
        extra = sorted(p.parent.name for p in DATA_DIR.glob("*/info.json")
                       if not p.parent.name.startswith(".") and p.parent.name not in listed)
        if extra:
            print(f"\nImported but not in {CONFIG.name} (still served): {', '.join(extra)}")
        return 0

    failed = []
    for e in entries:
        name = e["dataset_id"]
        if not args.force and up_to_date(e):
            print(f"{name}: up to date, skipped (--force to re-import)")
            continue
        print(f"{name}: importing GNPS2 task {e['task_id']} ({e['workflow']})", flush=True)
        t0 = time.time()
        try:
            info = import_task(name, e["task_id"], e["workflow"],
                               progress=lambda step: print(f"  {time.time() - t0:6.0f}s  {step}", flush=True))
        except Exception as exc:
            print(f"{name}: FAILED: {exc}", flush=True)
            failed.append(name)
            continue
        if e.get("description"):
            info["description"] = str(e["description"])
            (DATA_DIR / name / "info.json").write_text(json.dumps(info, indent=2))
        print(f"{name}: done in {time.time() - t0:.0f}s: {info['n_features']:,} features, {info['n_samples']:,} samples, "
              f"{info['n_with_formula']:,} with a formula" + (f"; warnings: {'; '.join(info['warnings'])}" if info["warnings"] else ""),
              flush=True)
    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
