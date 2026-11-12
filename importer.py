"""
Import GNPS2 tasks into the Ocean Metabolome API's data folder.

Two ways in, one pipeline:

- import_task(): downloads a GNPS2 task (FBMN or Everything Bagel). Used by
  `make import` (datasets.yaml) and by POST /datasets/import.
- import_files(): reads tables uploaded through POST /datasets/upload.

Both hand three tables to build_dataset(), which derives everything the API serves
(tables, formulas, Van Krevelen / NOSC, sparse abundance matrix, Bray-Curtis distances,
PCoA) and swaps it into data/<dataset_id>/. The running API notices the new files on
its next request.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

import analysis

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# GNPS2 download
# ---------------------------------------------------------------------------

GNPS2_SERVERS = ["https://gnps2.org", "https://beta.gnps2.org", "https://de.gnps2.org"]  # tried in order
HEADERS = {"User-Agent": "ocean-metabolome-api/0.1"}  # GNPS rejects Python's default UA

# Where each workflow keeps its three tables: (path, separator)
WORKFLOW_FILES = {
    "fbmn": {
        "features": ("nf_output/clustering/featuretable_reformated.csv", ","),
        "annotations": ("nf_output/library/merged_results_with_gnps.tsv", "\t"),
        "metadata": ("nf_output/metadata/merged_metadata.tsv", "\t"),
    },
    "eb": {
        "features": ("nf_output/feature_finding/feature_finding_results/aligned_features.csv", ","),
        "annotations": ("nf_output/feature_library_search/merged_feature_library_search_results.tsv", "\t"),
        "metadata": ("nf_output/metadata/merged_metadata.tsv", "\t"),
    },
}

# EB names its annotation columns differently; rename to the FBMN names.
EB_ANNOTATION_RENAME = {
    "query_scan": "#Scan#", "NAME": "Compound_Name", "cosine": "MQScore", "FORMULA": "molecular_formula",
    "SMILES": "Smiles", "ADDUCT": "Adduct", "SPECTRUMID": "SpectrumID",
}
# Annotation columns kept (when present), and their names in annotations.json
ANNOTATION_COLUMNS = {
    "#Scan#": "feature_id", "Compound_Name": "name", "MQScore": "score", "Adduct": "adduct",
    "molecular_formula": "library_formula", "Smiles": "smiles", "INCHI": "inchi", "SpectrumID": "spectrum_id",
    # compound classes (FBMN only): NPClassifier and ClassyFire
    "npclassifier_pathway": "np_pathway", "npclassifier_superclass": "np_superclass", "npclassifier_class": "np_class",
    "superclass": "classyfire_superclass", "class": "classyfire_class",
}

LAT_NAMES = ["ATTRIBUTE_Latitude", "Latitude", "latitude", "lat"]
LON_NAMES = ["ATTRIBUTE_Longitude", "Longitude", "longitude", "lon", "long"]


def _fetch(task_id: str, path: str) -> bytes:
    """Download one task file, falling back to the GNPS2 mirrors if a server fails."""
    errors = []
    for server in GNPS2_SERVERS:
        url = f"{server}/resultfile?task={task_id}&file={quote(path, safe='')}"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.read()
        except Exception as exc:
            errors.append(f"{server}: {exc}")
    raise RuntimeError("; ".join(errors))


def _fetch_table(task_id: str, path: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_fetch(task_id, path)), sep=sep, on_bad_lines="skip")


def read_table_file(path: Path) -> pd.DataFrame:
    """An uploaded table: comma-separated if the name ends in .csv (optionally .csv.gz),
    else tab-separated. Gzip is detected from the name."""
    name = path.name.lower().removesuffix(".gz")
    return pd.read_csv(path, sep="," if name.endswith(".csv") else "\t", on_bad_lines="skip")


def _fetch_eb_metadata(task_id: str) -> pd.DataFrame:
    """EB keeps the uploaded metadata in a folder named after the task parameter."""
    params = yaml.safe_load(_fetch(task_id, "job_parameters.yaml")) or {}
    for param in ("metadata_file", "metadata_filename"):
        value = str(params.get(param) or "").strip()
        if value and value != "NO_FILE":
            name = value.rsplit("/", 1)[-1]
            return _fetch_table(task_id, f"{param}/{name}", "," if name.endswith(".csv") else "\t")
    path, sep = WORKFLOW_FILES["eb"]["metadata"]
    return _fetch_table(task_id, path, sep)


def _sample_name(column: str) -> str:
    """'S1.mzML Peak area' -> 'S1.mzML' so feature-table columns match metadata filenames."""
    return re.sub(r"\s*Peak area$", "", str(column)).strip()


def _first_column(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n.lower()] for n in names if n.lower() in lower), None)


# ---------------------------------------------------------------------------
# Import: GNPS2 -> tidy tables -> JSON on disk
# ---------------------------------------------------------------------------

def import_task(dataset_id: str, task_id: str, workflow: str,
                progress: Callable[[str], None] = lambda step: None, extra: Optional[dict] = None) -> dict:
    """Download a GNPS2 task and build data/<dataset_id>/ from it (see build_dataset)."""
    files = WORKFLOW_FILES[workflow]
    started = time.time()
    warnings: list[str] = []
    progress("Downloading the feature table")
    ft = _fetch_table(task_id, *files["features"])
    progress("Downloading the annotation table")
    try:
        an = _fetch_table(task_id, *files["annotations"])
    except Exception as exc:
        warnings.append(f"No annotation table loaded: {exc}")
        an = None
    progress("Downloading the metadata table")
    try:
        md = _fetch_eb_metadata(task_id) if workflow == "eb" else _fetch_table(task_id, *files["metadata"])
    except Exception as exc:
        warnings.append(f"No metadata table loaded: {exc}")
        md = None
    source = {"source": "gnps2_task", "task_id": task_id, "workflow": workflow}
    return build_dataset(dataset_id, ft, an, md, {**source, **(extra or {})}, warnings, progress, started)


def import_files(dataset_id: str, feature_table: Path, annotations: Optional[Path] = None,
                 metadata: Optional[Path] = None, progress: Callable[[str], None] = lambda step: None,
                 extra: Optional[dict] = None) -> dict:
    """Build data/<dataset_id>/ from uploaded files (see build_dataset for the formats)."""
    started = time.time()
    warnings: list[str] = []
    progress("Reading the feature table")
    ft = read_table_file(feature_table)
    an = md = None
    if annotations:
        progress("Reading the annotation table")
        an = read_table_file(annotations)
    if metadata:
        progress("Reading the metadata table")
        md = read_table_file(metadata)
    source = {"source": "upload", "task_id": None, "workflow": None,
              "files": {k: p.name for k, p in [("feature_table", feature_table), ("annotations", annotations),
                                               ("metadata", metadata)] if p}}
    return build_dataset(dataset_id, ft, an, md, {**source, **(extra or {})}, warnings, progress, started)


def build_dataset(dataset_id: str, ft: pd.DataFrame, an: Optional[pd.DataFrame], md: Optional[pd.DataFrame],
                  extra: dict, warnings: list[str], progress: Callable[[str], None], started: float) -> dict:
    """Derive everything the endpoints serve and store it in data/<dataset_id>/. Built in a
    staging folder and swapped in at the end, so a failed re-import leaves the existing
    dataset untouched.

    - ft: feature table in the MZmine / FBMN quantification format: "row ID", "row m/z",
      "row retention time" and one "<sample> Peak area" column per sample (required).
    - an: library matches, GNPS2 FBMN or Everything Bagel column names (optional).
    - md: sample metadata with a "filename" (or "sample") column (optional).
    """
    # 1. Feature table (required): one row per feature, one "Peak area" column per sample
    area_cols = [c for c in ft.columns if str(c).endswith("Peak area")]
    missing = [c for c in ("row ID", "row m/z", "row retention time") if c not in ft.columns]
    if missing or not area_cols:
        raise ValueError("Feature table must be in the MZmine / FBMN quantification format: columns "
                         "'row ID', 'row m/z', 'row retention time' and one '<sample> Peak area' column "
                         f"per sample. Missing: {', '.join(missing + ([] if area_cols else ['<sample> Peak area']))}.")
    ft = ft.dropna(subset=["row ID"]).drop_duplicates("row ID")
    features = ft[["row ID", "row m/z", "row retention time"]].rename(
        columns={"row ID": "feature_id", "row m/z": "mz", "row retention time": "rt"}
    )
    abundance = ft[area_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    abundance.columns = [_sample_name(c) for c in area_cols]
    abundance.index = ft["row ID"].astype(int)
    # Precompute per-feature summaries so listing features never has to read the matrix
    features["total_abundance"] = abundance.sum(axis=1).to_numpy()
    features["mean_abundance"] = abundance.mean(axis=1).to_numpy()

    # 2. Annotation table (optional): library match per feature, with its molecular formula
    progress("Processing the annotation table")
    try:
        if an is None:
            raise ValueError("none given")
        # Everything Bagel names its columns differently; FBMN names win when both exist
        an = an.rename(columns={k: v for k, v in EB_ANNOTATION_RENAME.items() if v not in an.columns})
        if "#Scan#" not in an.columns:
            raise ValueError("no '#Scan#' (or EB 'query_scan') column linking matches to feature IDs")
        keep = [c for c in ANNOTATION_COLUMNS if c in an.columns]
        an = an[keep].rename(columns=ANNOTATION_COLUMNS)
        # keep the best-scoring match per feature
        if "score" in an.columns:
            an = an.sort_values("score", ascending=False)
        an = an.drop_duplicates("feature_id")
        an["feature_id"] = an["feature_id"].astype(int)
    except Exception as exc:
        if not any(w.startswith("No annotation table") for w in warnings):
            warnings.append(f"No annotation table loaded: {exc}")
        an = pd.DataFrame(columns=["feature_id", "name"])
    # formula: library formula, else from InChI, else from SMILES (RDKit)
    resolved = [analysis.resolve_formula(r.get("library_formula"), r.get("inchi"), r.get("smiles"))
                for r in an.to_dict("records")]
    an["formula"] = [f for f, _ in resolved]
    an["formula_source"] = [s for _, s in resolved]
    features = features.merge(an[["feature_id", "name"]], on="feature_id", how="left")

    # 3. Metadata table (optional): one row per sample, incl. Latitude / Longitude
    progress("Processing the metadata table")
    if md is not None:
        md = md.rename(columns={"filename": "sample"})
        # drop empty columns pandas creates from trailing tabs ("Unnamed: 15")
        md = md.loc[:, ~md.columns.astype(str).str.startswith("Unnamed:")]
        if "sample" in md.columns:
            md["sample"] = md["sample"].astype(str).str.strip()
            matched = int(md["sample"].isin(abundance.columns).sum())
            if matched == 0:
                warnings.append("No metadata 'filename' matches a feature-table sample column; "
                                "filters by metadata will match nothing.")
        else:
            warnings.append("Metadata table has no 'filename' column; it was not used.")
            md = None
    if md is None:
        md = pd.DataFrame({"sample": abundance.columns})
    lat, lon = _first_column(md, LAT_NAMES), _first_column(md, LON_NAMES)
    if lat and lon:
        md = md.rename(columns={lat: "latitude", lon: "longitude"})
        md["latitude"] = pd.to_numeric(md["latitude"], errors="coerce")
        md["longitude"] = pd.to_numeric(md["longitude"], errors="coerce")

    # 4. Save as JSON.
    progress("Writing the tables")
    out = DATA_DIR / f".staging-{dataset_id}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    features.to_json(out / "features.json", orient="records")
    an.to_json(out / "annotations.json", orient="records")
    md.to_json(out / "metadata.json", orient="records")

    # Abundance can be 100k features x 2k+ samples and is mostly zeros, so it is stored as
    # a sparse matrix (features x samples, float32): one file the API loads once and then
    # slices or multiplies in milliseconds (filters, map values, box plots).
    samples = list(abundance.columns)
    save_abundance(out, abundance.to_numpy(dtype=np.float32), samples, abundance.index.to_numpy())

    # 5. Chemistry of each annotated feature with a formula (Van Krevelen, carbon oxidation)
    progress("Computing formulas, Van Krevelen ratios and carbon oxidation states")
    chem = an[an["formula"].notna()].copy()
    chem = chem[chem["feature_id"].isin(abundance.index)]
    chem = pd.concat([chem.reset_index(drop=True),
                      pd.DataFrame([analysis.chemistry_row(f) for f in chem["formula"]])], axis=1)
    for col in ("name", "o_c", "h_c", "nosc", "compound_class", "n_c", "n_n", "n_s", "n_p"):
        if col not in chem.columns:  # no formulas at all (e.g. an upload without annotations)
            chem[col] = pd.Series(dtype=object if col in ("name", "compound_class") else float)
    chem = chem.merge(features[["feature_id", "mz", "rt", "total_abundance"]], on="feature_id", how="left")
    class_cols = [c for c in ["np_pathway", "np_superclass", "np_class", "classyfire_superclass", "classyfire_class"] if c in chem.columns]
    chem = chem[["feature_id", "name", "formula", "formula_source", "mz", "rt", "total_abundance",
                 "o_c", "h_c", "nosc", "compound_class", "n_c", "n_n", "n_s", "n_p"] + class_cols]
    chem.to_json(out / "chemistry.json", orient="records")
    # abundance of just these features (rows in chemistry.json order) for per-group views
    chem_ab = abundance.loc[chem["feature_id"]].to_numpy(dtype=np.float64)
    np.save(out / "chemistry_abundance.npy", chem_ab.astype(np.float32))
    # per sample: abundance-weighted mean NOSC / O/C / H/C over the features with a formula
    totals = chem_ab.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = {k: np.where(totals > 0, (chem[k].to_numpy()[:, None] * chem_ab).sum(axis=0) / totals, np.nan)
                    for k in ("nosc", "o_c", "h_c")}
    pd.DataFrame({
        "sample": samples, "nosc_weighted": weighted["nosc"], "o_c_weighted": weighted["o_c"],
        "h_c_weighted": weighted["h_c"], "n_formula_features": (chem_ab > 0).sum(axis=0),
    }).to_json(out / "sample_chemistry.json", orient="records")

    # 6. Bray-Curtis distances between samples (relative abundance, all features) and PCoA
    progress("Computing Bray-Curtis distances between samples")
    D, kept = analysis.bray_curtis_relative(abundance.to_numpy())
    np.save(out / "distance_braycurtis.npy", D.astype(np.float32))
    progress("Computing the PCoA")
    coords, explained = analysis.pcoa(D, dims=10)
    pcoa_samples = [s for s, k in zip(samples, kept) if k]
    (out / "pcoa.json").write_text(json.dumps({
        "metric": "braycurtis", "normalization": "relative abundance (each sample divided by its total peak area)",
        "samples": pcoa_samples, "coords": np.round(coords, 6).tolist(),
        "proportion_explained": [float(x) for x in explained],
    }))
    if not kept.all():
        warnings.append(f"{int((~kept).sum())} samples with no detected features are left out of the PCoA.")

    info = {
        "dataset_id": dataset_id,
        "task_id": extra.get("task_id"),
        "workflow": extra.get("workflow"),
        "n_features": len(features),
        "n_samples": abundance.shape[1],
        "n_annotated": int(features["name"].notna().sum()),
        "metadata_columns": list(md.columns),
        "has_coordinates": bool(lat and lon),
        "n_samples_with_coordinates": int(md[["latitude", "longitude"]].notna().all(axis=1).sum()) if lat and lon else 0,
        "n_with_formula": len(chem),
        "formula_sources": {k: int(v) for k, v in chem["formula_source"].value_counts().items()},
        "n_samples_in_pcoa": len(pcoa_samples),
        "analyses": ["van-krevelen", "carbon-oxidation", "boxplot", "pcoa"],
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "import_seconds": round(time.time() - started, 1),
        "warnings": warnings,
        **{k: v for k, v in extra.items() if k not in ("task_id", "workflow")},
    }
    (out / "info.json").write_text(json.dumps(info, indent=2))

    # swap the finished dataset in, replacing any older import of the same id
    final, old = DATA_DIR / dataset_id, DATA_DIR / f".old-{dataset_id}"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    out.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    return info



# ---------------------------------------------------------------------------
# Sparse abundance matrix
# ---------------------------------------------------------------------------

def save_abundance(folder: Path, matrix: np.ndarray | sp.spmatrix, samples: list[str], feature_ids) -> None:
    """abundance.npz: features x samples CSR (float32); abundance_index.json: the sample
    names (columns) and feature IDs (rows) in matrix order."""
    sp.save_npz(folder / "abundance.npz", sp.csr_matrix(matrix, dtype=np.float32), compressed=False)
    (folder / "abundance_index.json").write_text(json.dumps(
        {"samples": list(samples), "feature_ids": [int(x) for x in feature_ids]}))


def upgrade_abundance(folder: Path) -> bool:
    """Convert a dataset imported with the old abundance.jsonl layout to abundance.npz
    (no re-download). Returns True if it converted something."""
    if (folder / "abundance.npz").exists() or not (folder / "abundance.jsonl").exists():
        return False
    samples = json.loads((folder / "abundance_index.json").read_text())["samples"]
    ids, indptr, indices, data = [], [0], [], []
    with open(folder / "abundance.jsonl", "rb") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["feature_id"])
            indices.append(np.asarray(rec["i"], dtype=np.int32))
            data.append(np.asarray(rec["v"], dtype=np.float32))
            indptr.append(indptr[-1] + len(rec["i"]))
    m = sp.csr_matrix((np.concatenate(data) if data else np.zeros(0, np.float32),
                       np.concatenate(indices) if indices else np.zeros(0, np.int32),
                       np.asarray(indptr, dtype=np.int64)), shape=(len(ids), len(samples)))
    save_abundance(folder, m, samples, ids)
    (folder / "abundance.jsonl").unlink()
    return True
