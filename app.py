"""
Ocean Metabolome API: a read-only data API, with no UI and no login.

Serves marine metabolomics datasets imported from GNPS2 (FBMN or Everything Bagel):
features, samples and abundance, plus the data behind each analysis (Van Krevelen,
carbon oxidation state, metabolite box plots, PCoA). Any client (an AI agent, a
script, a notebook) reads the data through these endpoints and draws its own plots.

Datasets are added on the server, not through the API: see datasets.yaml and
import_datasets.py.

Run:   uvicorn app:app --reload --port 8001
Docs:  http://localhost:8001/docs   (interactive)
Spec:  http://localhost:8001/openapi.json   (what an AI agent reads to learn the API)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import analysis
from importer import DATA_DIR

# ---------------------------------------------------------------------------
# Reading stored datasets. Every cache is keyed on the dataset's version (the
# modification time of its info.json), so a re-import shows up on the next request
# without restarting the server.
# ---------------------------------------------------------------------------

def _path(dataset_id: str, name: str) -> Path:
    if dataset_id.startswith("."):
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    path = DATA_DIR / dataset_id / name
    if not path.exists():
        if (DATA_DIR / dataset_id / "info.json").exists():
            # imported before this analysis existed
            raise HTTPException(409, f"Dataset '{dataset_id}' was imported before this analysis was added. "
                                     "Re-import it on the server: python import_datasets.py --force " + dataset_id)
        raise HTTPException(404, f"Dataset '{dataset_id}' not found.")
    return path


def _version(dataset_id: str) -> int:
    """Changes whenever the dataset is re-imported (info.json is written last)."""
    return _path(dataset_id, "info.json").stat().st_mtime_ns


def versioned(maxsize: int):
    """lru_cache whose key includes the dataset's version (first argument is dataset_id)."""
    def wrap(fn):
        cached = lru_cache(maxsize=maxsize)(lambda ver, *args: fn(*args))
        def call(dataset_id, *args):
            return cached(_version(dataset_id), dataset_id, *args)
        call.__doc__, call.__name__ = fn.__doc__, fn.__name__
        return call
    return wrap


@versioned(8)
def _chem_abundance(dataset_id: str) -> np.ndarray:
    """features-with-formula x samples peak areas (rows in chemistry.json order)."""
    return np.load(_path(dataset_id, "chemistry_abundance.npy"))


@versioned(4)
def _distance(dataset_id: str) -> np.ndarray:
    return np.load(_path(dataset_id, "distance_braycurtis.npy"))


@versioned(8)
def _pcoa_base(dataset_id: str) -> dict:
    return json.loads(_path(dataset_id, "pcoa.json").read_text())


@versioned(32)
def _load(dataset_id: str, table: str) -> pd.DataFrame:
    """Read a stored table (features / annotations / metadata / chemistry...) once and keep it in memory."""
    return pd.read_json(_path(dataset_id, f"{table}.json"), orient="records", convert_dates=False)


@versioned(32)
def _abundance_index(dataset_id: str) -> tuple[list[str], dict[int, int]]:
    d = json.loads(_path(dataset_id, "abundance_index.json").read_text())
    return d["samples"], {int(k): v for k, v in d["offsets"].items()}


def _read_abundance(dataset_id: str, feature_ids: list[int]) -> dict[int, list[float]]:
    """Peak area in every sample (same order as the sample list) per feature, read from disk."""
    samples, offsets = _abundance_index(dataset_id)
    out = {}
    with open(_path(dataset_id, "abundance.jsonl"), "rb") as f:
        for fid in feature_ids:
            if fid in offsets:
                f.seek(offsets[fid])
                rec = json.loads(f.readline())
                dense = [0.0] * len(samples)
                for i, v in zip(rec["i"], rec["v"]):
                    dense[i] = v
                out[fid] = dense
    return out


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))  # NaN -> null


def _sample_values(dataset_id: str, column: str, samples: list[str]) -> pd.Series:
    """One metadata column for the given samples (in that order), as text; missing -> NaN."""
    md = _load(dataset_id, "metadata")
    if column not in md.columns or column == "sample":
        cols = [c for c in md.columns if c != "sample"]
        raise HTTPException(422, f"Unknown metadata column '{column}'. Columns: {', '.join(map(str, cols))}")
    if "sample" not in md.columns:
        raise HTTPException(409, "This dataset has no sample metadata.")
    s = md.drop_duplicates("sample").set_index("sample")[column].reindex(samples)
    s = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "" else str(v).strip())
    return s


def _is_numeric(values: pd.Series) -> bool:
    """Mostly numbers and more than 12 distinct values (else treated as categories)."""
    v = values.dropna()
    nums = pd.to_numeric(v, errors="coerce")
    return bool(len(v) > 0 and nums.notna().mean() >= 0.9 and v.nunique() > 12)


@versioned(32)
def _pcoa_subset(dataset_id: str, filter_by: Optional[str], filter_values: tuple) -> dict:
    base = _pcoa_base(dataset_id)
    if not filter_by:
        return {"samples": base["samples"], "coords": np.asarray(base["coords"]),
                "proportion_explained": base["proportion_explained"], "index": np.arange(len(base["samples"]))}
    vals = _sample_values(dataset_id, filter_by, base["samples"])
    idx = np.flatnonzero(vals.isin(filter_values).to_numpy())
    if len(idx) < 3:
        raise HTTPException(422, f"Only {len(idx)} samples have {filter_by} in {list(filter_values)}; PCoA needs at least 3.")
    D = _distance(dataset_id)[np.ix_(idx, idx)]
    coords, explained = analysis.pcoa(D, dims=10)
    return {"samples": [base["samples"][i] for i in idx], "coords": coords,
            "proportion_explained": [float(x) for x in explained], "index": idx}


@versioned(32)
def _permanova_cached(dataset_id: str, filter_by: Optional[str], filter_values: tuple,
                      group_by: str, permutations: int) -> dict:
    sub = _pcoa_subset(dataset_id, filter_by, filter_values)
    groups = _sample_values(dataset_id, group_by, sub["samples"])
    ok = groups.notna().to_numpy()
    g = groups[ok]
    counts = g.value_counts()
    if len(counts) < 2 or counts.min() < 2:
        return {"test": "PERMANOVA", "skipped": f"needs 2+ groups of {group_by} with 2+ samples each "
                f"(found {len(counts)} groups, smallest has {int(counts.min()) if len(counts) else 0})"}
    idx = sub["index"][ok]
    res = analysis.permanova(_distance(dataset_id)[np.ix_(idx, idx)], g.to_numpy(), permutations=permutations)
    res["group_by"] = group_by
    return res


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ocean Metabolome API",
    version="0.2.0",
    description=(
        "Marine metabolomics data from GNPS2 (FBMN / Everything Bagel). "
        "Each dataset has features (m/z, RT, annotation name), samples (metadata incl. "
        "latitude/longitude) and an abundance matrix (feature x sample peak areas), plus "
        "analysis endpoints that return the data behind Van Krevelen diagrams, carbon "
        "oxidation states, metabolite box plots and PCoA. Clients draw their own plots.\n\n"
        "Read-only and public: no sign-in, CORS open to any origin."
    ),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


@app.get("/", tags=["info"])
def hello():
    """What this API is, where its docs are, and which datasets are loaded."""
    return {"message": "Ocean Metabolome API: read-only marine metabolomics data from GNPS2.",
            "docs": "/docs", "openapi": "/openapi.json",
            "datasets": [d["dataset_id"] for d in list_datasets()]}


@app.get("/datasets", tags=["datasets"])
def list_datasets():
    """All imported datasets, with feature/sample counts."""
    return [json.loads(p.read_text()) for p in sorted(DATA_DIR.glob("*/info.json"))
            if not p.parent.name.startswith(".")]


@app.get("/datasets/{dataset_id}", tags=["datasets"])
def get_dataset(dataset_id: str):
    """Summary of one dataset."""
    return json.loads(_path(dataset_id, "info.json").read_text())


@app.get("/datasets/{dataset_id}/samples", tags=["samples"])
def get_samples(dataset_id: str):
    """Sample metadata (one row per sample), including latitude/longitude when present."""
    return _records(_load(dataset_id, "metadata"))


@app.get("/datasets/{dataset_id}/features", tags=["features"])
def get_features(
    dataset_id: str,
    annotated_only: bool = False,
    search: Optional[str] = Query(None, description="Case-insensitive substring of the annotation name"),
    sort: Literal["feature_id", "total_abundance", "mean_abundance"] = Query(
        "feature_id", description="Abundance sorts are highest first"),
    limit: int = Query(100, le=10000),
    offset: int = 0,
):
    """Features (row ID, m/z, RT, annotation name), plus total and mean abundance across samples."""
    df = _load(dataset_id, "features")
    if sort != "feature_id":
        df = df.sort_values(sort, ascending=False)
    if annotated_only:
        df = df[df["name"].notna()]
    if search:
        df = df[df["name"].fillna("").str.contains(search, case=False, regex=False)]
    return {"total": len(df), "offset": offset, "features": _records(df.iloc[offset: offset + limit])}


@app.get("/datasets/{dataset_id}/annotations", tags=["features"])
def get_annotations(dataset_id: str):
    """Best library match per feature (name, score, adduct, SMILES...)."""
    return _records(_load(dataset_id, "annotations"))


@app.get("/datasets/{dataset_id}/features/{feature_id}/abundance", tags=["features"])
def get_feature_abundance(dataset_id: str, feature_id: int):
    """Abundance of one feature in every sample, joined with that sample's metadata (incl. lat/lon).
    This is the shape a map or box plot needs."""
    samples, _ = _abundance_index(dataset_id)
    values = _read_abundance(dataset_id, [feature_id]).get(feature_id)
    if values is None:
        raise HTTPException(404, f"Feature {feature_id} not in dataset '{dataset_id}'.")
    row = pd.DataFrame({"sample": samples, "abundance": values})
    md = _load(dataset_id, "metadata")
    if "sample" in md.columns:
        row = row.merge(md, on="sample", how="left")
    feat = _load(dataset_id, "features").set_index("feature_id").loc[feature_id]
    name = feat.get("name")
    return {
        "feature_id": feature_id,
        "name": None if pd.isna(name) else name,
        "mz": float(feat["mz"]),
        "rt": float(feat["rt"]),
        "samples": _records(row),
    }


@app.get("/datasets/{dataset_id}/abundance", tags=["features"])
def get_abundance_matrix(
    dataset_id: str,
    feature_ids: str = Query(..., description="Comma-separated feature IDs, e.g. 1,2,3"),
):
    """Abundance matrix for selected features (rows) x all samples (columns)."""
    try:
        ids = [int(x) for x in feature_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(422, "feature_ids must be comma-separated integers, e.g. 1,2,3")
    if len(ids) > 1000:
        raise HTTPException(422, "At most 1000 feature_ids per request.")
    samples, _ = _abundance_index(dataset_id)
    rows = _read_abundance(dataset_id, ids)
    found = [i for i in ids if i in rows]
    return {
        "features": found,
        "samples": samples,
        "data": [rows[i] for i in found],
    }


# ---------------------------------------------------------------------------
# Analyses: each endpoint returns the data behind one kind of plot.
# The client (the portal page, an AI agent, a notebook) decides how to draw it.
# ---------------------------------------------------------------------------

def _group_mask(dataset_id: str, samples: list[str], group_by: Optional[str], group: Optional[str]):
    if not group_by:
        return None
    if group is None:
        raise HTTPException(422, "Give `group` (a value of the group_by column) together with `group_by`.")
    mask = (_sample_values(dataset_id, group_by, samples) == group).to_numpy()
    if not mask.any():
        raise HTTPException(422, f"No samples have {group_by} = '{group}'.")
    return mask


@app.get("/datasets/{dataset_id}/van-krevelen", tags=["analyses"])
def van_krevelen(
    dataset_id: str,
    group_by: Optional[str] = Query(None, description="Metadata column to restrict samples by (with `group`)"),
    group: Optional[str] = Query(None, description="Value of `group_by`: only features detected in these samples, with their abundance there"),
):
    """Van Krevelen data: O/C and H/C of every annotated feature with a molecular formula,
    plus its NOSC, compound class (from the O/C-H/C regions in `regions`) and NPClassifier /
    ClassyFire classes when the workflow provides them. `abundance` is the total peak area
    (in the chosen group's samples when `group_by`/`group` are given)."""
    chem = _load(dataset_id, "chemistry").copy()
    samples, _ = _abundance_index(dataset_id)
    mask = _group_mask(dataset_id, samples, group_by, group)
    if mask is None:
        chem["abundance"] = chem["total_abundance"]
        n_samples = len(samples)
    else:
        A = _chem_abundance(dataset_id)[:, mask]
        chem["abundance"] = A.sum(axis=1).astype(float)
        chem["n_samples_detected"] = (A > 0).sum(axis=1)
        chem = chem[chem["abundance"] > 0]
        n_samples = int(mask.sum())
    chem = chem.drop(columns=["total_abundance"])
    return {
        "dataset_id": dataset_id, "group_by": group_by, "group": group, "n_samples": n_samples,
        "n_features": len(chem), "x": "o_c", "y": "h_c", "regions": analysis.VK_REGIONS,
        "class_counts": {k: int(v) for k, v in chem["compound_class"].value_counts().items()},
        "features": _records(chem),
    }


@app.get("/datasets/{dataset_id}/carbon-oxidation", tags=["analyses"])
def carbon_oxidation(
    dataset_id: str,
    group_by: Optional[str] = Query(None, description="Metadata column to compare sample NOSC across"),
    max_groups: int = Query(30, ge=2, le=200),
):
    """Nominal oxidation state of carbon (NOSC = 4 - (4C + H - 3N - 2O + 5P - 2S) / C).

    - `features`: NOSC per annotated feature with a formula, with m/z and compound class
      (e.g. NOSC vs m/z, or a histogram).
    - `samples`: per sample, the abundance-weighted mean NOSC (and O/C, H/C) over those features.
    - with `group_by`: box-plot statistics of the per-sample NOSC in each group, plus a
      Kruskal-Wallis test (Mann-Whitney U for two groups)."""
    chem = _load(dataset_id, "chemistry")
    sc = _load(dataset_id, "sample_chemistry").copy()
    out = {
        "dataset_id": dataset_id, "formula": "NOSC = 4 - (4C + H - 3N - 2O + 5P - 2S) / C",
        "features": _records(chem[["feature_id", "name", "formula", "mz", "nosc", "compound_class", "total_abundance"]]),
    }
    if group_by:
        sc[group_by] = _sample_values(dataset_id, group_by, sc["sample"].tolist()).to_numpy()
        valid = sc.dropna(subset=["nosc_weighted", group_by])
        groups = {k: v["nosc_weighted"].to_numpy() for k, v in valid.groupby(group_by)}
        names = sorted(groups, key=lambda k: -len(groups[k]))
        shown = names[:max_groups]
        out.update({
            "group_by": group_by, "n_groups": len(names), "groups_shown": len(shown),
            "groups": [{"group": k, **analysis.box_stats(groups[k], count_detected=False), "values": groups[k].tolist()} for k in shown],
            "test": analysis.compare_groups({k: groups[k] for k in shown}),
            "n_samples_missing_group": int(sc[group_by].isna().sum()),
        })
    out["samples"] = _records(sc)
    return out


@app.get("/datasets/{dataset_id}/features/{feature_id}/boxplot", tags=["analyses"])
def feature_boxplot(
    dataset_id: str,
    feature_id: int,
    group_by: str = Query(..., description="Metadata column whose values form the boxes"),
    max_groups: int = Query(30, ge=2, le=200, description="Largest groups kept; the test uses the same groups"),
    include_values: bool = Query(True, description="Include each sample's value (for drawing points)"),
):
    """Metabolite box plot: one feature's peak area in each group of a metadata column, with
    quartiles per group and a Kruskal-Wallis test across the groups (Mann-Whitney U for two).
    Samples without a value for `group_by` are left out."""
    samples, _ = _abundance_index(dataset_id)
    values = _read_abundance(dataset_id, [feature_id]).get(feature_id)
    if values is None:
        raise HTTPException(404, f"Feature {feature_id} not in dataset '{dataset_id}'.")
    labels = _sample_values(dataset_id, group_by, samples)
    df = pd.DataFrame({"sample": samples, "value": values, "group": labels.to_numpy()}).dropna(subset=["group"])
    groups = {k: g for k, g in df.groupby("group")}
    names = sorted(groups, key=lambda k: -len(groups[k]))
    shown = names[:max_groups]
    feat = _load(dataset_id, "features").set_index("feature_id").loc[feature_id]
    name = feat.get("name")
    return {
        "dataset_id": dataset_id, "feature_id": feature_id, "name": None if pd.isna(name) else name,
        "mz": float(feat["mz"]), "rt": float(feat["rt"]), "group_by": group_by,
        "n_groups": len(names), "groups_shown": len(shown),
        "n_samples_missing_group": len(samples) - len(df),
        "groups": [
            {"group": k, **analysis.box_stats(groups[k]["value"]),
             **({"values": groups[k]["value"].tolist(), "samples": groups[k]["sample"].tolist()} if include_values else {})}
            for k in shown
        ],
        "test": analysis.compare_groups({k: groups[k]["value"].to_numpy() for k in shown}),
    }


@app.get("/datasets/{dataset_id}/pcoa", tags=["analyses"])
def pcoa_endpoint(
    dataset_id: str,
    color_by: Optional[str] = Query(None, description="Metadata column to return with each sample (and test with PERMANOVA)"),
    filter_by: Optional[str] = Query(None, description="Metadata column to subset samples by; the PCoA is recomputed on the subset"),
    filter_value: Optional[list[str]] = Query(None, description="Values of `filter_by` to keep (repeat the parameter for several)"),
    dims: int = Query(3, ge=2, le=10, description="Number of axes returned"),
    permutations: int = Query(999, ge=0, le=9999, description="PERMANOVA permutations; 0 skips the test"),
):
    """PCoA of samples on Bray-Curtis dissimilarity (relative abundance, all features),
    computed at import. With `filter_by` + `filter_value` it is recomputed on that subset.
    With a categorical `color_by`, a PERMANOVA tests whether its groups differ."""
    if bool(filter_by) != bool(filter_value):
        raise HTTPException(422, "Give `filter_by` and `filter_value` together.")
    fv = tuple(sorted(filter_value)) if filter_value else ()
    sub = _pcoa_subset(dataset_id, filter_by or None, fv)
    base = _pcoa_base(dataset_id)
    axes = [f"PC{i + 1}" for i in range(min(dims, sub["coords"].shape[1]))]
    rows = pd.DataFrame(np.asarray(sub["coords"])[:, :len(axes)], columns=axes)
    rows.insert(0, "sample", sub["samples"])
    out = {
        "dataset_id": dataset_id, "metric": base["metric"], "normalization": base["normalization"],
        "filter_by": filter_by, "filter_value": list(fv) or None, "n_samples": len(rows),
        "axes": [{"axis": a, "proportion_explained": sub["proportion_explained"][i]} for i, a in enumerate(axes)],
        "proportion_explained": sub["proportion_explained"],
        "color_by": color_by, "permanova": None,
    }
    if color_by:
        vals = _sample_values(dataset_id, color_by, sub["samples"])
        rows[color_by] = vals.to_numpy()
        out["color_by_numeric"] = _is_numeric(vals)
        if out["color_by_numeric"]:
            rows[color_by] = pd.to_numeric(rows[color_by], errors="coerce")
            out["permanova"] = {"test": "PERMANOVA", "skipped": f"{color_by} is numeric; PERMANOVA compares groups"}
        elif permutations:
            out["permanova"] = _permanova_cached(dataset_id, filter_by or None, fv, color_by, permutations)
    out["samples"] = _records(rows)
    return out

