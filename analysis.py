"""
Chemistry and statistics behind the portal's analysis endpoints.

- Molecular formulas per annotation (library formula, else InChI, else SMILES via RDKit)
- Van Krevelen ratios (O/C, H/C), compound class regions, and the nominal oxidation
  state of carbon (NOSC; LaRowe & Van Cappellen 2011)
- Bray-Curtis distances between samples, PCoA, PERMANOVA
- Group comparisons for metabolite box plots (Kruskal-Wallis / Mann-Whitney U)

Kept free of FastAPI so it can be tested and reused on its own.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from scipy import linalg, stats

try:  # RDKit is optional: without it, SMILES-only annotations just get no formula
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except ImportError:  # pragma: no cover
    HAVE_RDKIT = False

try:  # numba makes Bray-Curtis over 100k features x 2.4k samples take ~1 min instead of ~30
    import numba
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

_MISSING = frozenset({"", "nan", "none", "null", "n/a", "na", "-", "--", "?", "unknown"})


def _text(v) -> Optional[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return None if s.lower() in _MISSING else s


def formula_from_inchi(inchi) -> Optional[str]:
    s = _text(inchi)
    if s and s.startswith("InChI=") and "/" in s:
        layer = s.split("/")[1].strip()
        return layer or None
    return None


@lru_cache(maxsize=65536)
def formula_from_smiles(smiles: str) -> Optional[str]:
    if not HAVE_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else None
    except Exception:
        return None


def resolve_formula(formula=None, inchi=None, smiles=None) -> tuple[Optional[str], Optional[str]]:
    """(formula, source) from the first usable of: library formula, InChI, SMILES."""
    f = _text(formula)
    if f and element_counts(f).get("C"):
        return f, "library"
    f = formula_from_inchi(inchi)
    if f and element_counts(f).get("C"):
        return f, "inchi"
    s = _text(smiles)
    f = formula_from_smiles(s) if s else None
    if f and element_counts(f).get("C"):
        return f, "smiles"
    return None, None


def element_counts(formula: str) -> dict:
    """Atom counts of a formula such as 'C6H12O6'. For salts/hydrates ('C37H36N2O9S3.2H3N')
    only the part with the most carbon (the organic molecule) is counted."""
    if not isinstance(formula, str) or not formula:
        return {}
    best: dict = {}
    for part in formula.split("."):
        counts: dict = {}
        for elem, n in re.findall(r"([A-Z][a-z]?)(\d*)", part):
            counts[elem] = counts.get(elem, 0) + (int(n) if n else 1)
        if counts.get("C", 0) > best.get("C", 0):
            best = counts
    return best


def nosc(counts: dict) -> Optional[float]:
    """Nominal oxidation state of carbon of a neutral molecule:
    NOSC = 4 - (4C + H - 3N - 2O + 5P - 2S) / C."""
    c = counts.get("C", 0)
    if not c:
        return None
    h, n, o, p, s = (counts.get(e, 0) for e in ("H", "N", "O", "P", "S"))
    return 4 - (4 * c + h - 3 * n - 2 * o + 5 * p - 2 * s) / c


# Van Krevelen compound class regions (O/C, H/C). The first match wins; features in
# no box fall back to "Alkaloids / Organonitrogen" (has N) or "Other".
VK_REGIONS = [
    {"name": "Lipids", "o_c": [0.0, 0.3], "h_c": [1.5, 2.5]},
    {"name": "Amino acids / Peptides", "o_c": [0.3, 0.6], "h_c": [1.5, 2.2], "requires_n": True},
    {"name": "Carbohydrates", "o_c": [0.6, 1.2], "h_c": [1.5, 2.5]},
    {"name": "Condensed aromatics", "o_c": [0.0, 0.1], "h_c": [0.0, 0.5]},
    {"name": "Lignins", "o_c": [0.1, 0.6], "h_c": [0.5, 1.5]},
    {"name": "Unsaturated hydrocarbons", "o_c": [0.0, 0.1], "h_c": [1.0, 1.5]},
]


def compound_class(o_c: float, h_c: float, n_atoms: int) -> str:
    for r in VK_REGIONS:
        if r.get("requires_n") and not n_atoms:
            continue
        if r["o_c"][0] <= o_c <= r["o_c"][1] and r["h_c"][0] <= h_c <= r["h_c"][1]:
            return r["name"]
    return "Alkaloids / Organonitrogen" if n_atoms else "Other"


def chemistry_row(formula: str) -> dict:
    counts = element_counts(formula)
    c = counts.get("C", 0)
    o_c, h_c = counts.get("O", 0) / c, counts.get("H", 0) / c
    return {
        "o_c": o_c, "h_c": h_c, "nosc": nosc(counts),
        "compound_class": compound_class(o_c, h_c, counts.get("N", 0)),
        "n_c": c, "n_n": counts.get("N", 0), "n_s": counts.get("S", 0), "n_p": counts.get("P", 0),
    }


# ---------------------------------------------------------------------------
# Bray-Curtis, PCoA, PERMANOVA
# ---------------------------------------------------------------------------

def _min_sums_numpy(indptr, idx, val, n):
    S = np.zeros((n, n))
    for f in range(len(indptr) - 1):
        i, v = idx[indptr[f]:indptr[f + 1]], val[indptr[f]:indptr[f + 1]]
        if len(i):
            S[np.ix_(i, i)] += np.minimum.outer(v, v)
    return S


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _min_sums(indptr, idx, val, n):  # pragma: no cover - compiled
        # S[a, b] = sum over features of min(x_a, x_b), visiting only the samples each
        # feature occurs in. Parallel over sample a, so threads never write the same row.
        nf = len(indptr) - 1
        counts = np.zeros(n + 1, np.int64)
        for p in range(len(idx)):
            counts[idx[p] + 1] += 1
        sptr = np.cumsum(counts)
        sfeat = np.empty(len(idx), np.int64)
        spos = np.empty(len(idx), np.int64)
        fill = sptr[:-1].copy()
        for f in range(nf):
            for p in range(indptr[f], indptr[f + 1]):
                s = idx[p]
                sfeat[fill[s]] = f
                spos[fill[s]] = p
                fill[s] += 1
        S = np.zeros((n, n))
        for a in numba.prange(n):
            row = S[a]
            for q in range(sptr[a], sptr[a + 1]):
                f = sfeat[q]
                va = val[spos[q]]
                for p in range(indptr[f], indptr[f + 1]):
                    vb = val[p]
                    row[idx[p]] += va if va < vb else vb
        return S
else:  # pragma: no cover
    _min_sums = _min_sums_numpy


def bray_curtis_relative(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bray-Curtis dissimilarity between samples on relative abundance.

    `matrix` is features x samples (peak areas). Each sample is divided by its total, so
    BC(a, b) = 1 - sum_k min(p_ak, p_bk). Exact, over all features; it only visits the
    non-zero entries, which is what makes 100k features feasible.
    Returns (D for the kept samples, boolean mask of kept samples = total > 0).
    """
    m = np.asarray(matrix, dtype=np.float64)
    totals = m.sum(axis=0)
    keep = totals > 0
    m = m[:, keep] / totals[keep]
    rows, cols = np.nonzero(m)                      # row-major: grouped by feature
    indptr = np.zeros(m.shape[0] + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=m.shape[0]), out=indptr[1:])
    S = _min_sums(indptr, cols.astype(np.int64), m[rows, cols], int(keep.sum()))
    D = np.clip(1.0 - S, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D, keep


def pcoa(D: np.ndarray, dims: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Classical PCoA. Returns (coordinates n x dims, proportion explained per axis),
    with proportions relative to the trace of the centred matrix, as scikit-bio does."""
    n = D.shape[0]
    dims = max(1, min(dims, n - 1))
    A = -0.5 * np.asarray(D, dtype=np.float64) ** 2
    A -= A.mean(axis=0, keepdims=True)
    A -= A.mean(axis=1, keepdims=True)              # double centring (A is symmetric)
    trace = float(np.trace(A))
    vals, vecs = linalg.eigh(A, subset_by_index=[n - dims, n - 1])
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = np.clip(vals, 0, None)
    coords = vecs * np.sqrt(pos)
    # sign convention: largest-magnitude loading positive, so axes don't flip between runs
    flip = np.sign(coords[np.abs(coords).argmax(axis=0), np.arange(coords.shape[1])])
    coords *= np.where(flip == 0, 1, flip)
    return coords, (vals / trace if trace else np.zeros_like(vals))


def _within_ss_numpy(D2, labels, sizes):
    G = np.eye(len(sizes))[labels]
    return float((np.einsum("ij,ik,jk->k", D2, G, G) / 2 / sizes).sum())


if HAVE_NUMBA:
    @numba.njit(parallel=True, cache=True)
    def _within_ss_many(D2, label_sets, sizes):  # pragma: no cover - compiled
        # one within-group sum of squares per row of label_sets, permutations in parallel
        out = np.zeros(label_sets.shape[0])
        n = D2.shape[0]
        for p in numba.prange(label_sets.shape[0]):
            labels = label_sets[p]
            acc = np.zeros(len(sizes))
            for i in range(n):
                gi = labels[i]
                row = D2[i]
                for j in range(i + 1, n):
                    if labels[j] == gi:
                        acc[gi] += row[j]
            out[p] = (acc / sizes).sum()
        return out
else:  # pragma: no cover
    def _within_ss_many(D2, label_sets, sizes):
        return np.array([_within_ss_numpy(D2, l, sizes) for l in label_sets])


def permanova(D: np.ndarray, groups, permutations: int = 999, seed: int = 0) -> dict:
    """PERMANOVA (Anderson 2001) on a distance matrix: pseudo-F, permutation p-value, R²."""
    codes, names = pd.factorize(pd.Series(groups).astype(str))
    n, k = len(codes), len(names)
    sizes = np.bincount(codes, minlength=k).astype(np.float64)
    D2 = np.asarray(D, dtype=np.float64) ** 2
    ss_total = D2[np.triu_indices(n, 1)].sum() / n
    rng = np.random.default_rng(seed)
    label_sets = np.vstack([codes] + [rng.permutation(codes) for _ in range(permutations)]).astype(np.int64)
    ss_w = _within_ss_many(D2, label_sets, sizes)
    f = ((ss_total - ss_w) / (k - 1)) / (ss_w / (n - k))
    f_stat = float(f[0])
    hits = int((f[1:] >= f_stat).sum())
    return {
        "test": "PERMANOVA", "pseudo_F": f_stat,
        "p_value": (hits + 1) / (permutations + 1), "R2": float(1 - ss_w[0] / ss_total),
        "permutations": permutations, "n_samples": n, "n_groups": k,
    }


# ---------------------------------------------------------------------------
# Group comparisons
# ---------------------------------------------------------------------------

def box_stats(values, count_detected: bool = True) -> dict:
    """Quartiles etc. `n_detected` (values > 0) only makes sense for peak areas."""
    v = np.asarray(values, dtype=np.float64)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {
        "n": int(v.size), **({"n_detected": int((v > 0).sum())} if count_detected else {}),
        "mean": float(v.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
        "min": float(v.min()), "max": float(v.max()),
    }


def compare_groups(groups: dict) -> Optional[dict]:
    """Kruskal-Wallis across all groups (Mann-Whitney U when there are exactly two).
    Needs at least two groups with 2+ values; returns None otherwise."""
    usable = {k: np.asarray(v, float) for k, v in groups.items() if len(v) >= 2}
    if len(usable) < 2:
        return None
    vals = list(usable.values())
    if np.ptp(np.concatenate(vals)) == 0:
        return {"test": "Kruskal-Wallis", "statistic": 0.0, "p_value": 1.0, "n_groups": len(vals),
                "note": "All values are identical."}
    if len(vals) == 2:
        r = stats.mannwhitneyu(vals[0], vals[1], alternative="two-sided")
        return {"test": "Mann-Whitney U", "statistic": float(r.statistic), "p_value": float(r.pvalue), "n_groups": 2}
    r = stats.kruskal(*vals)
    return {"test": "Kruskal-Wallis", "statistic": float(r.statistic), "p_value": float(r.pvalue), "n_groups": len(vals)}
