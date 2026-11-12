# Ocean Metabolome API

A read-only data API for marine metabolomics data from GNPS2 (FBMN or Everything Bagel).
There is no web interface and no login: the endpoints return the data behind each plot
(Van Krevelen, carbon oxidation state, metabolite box plots, PCoA, ...), and whatever
calls them (an AI agent, a script, a notebook) draws the plots.

Production: **https://ocean-metabolome-portal.gnps2.org** (Docker on GNPS2's servers).

- Interactive docs: `/docs`
- OpenAPI spec (give this to an AI agent): `/openapi.json`

## Run

With Docker (local, http://localhost:8001):

```bash
make server-compose           # or: make server-compose-interactive (foreground)
```

Without Docker:

```bash
uv venv && source .venv/bin/activate      # or: python3 -m venv .venv (needs python3-venv)
uv pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

## Add a dataset

Datasets are added on the server, not through the API, so the API needs no passwords.

1. Add the GNPS2 task to [`datasets.yaml`](datasets.yaml):
   ```yaml
   - dataset_id: my_dataset          # name used in the URLs
     task_id: 454547be77d94ae1ad00d798046ee7a0
     workflow: fbmn                  # fbmn or eb (Everything Bagel)
     description: Optional text shown in GET /datasets
   ```
2. Run the import:
   ```bash
   make import                       # in Docker; imports new or changed entries only
   python import_datasets.py         # without Docker
   ```

The running API picks up the new data on its next request; no restart needed.
`datasets.yaml` is mounted into the container, so editing it needs no rebuild.

| Command | Does |
|---|---|
| `make import` | import entries that are new, or whose task/workflow changed |
| `make import ARGS="go_dom_eb"` | only that dataset |
| `make import ARGS="--force"` | re-import everything (e.g. after changing the import code) |
| `make import ARGS="--list"` | show each dataset and whether it is imported |

Imports take ~30 s (FBMN, 16k features × 2.4k samples) to ~4 min (EB, 112k features × 2.4k
samples) and briefly need several GB of RAM. A failed import leaves the previous version of
that dataset in place. To remove a dataset, delete its entry and its `data/<dataset_id>/` folder.

## Endpoints

All endpoints are `GET`, public, and CORS-open. Metadata columns (for `group_by`, `color_by`,
`filter_by`) are listed in `GET /datasets/{id}` as `metadata_columns`.

| Endpoint | Returns |
|---|---|
| `/` | what this is, links to the docs, loaded datasets |
| `/datasets` | all datasets with counts |
| `/datasets/{id}` | one dataset's summary |
| `/datasets/{id}/samples` | sample metadata, including lat/lon |
| `/datasets/{id}/features?search=&annotated_only=&sort=&limit=&offset=` | features with name and mean/total abundance |
| `/datasets/{id}/annotations` | best library match per feature: formula (and its source), SMILES, InChI, compound classes |
| `/datasets/{id}/features/{feature_id}/abundance` | one feature in every sample, joined with metadata (map-ready) |
| `/datasets/{id}/abundance?feature_ids=1,2,3` | abundance matrix for chosen features |
| `/datasets/{id}/van-krevelen?group_by=&group=` | O/C, H/C, NOSC and compound class of every annotated feature with a formula, plus the class regions; optionally only features detected in one group of samples |
| `/datasets/{id}/carbon-oxidation?group_by=` | NOSC per feature; abundance-weighted NOSC per sample; with `group_by`, box statistics per group + Kruskal-Wallis test |
| `/datasets/{id}/features/{feature_id}/boxplot?group_by=` | one metabolite's peak area per group: quartiles, values, Kruskal-Wallis (Mann-Whitney U for 2 groups) |
| `/datasets/{id}/pcoa?color_by=&filter_by=&filter_value=&dims=&permutations=` | PCoA coordinates (Bray-Curtis), variance explained, PERMANOVA on a categorical `color_by`; recomputed on a subset with `filter_by` + `filter_value` (repeatable) |

Example: `GET /datasets/ocean_fbmn/pcoa?color_by=ATTRIBUTE_ecosystem` returns 2,439 samples
with PC1–PC3 and `"permanova": {"pseudo_F": 149.1, "R2": 0.155, "p_value": 0.001, ...}`.

### How the analyses are computed

- **Formulas**: the library match's formula when given, else the InChI formula layer,
  else computed from SMILES with RDKit. Salts/hydrates count only the organic part.
- **Van Krevelen**: atomic O/C and H/C. Compound classes from O/C-H/C regions (lipids,
  amino acids/peptides, carbohydrates, condensed aromatics, lignins, unsaturated
  hydrocarbons; otherwise alkaloids/organonitrogen if N, else other). FBMN tasks also
  carry NPClassifier and ClassyFire classes.
- **NOSC** (LaRowe & Van Cappellen 2011): 4 − (4C + H − 3N − 2O + 5P − 2S) / C.
  Per sample: mean NOSC of the features with a formula, weighted by peak area.
- **PCoA**: Bray-Curtis on relative abundance (each sample divided by its total peak area),
  over all features, computed exactly at import; classical PCoA (scikit-bio's proportions).
- **PERMANOVA**: pseudo-F with 999 permutations (default), R² = 1 − SS_within / SS_total.

## Stored files (`data/<dataset_id>/`)

| File | Contents |
|---|---|
| `info.json` | counts, metadata columns, formula sources, import time, warnings |
| `features.json` | per feature: `feature_id`, `mz`, `rt`, `name`, `total_abundance`, `mean_abundance` |
| `annotations.json` | best library match per feature, with `formula` and `formula_source` |
| `metadata.json` | one record per sample, with `latitude` / `longitude` when present |
| `abundance.jsonl` + `abundance_index.json` | feature × sample peak areas, one line per feature (non-zero values only), with byte offsets so one feature is read with a single seek |
| `chemistry.json` | per feature with a formula: O/C, H/C, NOSC, compound class |
| `chemistry_abundance.npy` | peak areas of those features × samples |
| `sample_chemistry.json` | per sample: abundance-weighted NOSC, O/C, H/C |
| `distance_braycurtis.npy`, `pcoa.json` | sample × sample Bray-Curtis distances, PCoA coordinates |

## Code

| File | Role |
|---|---|
| `app.py` | the API (read-only) |
| `analysis.py` | formulas, NOSC, Van Krevelen classes, Bray-Curtis, PCoA, PERMANOVA, group tests |
| `importer.py` | downloads a GNPS2 task and writes `data/<dataset_id>/` |
| `import_datasets.py` + `datasets.yaml` | which datasets to import, and the command that imports them |

## Deploy (ocean-metabolome-portal.gnps2.org)

1. On the GNPS2 server: `make server-compose-production`. It joins the `nginx-net` network
   of nginx-proxy, which serves `ocean-metabolome-portal.gnps2.org` with a Let's Encrypt certificate.
2. `make import` to load the datasets in `datasets.yaml`.
