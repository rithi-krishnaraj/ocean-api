"""
Background imports for POST /datasets/import and POST /datasets/upload.

Jobs run one at a time, in order. Each runs in its own child process: an Everything
Bagel import briefly needs ~5 GB, which the server process would never give back, and
running out of memory then fails the import instead of killing the API.

Job records live in memory: a server restart forgets finished jobs (the datasets they
imported stay) and fails nothing that was still queued, it just drops it.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from importer import DATA_DIR

UPLOAD_DIR = DATA_DIR / ".uploads"
KEEP_FINISHED = 200

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_queue: "queue.Queue[str]" = queue.Queue()
_worker: Optional[threading.Thread] = None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def busy(dataset_id: str) -> Optional[dict]:
    """The queued or running job for this dataset, if any."""
    with _lock:
        return next((j for j in _jobs.values() if j["dataset_id"] == dataset_id
                     and j["status"] in ("queued", "running")), None)


def new_upload_dir() -> tuple[str, Path]:
    job_id = uuid.uuid4().hex[:12]
    folder = UPLOAD_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)
    return job_id, folder


def submit(kind: str, dataset_id: str, params: dict, user: str, job_id: Optional[str] = None) -> dict:
    """Queue an import. kind: 'gnps2_task' (params: task_id, workflow) or 'upload'
    (params: paths of the saved files). params['extra'] is merged into info.json."""
    global _worker
    job = {
        "job_id": job_id or uuid.uuid4().hex[:12], "kind": kind, "dataset_id": dataset_id,
        "status": "queued", "submitted_by": user, "submitted_at": _now(),
        "started_at": None, "finished_at": None, "step": None, "log": [], "error": None, "dataset": None,
        "_params": params,
    }
    with _lock:
        if any(j["dataset_id"] == dataset_id and j["status"] in ("queued", "running") for j in _jobs.values()):
            raise RuntimeError(f"An import of '{dataset_id}' is already queued or running.")
        _jobs[job["job_id"]] = job
        finished = [j for j in _jobs.values() if j["status"] in ("done", "failed")]
        for old in sorted(finished, key=lambda j: j["finished_at"])[:-KEEP_FINISHED or None]:
            _jobs.pop(old["job_id"], None)
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_work, name="import-worker", daemon=True)
            _worker.start()
    _queue.put(job["job_id"])
    return public(job)


def public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    out["queue_position"] = None
    if job["status"] == "queued":
        with _lock:
            queued = [j for j in _jobs.values() if j["status"] == "queued"]
        queued.sort(key=lambda j: j["submitted_at"])
        out["queue_position"] = next((i + 1 for i, j in enumerate(queued) if j["job_id"] == job["job_id"]), None)
    return out


def get(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
    return public(job) if job else None


def all_jobs() -> list[dict]:
    with _lock:
        jobs = list(_jobs.values())
    return [public(j) for j in sorted(jobs, key=lambda j: j["submitted_at"], reverse=True)]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _child(kind: str, dataset_id: str, params: dict, q) -> None:
    """Runs in the child process."""
    try:
        import importer
        progress = lambda step: q.put(("progress", step))
        if kind == "gnps2_task":
            info = importer.import_task(dataset_id, params["task_id"], params["workflow"],
                                        progress=progress, extra=params.get("extra"))
        else:
            paths = {k: Path(v) for k, v in params["files"].items() if v}
            info = importer.import_files(dataset_id, paths["feature_table"], paths.get("annotations"),
                                         paths.get("metadata"), progress=progress, extra=params.get("extra"))
        q.put(("done", info))
    except Exception as exc:
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def _set(job: dict, **fields) -> None:
    with _lock:
        job.update(fields)


def _work() -> None:
    ctx = mp.get_context("spawn")
    while True:
        job_id = _queue.get()
        with _lock:
            job = _jobs.get(job_id)
        if job is None:
            continue
        _set(job, status="running", started_at=_now())
        t0 = time.time()
        q = ctx.Queue()
        proc = ctx.Process(target=_child, args=(job["kind"], job["dataset_id"], job["_params"], q),
                           name=f"import-{job['dataset_id']}", daemon=True)
        result: Optional[tuple] = None
        try:
            proc.start()
            while result is None:
                try:
                    kind, payload = q.get(timeout=1)
                except queue.Empty:
                    if not proc.is_alive():
                        break
                    continue
                if kind == "progress":
                    with _lock:
                        job["step"] = payload
                        job["log"].append(f"{time.time() - t0:6.0f}s  {payload}")
                else:
                    result = (kind, payload)
            proc.join(timeout=30)
        except Exception as exc:
            result = ("error", f"Could not run the import: {exc}")
        finally:
            folder = job["_params"].get("upload_dir")
            if folder:
                shutil.rmtree(folder, ignore_errors=True)
        if result is None:
            code = proc.exitcode
            hint = " (killed: most likely out of memory)" if code in (-9, 137) else ""
            result = ("error", f"Import process exited with code {code}{hint}.")
        # the dataset is already swapped in (or untouched) when the status flips, so a
        # client that polls for "done" can immediately start its next import
        if result[0] == "done":
            _set(job, status="done", finished_at=_now(), step="Done", dataset=result[1])
        else:
            _set(job, status="failed", finished_at=_now(), error=result[1])
