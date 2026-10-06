"""Live training output: stage headers with timing, tqdm bars for parallel fits, log tee."""
import contextlib
import sys
import time
from datetime import datetime

import joblib
from tqdm import tqdm

ENABLED = True  # set False by `train_all --no-progress`


def _ts():
    return datetime.now().strftime("%H:%M:%S")


@contextlib.contextmanager
def stage(title, index=None, total=None):
    """Print a header, run the block, print elapsed time."""
    prefix = f"[{index}/{total}] " if index is not None else ""
    print(f"\n{'━' * 72}\n{_ts()}  ▶ {prefix}{title}\n{'━' * 72}", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"{_ts()}  ✔ {title} — {time.perf_counter() - t0:.1f}s", flush=True)


def step(message):
    print(f"{_ts()}  · {message}", flush=True)


def fmt_params(params):
    """{'alpha': np.float64(1.0), 'degree': 2} -> 'alpha=1 degree=2'"""
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


@contextlib.contextmanager
def fit_progress(total, desc):
    """tqdm bar driven by the first joblib.Parallel run inside the block (e.g. GridSearchCV's CV fits).

    joblib calls Parallel.print_progress in the main process whenever tasks finish; nested Parallel
    runs in the main process (e.g. a RandomForest refit with n_jobs=-1) are ignored.
    """
    if not ENABLED:
        yield
        return
    bar = tqdm(total=total, desc=f"  {desc}", unit="fit", ncols=88, leave=True,
               bar_format="{desc:<28} {percentage:3.0f}%|{bar}| {n}/{total} fits [{elapsed}<{remaining}]")
    original = joblib.Parallel.print_progress
    owner = []

    def print_progress(self):
        if not owner:
            owner.append(self)
        if self is owner[0]:
            bar.n = min(self.n_completed_tasks, total)
            bar.refresh()
        return original(self)

    joblib.Parallel.print_progress = print_progress
    try:
        yield
        bar.n = total
        bar.refresh()
    finally:
        joblib.Parallel.print_progress = original
        bar.close()


class Tee:
    """Duplicate stdout into a log file (progress bars go to stderr and stay out of the log)."""

    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()
