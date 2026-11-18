"""Output locations shared by the release scripts (no Spark dependency).

JOGJAPULSE_OUT sets the root folder for everything the scripts write: output/ (released files),
output_k10/, od_work/ (internal unsuppressed counts) and the presence suppression statistics.
Default: sample_run/ in the repository root.

JOGJAPULSE_FIG sets where the figure scripts save figures and their statistics (default:
sample_run/figures/).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_ROOT = os.path.abspath(os.environ.get("JOGJAPULSE_OUT", os.path.join(ROOT, "sample_run")))
FIG_DIR = os.path.abspath(os.environ.get("JOGJAPULSE_FIG", os.path.join(OUT_ROOT, "figures")))


def out_path(*parts):
    """File path under JOGJAPULSE_OUT; parent folders are created."""
    path = os.path.join(OUT_ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def out_dir(*parts):
    """Folder under JOGJAPULSE_OUT, created if needed."""
    path = os.path.join(OUT_ROOT, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def fig_path(name):
    """Figure (or figure statistics) path under JOGJAPULSE_FIG (default sample_run/figures/)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)
