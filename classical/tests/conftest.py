import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    """Every test gets its own NPEXP_ROOT and leaves no provider or recording source behind."""
    from classical import features
    from common import splits
    monkeypatch.setenv("NPEXP_ROOT", str(tmp_path / "npexp"))
    monkeypatch.delenv("NPEXP_KERNEL_CACHE", raising=False)
    monkeypatch.delenv("NPEXP_GRAM_DEVICE", raising=False)
    monkeypatch.setenv("NPEXP_KERNEL_STORE_FS", "any")   # synthetic kernels may live on any filesystem
    yield
    splits.set_provider(None)
    splits._provider = None
    features.set_recording_source(None)
