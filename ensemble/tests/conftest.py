import pytest

from common import lock, splits, stats
from ensemble.tests import synthetic


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A fresh server root with the synthetic benchmark installed; returns its results dir."""
    monkeypatch.setenv("NPEXP_ROOT", str(tmp_path))
    monkeypatch.setenv("NPEXP_REPRO_ROOT", str(tmp_path / "no-reproduction"))
    monkeypatch.setenv("NPEXP_COMMIT", "0123456789abcdef")
    monkeypatch.setenv("NPEXP_KERNEL_STORE_FS", "any")      # synthetic kernels in a temp directory
    monkeypatch.delenv("NPEXP_TEST_PURPOSE", raising=False)
    monkeypatch.delenv("NPEXP_TEST_MEMBER", raising=False)
    synthetic.install()
    stats._label_cache.clear()
    lock._granted.clear()
    yield tmp_path / "results"
    splits.set_provider(None)
    stats._label_cache.clear()
    lock._granted.clear()
