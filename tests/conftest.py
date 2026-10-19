from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def run_from_repo_root(monkeypatch):
    # The scripts read sql/views.sql and config/monitor.json by relative path
    monkeypatch.chdir(ROOT)
