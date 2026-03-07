"""
Pytest fixtures. Ensures data and champion exist so tests need no manual setup.
Single pytest run: generate data and champion once per session when needed.
"""

from __future__ import annotations

import pytest

from ml_drift_monitor.config import get_default_config


@pytest.fixture(scope="session")
def session_config():
    """Session-scoped config."""
    return get_default_config()


@pytest.fixture(scope="session")
def ensure_data_and_champion(session_config):
    """
    Ensure monthly data and initial champion exist on disk so pipeline tests can run.
    No manual setup: programmatic generation once per test session.
    """
    from ml_drift_monitor.models.baseline import ensure_initial_champion
    ensure_initial_champion(session_config)
    return session_config
