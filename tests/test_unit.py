"""Unit tests for configuration and pure helpers.

Expanded in later phases with the baseline gate and PSI tests.
"""

from driftguard import __version__
from driftguard.config import AG_NEWS_LABELS, get_settings


def test_version_is_set():
    assert __version__


def test_settings_defaults_are_sane():
    s = get_settings()
    assert s.app_name == "driftguard"
    assert 0.0 < s.val_fraction < 1.0
    assert s.psi_threshold > 0
    assert s.random_seed == 42


def test_ag_news_label_order_is_fixed():
    # This order is a hard contract with the dataset and the served label ids.
    assert AG_NEWS_LABELS == ("World", "Sports", "Business", "Sci/Tech")
    assert len(AG_NEWS_LABELS) == 4
