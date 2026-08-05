from decrochage.config import settings


def test_settings_defaults_are_sane():
    assert settings.target_col == "abandon"
    assert 0.0 <= settings.decision_threshold <= 1.0
    assert isinstance(settings.random_seed, int)
