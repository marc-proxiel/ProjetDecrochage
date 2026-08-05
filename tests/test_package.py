import decrochage


def test_version_is_a_non_empty_string():
    assert isinstance(decrochage.__version__, str)
    assert decrochage.__version__
