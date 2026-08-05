import pytest
from fastapi import HTTPException, Request

from decrochage.api import security


def make_request(client_host: str) -> Request:
    scope = {"type": "http", "headers": [], "client": (client_host, 12345)}
    return Request(scope)


@pytest.fixture(autouse=True)
def reset_rate_limit_state():
    """Le compteur `_hits` est un etat global du module : on l'isole entre tests."""
    security._hits.clear()
    yield
    security._hits.clear()


def test_rate_limit_allows_requests_under_the_limit():
    request = make_request("10.0.0.1")
    for _ in range(5):
        security.rate_limit(request, limit=5, window=60)  # ne doit pas lever


def test_rate_limit_blocks_once_the_limit_is_reached():
    request = make_request("10.0.0.2")
    for _ in range(5):
        security.rate_limit(request, limit=5, window=60)
    with pytest.raises(HTTPException) as exc_info:
        security.rate_limit(request, limit=5, window=60)
    assert exc_info.value.status_code == 429


def test_rate_limit_is_isolated_per_ip():
    request_a = make_request("10.0.0.3")
    request_b = make_request("10.0.0.4")
    for _ in range(3):
        security.rate_limit(request_a, limit=3, window=60)
    security.rate_limit(request_b, limit=3, window=60)  # IP differente -> ne doit pas lever
