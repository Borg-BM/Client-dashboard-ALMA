import base64
import json
import logging
import time
from urllib.parse import parse_qs, urlsplit

import pytest
import respx
from fake_idp import CLIENT_ID, CLIENT_SECRET, ISSUER, PUBLIC, USER, FakeIdp
from fastapi.testclient import TestClient
from pydantic import ValidationError

from alma.auth import safe_next
from alma.config import Settings
from alma.main import create_app

COOKIE = "__Host-alma_session"
SESSION_SECRET = "s" * 48


@pytest.fixture
def idp():
    # assert_all_mocked (default): every outgoing request that is not mocked makes the test fail
    with respx.mock(assert_all_called=False) as router:
        yield FakeIdp(router)


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        source="mock",
        public_url=PUBLIC,
        oidc_issuer=ISSUER,
        oidc_client_id=CLIENT_ID,
        oidc_client_secret=CLIENT_SECRET,
        session_secret=SESSION_SECRET,
    )
    return Settings(**{**base, **overrides})


def make_client(**overrides) -> TestClient:
    return TestClient(create_app(make_settings(**overrides)), base_url=PUBLIC, follow_redirects=False)


@pytest.fixture
def client(idp):
    return make_client()


def start_login(client, next_="/"):
    r = client.get("/auth/login", params={"next": next_})
    assert r.status_code == 302, r.text
    return r.headers["location"]


def log_in(client, idp, user=USER, next_="/"):
    """Runs the whole flow: login -> IdP -> callback. Returns the callback response."""
    callback_url = idp.authorize(start_login(client, next_), user)
    return client.get(callback_url)


# --- the gate: everything closed except the public paths ----------------------------------------


def test_everything_requires_login(client):
    assert client.get("/healthz").status_code == 200  # public, for the Docker healthcheck

    page = client.get("/")
    assert page.status_code == 302
    assert page.headers["location"] == "/auth/login?next=%2F"

    deep = client.get("/vendor/uplot/uPlot.iife.min.js?x=1")
    assert deep.headers["location"] == "/auth/login?next=%2Fvendor%2Fuplot%2FuPlot.iife.min.js%3Fx%3D1"

    for path in ("/api/overview", "/api/timeseries", "/api/health", "/api/me", "/api/does-not-exist"):
        r = client.get(path)
        assert r.status_code == 401, path
        assert r.json() == {"detail": "Not logged in"}
    assert client.post("/api/overview").status_code == 401
    assert client.post("/").status_code == 401  # non-GET gets a 401, not a redirect


def test_path_tricks_do_not_bypass_gate(client):
    # Exact path comparison: a prefix like /auth/ does not grant access to anything else.
    for path in ("/auth/../api/overview", "/auth/login/../../api/overview", "/healthz/../api/overview", "//api/overview"):
        r = client.get(path)
        assert r.status_code in (302, 401, 404), (path, r.status_code)
        assert "samples" not in r.text


# --- the complete login flow --------------------------------------------------------------------


def test_full_login_flow_and_logout(client, idp):
    callback = log_in(client, idp)
    assert callback.status_code == 302
    assert callback.headers["location"] == "/"

    # PKCE: the verifier at the token endpoint matches the challenge from the authorize request
    assert idp.token_requests and idp.token_requests[-1]["code_verifier"]

    assert client.get("/").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/overview?minutes=1").status_code == 200
    me = client.get("/api/me").json()
    assert me == {
        "auth": True,
        "sub": "user-1",
        "name": "Jan Jansen",
        "email": "jan@example.com",
        "username": "jan",
    }

    out = client.post("/auth/logout")
    assert out.status_code == 303 and out.headers["location"] == "/auth/logged-out"
    assert client.get("/api/me").status_code == 401  # session is gone
    assert client.get("/auth/logged-out").status_code == 200  # public


def test_login_returns_to_requested_page(client, idp):
    callback = log_in(client, idp, next_="/?minutes=30")
    assert callback.headers["location"] == "/?minutes=30"


def test_login_when_already_logged_in_skips_idp(client, idp):
    log_in(client, idp)
    again = client.get("/auth/login", params={"next": "/x"})
    assert again.status_code == 302 and again.headers["location"] == "/x"


def test_session_cookie_attributes(client, idp):
    cookie = log_in(client, idp).headers["set-cookie"]
    lowered = cookie.lower()
    assert cookie.startswith(f"{COOKIE}=")
    assert "httponly" in lowered and "secure" in lowered and "samesite=lax" in lowered
    assert "path=/" in lowered and "domain" not in lowered  # requirements for the __Host- prefix
    assert "max-age=28800" in lowered


def test_session_cookie_name_is_plain_on_local_http(idp):
    settings = make_settings(public_url="http://localhost:8000")
    client = TestClient(create_app(settings), base_url="http://localhost:8000", follow_redirects=False)
    r = client.get("/auth/login")
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("alma_session=") and "secure" not in cookie.lower()


# --- attacks on the flow ------------------------------------------------------------------------


@pytest.mark.parametrize("tamper", ["nonce", "signature", "expired", "issuer", "audience", "hs256", "no_id_token"])
def test_bad_id_token_is_rejected(client, idp, tamper):
    idp.tamper.add(tamper)
    r = log_in(client, idp)
    assert r.status_code in (400, 502), (tamper, r.status_code)
    assert client.get("/api/me").status_code == 401  # not logged in


def test_state_mismatch_is_rejected(client, idp):
    callback_url = idp.authorize(start_login(client))
    parts = urlsplit(callback_url)
    forged = f"{parts.path}?code={parse_qs(parts.query)['code'][0]}&state=forged-state"
    r = client.get(forged)
    assert r.status_code == 400
    assert idp.token_requests == []  # no code was even exchanged
    assert client.get("/api/me").status_code == 401


def test_callback_without_login_attempt_is_rejected(client, idp):
    assert client.get("/auth/callback?code=abc&state=def").status_code == 400
    assert client.get("/auth/callback").status_code == 400
    assert client.get("/api/me").status_code == 401


def test_idp_error_is_not_reflected(client, idp):
    start_login(client)
    r = client.get("/auth/callback?error=access_denied&error_description=%3Cscript%3Ealert(1)%3C/script%3E")
    assert r.status_code == 400
    assert "<script>" not in r.text and "access_denied" not in r.text


def test_callback_cannot_be_replayed(client, idp):
    callback_url = idp.authorize(start_login(client))
    assert client.get(callback_url).status_code == 302
    client.post("/auth/logout")
    assert client.get(callback_url).status_code == 400  # state/code have been consumed
    assert client.get("/api/me").status_code == 401


def test_failed_callback_leaves_existing_session_logged_out(client, idp):
    idp.tamper.add("signature")
    log_in(client, idp)
    idp.tamper.clear()
    assert client.get("/api/me").status_code == 401


@pytest.mark.parametrize(
    "evil",
    ["https://evil.example/", "//evil.example", "///evil.example", "/\\evil.example", "/\t/evil.example", "/\n/evil.example", "javascript:alert(1)", "evil", ""],
)
def test_open_redirect_is_blocked(client, idp, evil):
    callback = log_in(client, idp, next_=evil)
    assert callback.status_code == 302
    assert callback.headers["location"] == "/"


@pytest.mark.parametrize(
    "value,expected",
    [("/", "/"), ("/?minutes=30", "/?minutes=30"), ("/a/b?c=d#e", "/a/b?c=d#e"), (None, "/"), ("http://x", "/"), ("/%2F/x", "/%2F/x")],
)
def test_safe_next(value, expected):
    assert safe_next(value) == expected


def test_tampered_or_forged_session_cookie_is_rejected(client, idp):
    log_in(client, idp)
    value = client.cookies[COOKIE]
    flipped = value[:-3] + ("AAA" if not value.endswith("AAA") else "BBB")

    forged_payload = base64.b64encode(json.dumps({"user": {"sub": "admin"}}).encode()).decode()
    for bad in (flipped, forged_payload, f"{forged_payload}.AAAAAA.BBBBBB", "x"):
        client.cookies.clear()
        r = client.get("/api/me", headers={"cookie": f"{COOKIE}={bad}"})
        assert r.status_code == 401, bad


def test_session_is_rejected_server_side_after_max_age(idp):
    client = make_client(session_max_age_s=1)
    log_in(client, idp)
    old_cookie = client.cookies[COOKIE]
    assert client.get("/api/me").status_code == 200
    time.sleep(2.2)  # the cookie timestamp has one-second resolution: wait > max_age + 1 s
    client.cookies.clear()  # so that only the server's signature check counts, not the cookie expiry
    r = client.get("/api/me", headers={"cookie": f"{COOKIE}={old_cookie}"})
    assert r.status_code == 401


def test_session_cookie_from_other_secret_is_rejected(idp):
    first = make_client()
    log_in(first, idp)
    stolen = first.cookies[COOKIE]
    second = make_client(session_secret="o" * 48)
    assert second.get("/api/me", headers={"cookie": f"{COOKIE}={stolen}"}).status_code == 401


# --- authorization by group ---------------------------------------------------------------------


def test_allowed_group_grants_access(idp):
    client = make_client(allowed_groups="alma-viewers, alma-admins")
    assert log_in(client, idp).status_code == 302
    assert client.get("/api/me").status_code == 200


def test_missing_group_is_denied_and_logged(idp, caplog):
    client = make_client(allowed_groups=["alma-admins"])
    with caplog.at_level(logging.WARNING, logger="alma.auth"):
        r = log_in(client, idp)
    assert r.status_code == 403
    assert client.get("/api/me").status_code == 401
    assert "Access denied" in caplog.text and "jan" in caplog.text


def test_groups_fall_back_to_userinfo_endpoint(idp):
    idp.groups_in_id_token = False
    client = make_client(allowed_groups="alma-viewers")
    assert log_in(client, idp).status_code == 302
    assert client.get("/api/me").status_code == 200

    idp.groups_in_id_token = False
    denied = make_client(allowed_groups="alma-admins")
    assert log_in(denied, idp).status_code == 403


def test_no_group_restriction_by_default(idp):
    client = make_client()
    user = {**USER, "groups": []}
    assert log_in(client, idp, user=user).status_code == 302


# --- availability -------------------------------------------------------------------------------


def test_idp_down_gives_friendly_error_not_a_crash():
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{ISSUER}.well-known/openid-configuration").respond(503)
        client = make_client()
        r = client.get("/auth/login")
        assert r.status_code == 502
        assert "Authentik" in r.text
        assert client.get("/healthz").status_code == 200  # the app itself stays healthy


# --- security headers ---------------------------------------------------------------------------


def test_security_headers_on_every_kind_of_response(client, idp):
    log_in(client, idp)
    for response in (client.get("/"), client.get("/api/me"), client.get("/healthz"), TestClient(create_app(make_settings()), base_url=PUBLIC).get("/api/me")):
        h = response.headers
        assert "default-src 'self'" in h["content-security-policy"]
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["cache-control"] == "no-store"
        assert h["strict-transport-security"].startswith("max-age=")


# --- configuration: the app refuses to start with an unsafe or incomplete setup -----------------


def test_settings_require_auth_values():
    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)
    for name in ("ALMA_OIDC_ISSUER", "ALMA_OIDC_CLIENT_ID", "ALMA_OIDC_CLIENT_SECRET", "ALMA_SESSION_SECRET"):
        assert name in str(exc.value)


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"session_secret": "too-short"}, "at least 32"),
        ({"oidc_scopes": "profile email"}, "openid"),
        ({"public_url": "http://alma.example.com"}, "https"),
        ({"oidc_issuer": "http://auth.example.com/application/o/alma/"}, "https"),
        ({"public_url": "https://alma.example.com/app"}, "path"),
        ({"public_url": "alma.example.com"}, "base URL"),
    ],
)
def test_settings_reject_unsafe_values(overrides, fragment):
    with pytest.raises(ValidationError) as exc:
        make_settings(**overrides)
    assert fragment in str(exc.value)


def test_validation_errors_do_not_leak_secrets():
    with pytest.raises(ValidationError) as exc:
        make_settings(public_url="http://alma.example.com", oidc_client_secret="very-secret-client-value")
    text = str(exc.value)  # this is what a failed start shows in `docker logs`
    assert "ALMA_PUBLIC_URL" in text
    assert "very-secret-client-value" not in text
    assert SESSION_SECRET not in text


def test_settings_accept_localhost_http_and_normalize_url():
    settings = make_settings(public_url="http://localhost:8000/", oidc_issuer="http://localhost:9000/application/o/alma/")
    assert settings.public_url == "http://localhost:8000"
    assert make_settings(public_url="https://alma.example.com/").public_url == "https://alma.example.com"


def test_auth_can_only_be_disabled_explicitly(caplog):
    with caplog.at_level(logging.WARNING, logger="alma.auth"):
        client = TestClient(create_app(Settings(_env_file=None, source="mock", auth_enabled=False)))
    assert "LOGIN IS DISABLED" in caplog.text
    assert client.get("/api/me").json() == {"auth": False}
    assert client.get("/api/overview").status_code == 200
