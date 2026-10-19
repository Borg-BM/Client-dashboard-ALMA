"""Login via OpenID Connect (Authentik).

Everything is denied by default: every route except PUBLIC_PATHS requires a valid session, so
a new route cannot be left open by accident. The session lives in a signed cookie; we only
store who is logged in there, never tokens.
"""

import logging
from urllib.parse import quote, urlsplit

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware

from alma.config import Settings

log = logging.getLogger("alma.auth")

# Exact paths, no prefix: this way a path like /auth/../api/x can never count as 'public'.
PUBLIC_PATHS = frozenset({"/healthz", "/auth/login", "/auth/callback", "/auth/logout", "/auth/logged-out"})

USER_KEY = "user"
NEXT_KEY = "next"
MAX_CLAIM_LENGTH = 200


def safe_next(value: str | None) -> str:
    """Return value if it is a relative path within this app, otherwise '/' (no open redirect)."""
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    if any(ord(c) < 32 or ord(c) == 127 for c in value):  # browsers strip tab/newline from URLs
        return "/"
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    return value


def _page(status: int, title: str, message: str, link: tuple[str, str] | None = ("/auth/login", "Log in again")) -> HTMLResponse:
    """Simple error page; contains fixed text only, never values from the request or from the IdP."""
    anchor = f'<p><a href="{link[0]}">{link[1]}</a></p>' if link else ""
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>{title}</title><h1>{title}</h1><p>{message}</p>{anchor}</html>"
    )
    return HTMLResponse(html, status_code=status)


def _clip(value: object) -> str | None:
    return str(value)[:MAX_CLAIM_LENGTH] if value else None


def install_auth(app: FastAPI, settings: Settings) -> None:
    if not settings.auth_enabled:
        log.warning("LOGIN IS DISABLED (ALMA_AUTH_ENABLED=false): anyone with network access can see the dashboard")

        @app.get("/api/me")
        def me_anonymous():
            return {"auth": False}

        return

    redirect_uri = f"{settings.public_url}/auth/callback"
    issuer = settings.oidc_issuer.rstrip("/")

    oauth = OAuth()
    oauth.register(
        name="authentik",
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret.get_secret_value(),
        server_metadata_url=f"{issuer}/.well-known/openid-configuration",
        client_kwargs={"scope": settings.oidc_scopes, "code_challenge_method": "S256", "timeout": 10.0},
    )
    oidc = oauth.authentik

    @app.middleware("http")
    async def require_login(request: Request, call_next) -> Response:
        path = request.url.path
        user = request.session.get(USER_KEY)
        if path in PUBLIC_PATHS or (isinstance(user, dict) and user.get("sub")):
            return await call_next(request)
        if path.startswith("/api/") or request.method not in ("GET", "HEAD"):
            return JSONResponse({"detail": "Not logged in"}, status_code=401)
        target = safe_next(path + (f"?{request.url.query}" if request.url.query else ""))
        return RedirectResponse(f"/auth/login?next={quote(target, safe='')}", status_code=302)

    # Must be added after the gate: the last middleware added runs first.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret.get_secret_value(),
        # __Host- makes the browser accept only a host-only, Secure cookie on '/', so a sibling
        # domain cannot force a session cookie on us. Requires https, so not used on local http.
        session_cookie="__Host-alma_session" if settings.public_is_https else "alma_session",
        max_age=settings.session_max_age_s,
        same_site="lax",  # 'lax' is required: the return from Authentik is a cross-site navigation
        https_only=settings.public_is_https,
    )

    @app.get("/auth/login")
    async def login(request: Request, next: str = "/"):
        next_url = safe_next(next)
        user = request.session.get(USER_KEY)
        if isinstance(user, dict) and user.get("sub"):
            return RedirectResponse(next_url, status_code=302)
        request.session[NEXT_KEY] = next_url
        try:
            return await oidc.authorize_redirect(request, redirect_uri)
        except Exception:
            # Deliberately broad: network errors, a broken discovery response or a wrong issuer URL
            # must all give a clean error page (and a traceback in the log), not a 500.
            log.exception("Starting login with Authentik failed")
            return _page(502, "Login unavailable", "Authentik cannot be reached right now. Please try again shortly.")

    @app.get("/auth/callback")
    async def callback(request: Request):
        next_url = safe_next(request.session.get(NEXT_KEY))
        try:
            token = await oidc.authorize_access_token(request)
        except OAuthError as exc:  # wrong state, error from Authentik or a rejected code
            log.warning("Login failed: %s", exc.error)
            return _page(400, "Login failed", "The login attempt has expired or is invalid. This also happens if you log in from several tabs at once.")
        except Exception:
            # Network error, or an id_token with an invalid signature/claims: never log in, but do log it.
            log.exception("Completing login failed")
            return _page(
                502,
                "Login failed",
                "The response from Authentik could not be verified, or Authentik cannot be reached. "
                "Please try again; if this keeps happening, contact the administrator.",
            )

        claims = token.get("userinfo")  # only populated if the id_token (signature, nonce, iss, aud, exp) is valid
        if not claims or not claims.get("sub"):
            log.warning("Login denied: no validated id_token in the response")
            return _page(400, "Login failed", "Authentik did not return a valid identity.")

        if settings.allowed_groups:
            groups = claims.get("groups")
            if groups is None:  # not in the id_token: try the userinfo endpoint
                try:
                    groups = (await oidc.userinfo(token=token)).get("groups")
                except Exception:
                    log.exception("Fetching groups via userinfo failed")
                    return _page(502, "Login unavailable", "Group membership could not be verified.")
            if isinstance(groups, str):
                groups = [groups]
            if not set(groups or []) & set(settings.allowed_groups):
                log.warning("Access denied (no allowed group): %s", _clip(claims.get("preferred_username") or claims["sub"]))
                return _page(403, "No access", "Your account does not have access to this dashboard. Ask the administrator for access.", link=None)

        user = {
            "sub": _clip(claims["sub"]),
            "name": _clip(claims.get("name") or claims.get("preferred_username") or claims.get("email") or claims["sub"]),
            "email": _clip(claims.get("email")),
            "username": _clip(claims.get("preferred_username")),
        }
        request.session.clear()  # fresh session after login: old state/next data is gone
        request.session[USER_KEY] = user
        log.info("Logged in: %s", user["username"] or user["sub"])
        return RedirectResponse(next_url, status_code=302)

    @app.post("/auth/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/auth/logged-out", status_code=303)

    @app.get("/auth/logged-out")
    async def logged_out():
        return _page(
            200,
            "Logged out",
            "You have been logged out of the dashboard. Your Authentik session stays active; log out there too if you are using a shared computer.",
        )

    @app.get("/api/me")
    def me(request: Request):
        return {"auth": True, **request.session[USER_KEY]}
