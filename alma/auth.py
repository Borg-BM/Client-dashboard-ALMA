"""Inloggen via OpenID Connect (Authentik).

Alles is standaard geweigerd: elke route behalve PUBLIC_PATHS vereist een geldige sessie, dus
een nieuwe route kan niet per ongeluk open staan. De sessie zit in een ondertekende cookie;
we bewaren daarin alleen wie er is ingelogd, geen tokens.
"""

import logging
from urllib.parse import quote, urlsplit

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.middleware.sessions import SessionMiddleware

from alma.config import Settings

log = logging.getLogger("alma.auth")

# Exacte paden, geen prefix: zo kan een pad als /auth/../api/x nooit als 'publiek' tellen.
PUBLIC_PATHS = frozenset({"/healthz", "/auth/login", "/auth/callback", "/auth/logout", "/auth/logged-out"})

USER_KEY = "user"
NEXT_KEY = "next"
MAX_CLAIM_LENGTH = 200


def safe_next(value: str | None) -> str:
    """Geef value terug als het een relatief pad binnen deze app is, anders '/' (geen open redirect)."""
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    if any(ord(c) < 32 or ord(c) == 127 for c in value):  # browsers strippen tab/newline uit URL's
        return "/"
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    return value


def _page(status: int, title: str, message: str, link: tuple[str, str] | None = ("/auth/login", "Opnieuw inloggen")) -> HTMLResponse:
    """Eenvoudige foutpagina; bevat alleen vaste tekst, nooit waarden uit het verzoek of van de IdP."""
    anchor = f'<p><a href="{link[0]}">{link[1]}</a></p>' if link else ""
    html = (
        '<!doctype html><html lang="nl"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<meta name="color-scheme" content="light dark">'
        f"<title>{title}</title><h1>{title}</h1><p>{message}</p>{anchor}</html>"
    )
    return HTMLResponse(html, status_code=status)


def _clip(value: object) -> str | None:
    return str(value)[:MAX_CLAIM_LENGTH] if value else None


def install_auth(app: FastAPI, settings: Settings) -> None:
    if not settings.auth_enabled:
        log.warning("INLOGGEN STAAT UIT (ALMA_AUTH_ENABLED=false): iedereen met netwerktoegang ziet het dashboard")

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
            return JSONResponse({"detail": "Niet ingelogd"}, status_code=401)
        target = safe_next(path + (f"?{request.url.query}" if request.url.query else ""))
        return RedirectResponse(f"/auth/login?next={quote(target, safe='')}", status_code=302)

    # Moet na de poort worden toegevoegd: de laatst toegevoegde middleware draait als eerste.
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret.get_secret_value(),
        # __Host- laat de browser alleen een host-only, Secure cookie op '/' accepteren, zodat een
        # zusterdomein geen sessiecookie kan opdringen. Vereist https, dus niet bij lokaal http.
        session_cookie="__Host-alma_session" if settings.public_is_https else "alma_session",
        max_age=settings.session_max_age_s,
        same_site="lax",  # 'lax' is nodig: de terugkeer van Authentik is een cross-site navigatie
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
            # Bewust breed: netwerkfouten, een kapotte discovery-respons of een verkeerde issuer-URL
            # moeten allemaal een nette melding geven (en een traceback in de log), geen 500.
            log.exception("Starten van inloggen bij Authentik mislukt")
            return _page(502, "Inloggen niet beschikbaar", "Authentik is nu niet bereikbaar. Probeer het straks opnieuw.")

    @app.get("/auth/callback")
    async def callback(request: Request):
        next_url = safe_next(request.session.get(NEXT_KEY))
        try:
            token = await oidc.authorize_access_token(request)
        except OAuthError as exc:  # onjuiste state, fout van Authentik of een geweigerde code
            log.warning("Inloggen mislukt: %s", exc.error)
            return _page(400, "Inloggen mislukt", "De inlogpoging is verlopen of ongeldig. Dit gebeurt ook als u in meerdere tabbladen tegelijk inlogt.")
        except Exception:
            # Netwerkfout, of een id_token met ongeldige handtekening/claims: nooit inloggen, wel loggen.
            log.exception("Afronden van inloggen mislukt")
            return _page(
                502,
                "Inloggen mislukt",
                "Het antwoord van Authentik kon niet worden gecontroleerd, of Authentik is niet bereikbaar. "
                "Probeer het opnieuw; blijft dit gebeuren, neem dan contact op met de beheerder.",
            )

        claims = token.get("userinfo")  # alleen gevuld als de id_token (handtekening, nonce, iss, aud, exp) klopt
        if not claims or not claims.get("sub"):
            log.warning("Inloggen geweigerd: geen gevalideerde id_token in het antwoord")
            return _page(400, "Inloggen mislukt", "Authentik gaf geen geldige identiteit terug.")

        if settings.allowed_groups:
            groups = claims.get("groups")
            if groups is None:  # niet in de id_token: probeer het userinfo-endpoint
                try:
                    groups = (await oidc.userinfo(token=token)).get("groups")
                except Exception:
                    log.exception("Groepen ophalen via userinfo mislukt")
                    return _page(502, "Inloggen niet beschikbaar", "Groepslidmaatschap kon niet worden gecontroleerd.")
            if isinstance(groups, str):
                groups = [groups]
            if not set(groups or []) & set(settings.allowed_groups):
                log.warning("Toegang geweigerd (geen toegestane groep): %s", _clip(claims.get("preferred_username") or claims["sub"]))
                return _page(403, "Geen toegang", "Uw account heeft geen toegang tot dit dashboard. Vraag de beheerder om toegang.", link=None)

        user = {
            "sub": _clip(claims["sub"]),
            "name": _clip(claims.get("name") or claims.get("preferred_username") or claims.get("email") or claims["sub"]),
            "email": _clip(claims.get("email")),
            "username": _clip(claims.get("preferred_username")),
        }
        request.session.clear()  # nieuwe sessie na inloggen: oude state/next-data weg
        request.session[USER_KEY] = user
        log.info("Ingelogd: %s", user["username"] or user["sub"])
        return RedirectResponse(next_url, status_code=302)

    @app.post("/auth/logout")
    async def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/auth/logged-out", status_code=303)

    @app.get("/auth/logged-out")
    async def logged_out():
        return _page(
            200,
            "U bent uitgelogd",
            "U bent uitgelogd bij het dashboard. Uw Authentik-sessie blijft actief; log daar ook uit als u een gedeelde computer gebruikt.",
        )

    @app.get("/api/me")
    def me(request: Request):
        return {"auth": True, **request.session[USER_KEY]}
