"""Een minimale nep-Authentik voor tests: discovery, JWKS, token- en userinfo-endpoint.

Draait in-process via respx (de uitgaande httpx-aanvragen van de app worden onderschept), dus de
app gebruikt zijn echte Authlib-code zonder netwerk. De nep-IdP controleert zelf PKCE, de
client-authenticatie en de redirect_uri, en kan bewust foute tokens uitgeven (self.tamper).
"""

import base64
import hashlib
import secrets
import time
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import respx
from joserfc import jwt
from joserfc.jwk import OctKey, RSAKey

PUBLIC = "https://alma.test"
IDP = "https://idp.test"
ISSUER = f"{IDP}/application/o/alma/"
CLIENT_ID = "alma-client"
CLIENT_SECRET = "client-secret-value"

USER = {
    "sub": "user-1",
    "name": "Jan Jansen",
    "email": "jan@example.com",
    "preferred_username": "jan",
    "groups": ["alma-viewers", "everyone"],
}


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


# Sleutels genereren is traag; één keer per testrun is genoeg.
SIGNING_KEY = RSAKey.generate_key(2048, parameters={"kid": "idp-key", "use": "sig"})
OTHER_KEY = RSAKey.generate_key(2048, parameters={"kid": "idp-key", "use": "sig"})  # zelfde kid, andere sleutel


class FakeIdp:
    def __init__(self, router: respx.MockRouter) -> None:
        self.key = SIGNING_KEY
        self.other_key = OTHER_KEY
        self.codes: dict[str, dict] = {}
        self.access_tokens: dict[str, dict] = {}
        self.token_requests: list[dict] = []
        self.groups_in_id_token = True
        # Mogelijke waarden: nonce, signature, expired, issuer, audience, hs256, no_id_token
        self.tamper: set[str] = set()

        router.get(f"{ISSUER}.well-known/openid-configuration").mock(
            return_value=httpx.Response(
                200,
                json={
                    "issuer": ISSUER,
                    "authorization_endpoint": f"{IDP}/application/o/authorize/",
                    "token_endpoint": f"{IDP}/application/o/token/",
                    "userinfo_endpoint": f"{IDP}/application/o/userinfo/",
                    "end_session_endpoint": f"{ISSUER}end-session/",
                    "jwks_uri": f"{ISSUER}jwks/",
                    "response_types_supported": ["code"],
                    "subject_types_supported": ["public"],
                    "id_token_signing_alg_values_supported": ["RS256"],
                    "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
                    "code_challenge_methods_supported": ["plain", "S256"],
                },
            )
        )
        router.get(f"{ISSUER}jwks/").mock(
            side_effect=lambda request: httpx.Response(200, json={"keys": [self.key.as_dict(private=False)]})
        )
        router.post(f"{IDP}/application/o/token/").mock(side_effect=self._token)
        router.get(f"{IDP}/application/o/userinfo/").mock(side_effect=self._userinfo)

    def authorize(self, location: str, user: dict = USER) -> str:
        """Wat de browser doet nadat de gebruiker bij de IdP heeft ingelogd; geeft de callback-URL."""
        url = urlsplit(location)
        assert f"{url.scheme}://{url.netloc}{url.path}" == f"{IDP}/application/o/authorize/", location
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        assert q["response_type"] == "code"
        assert q["client_id"] == CLIENT_ID
        assert q["redirect_uri"] == f"{PUBLIC}/auth/callback"
        assert q["code_challenge_method"] == "S256", "PKCE (S256) moet aan staan"
        assert "openid" in q["scope"].split()
        assert q["state"] and q["nonce"]
        code = secrets.token_urlsafe(16)
        self.codes[code] = {
            "nonce": q["nonce"],
            "challenge": q["code_challenge"],
            "redirect_uri": q["redirect_uri"],
            "user": user,
        }
        return f"{q['redirect_uri']}?{urlencode({'code': code, 'state': q['state']})}"

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.token_requests.append(form)
        expected = "Basic " + base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
        if request.headers.get("authorization") != expected:
            return httpx.Response(401, json={"error": "invalid_client"})
        grant = self.codes.pop(form.get("code", ""), None)  # een code werkt maar één keer
        if grant is None or form.get("grant_type") != "authorization_code" or form.get("redirect_uri") != grant["redirect_uri"]:
            return httpx.Response(400, json={"error": "invalid_grant"})
        if b64url(hashlib.sha256(form.get("code_verifier", "").encode()).digest()) != grant["challenge"]:
            return httpx.Response(400, json={"error": "invalid_grant", "error_description": "PKCE mismatch"})

        access_token = secrets.token_urlsafe(24)
        self.access_tokens[access_token] = grant["user"]
        body = {"access_token": access_token, "token_type": "Bearer", "expires_in": 300}
        if "no_id_token" not in self.tamper:
            body["id_token"] = self._id_token(grant, access_token)
        return httpx.Response(200, json=body)

    def _id_token(self, grant: dict, access_token: str) -> str:
        now = int(time.time())
        user = grant["user"]
        claims = {
            "iss": "https://evil.test/" if "issuer" in self.tamper else ISSUER,
            "aud": "other-client" if "audience" in self.tamper else CLIENT_ID,
            "iat": now,
            "exp": now - 3600 if "expired" in self.tamper else now + 300,
            "nonce": "wrong-nonce" if "nonce" in self.tamper else grant["nonce"],
            "at_hash": b64url(hashlib.sha256(access_token.encode()).digest()[:16]),
            **{k: v for k, v in user.items() if k != "groups" or self.groups_in_id_token},
        }
        if "hs256" in self.tamper:  # wat Authentik doet zonder gekozen signing key
            return jwt.encode({"alg": "HS256"}, claims, OctKey.import_key(CLIENT_SECRET))
        key = self.other_key if "signature" in self.tamper else self.key
        return jwt.encode({"alg": "RS256", "kid": "idp-key"}, claims, key)

    def _userinfo(self, request: httpx.Request) -> httpx.Response:
        token = request.headers.get("authorization", "").removeprefix("Bearer ")
        user = self.access_tokens.get(token)
        if user is None:
            return httpx.Response(401, json={"error": "invalid_token"})
        return httpx.Response(200, json=user)
