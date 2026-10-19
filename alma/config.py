import ipaddress
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# Logische signalen die de rest van de code gebruikt, losgekoppeld van ibaHD-kanaal-ID's.
# Tower/vessel signals are MRU displacements in mm; pressures in bar; system_mode is the integer mode code of the
# PLC. Twee skids trekken aan de toren (skid 1 in X, skid 2 in Y), elk met druk aan bodem- en stangzijde.
SKIDS = (1, 2)
SIGNALS = [
    "system_mode",
    *[f"pressure_{side}_skid{n}" for n in SKIDS for side in ("bore", "rod")],
    "tower_x",
    "tower_y",
    "vessel_x",
    "vessel_y",
]

# LET OP: placeholdercodes, vervangen door de echte modenummers van de BM010-PLC (ALMA_SYSTEM_MODES).
DEFAULT_SYSTEM_MODES = {0: "System off", 1: "HPU on", 2: "Position control", 3: "CT low", 4: "Damping"}
# Modi waarin het systeem kracht op de toren zet; die tellen als 'compensatie aan' (arcering, dempingspercentage).
DEFAULT_COMPENSATING_MODES = [3, 4]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class CylinderGeometry(BaseModel):
    """Cilindergeometrie voor de kabelspanningsberekening.

    LET OP: placeholderwaarden, vervangen door de echte BM010-gegevens
    (in Docker via bv. ALMA_CYLINDER__BORE_DIAMETER_M=0.25).
    """

    bore_diameter_m: float = 0.200
    rod_diameter_m: float = 0.140
    reeving_falls: int = 4  # aantal kabelparten (takel-overbrenging)
    efficiency: float = 0.95  # mechanisch rendement takel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALMA_",
        env_file=ROOT / ".env",
        env_nested_delimiter="__",
        extra="ignore",
        # Zonder dit drukt pydantic bij een validatiefout de ruwe invoer af, inclusief geheimen.
        hide_input_in_errors=True,
    )

    source: str = "mock"  # mock, ibahd of udp
    ibahd_quick_connect: Path = ROOT / "secrets" / "quick_connect.json"
    signal_map: Path = ROOT / "signals.json"
    cylinder: CylinderGeometry = CylinderGeometry()  # beide skids: zelfde cilinder tot er andere gegevens zijn

    # Systeemmodus: de PLC stuurt een modenummer (signaal system_mode). Hier staat welk nummer welke naam is, en
    # welke modi als compensatie aan tellen. Als JSON in de omgeving: ALMA_SYSTEM_MODES='{"0": "System off", ...}'
    system_modes: dict[int, str] = DEFAULT_SYSTEM_MODES
    compensating_modes: Annotated[list[int], NoDecode] = DEFAULT_COMPENSATING_MODES  # bv. 3,4

    # Live ingest (ALMA_SOURCE=udp): de ibaHD-server levert niet in realtime, dus iba (ibaPDA, Generic UDP
    # output module) stuurt de metingen als UDP-telegrammen naar deze poort. De server bewaart de laatste
    # cache_minutes minuten in het geheugen om het dashboard te vullen.
    udp_bind: str = "0.0.0.0"  # in Docker moet dit 0.0.0.0 blijven; beperk de poort via compose/firewall
    udp_port: int = Field(5005, ge=0, le=65535)  # 0 = willekeurige vrije poort, alleen voor tests
    udp_format: Path = ROOT / "udp_format.json"
    # UDP is niet geauthenticeerd: zonder bronfilter kan iedereen die de poort bereikt metingen verzinnen.
    udp_allowed_sources: Annotated[list[str], NoDecode] = []  # IP's of netwerken, bv. 10.0.0.5,10.0.1.0/24
    udp_allow_any_source: bool = False  # bewust aanzetten, bedoeld voor lokale ontwikkeling
    cache_minutes: int = Field(10, ge=1, le=60)
    udp_max_samples: int = Field(200_000, ge=1_000)  # harde geheugengrens, ook bij een stortvloed aan pakketten

    # Dashboard: schaal van de joystick-cirkels (buitenste ring) en wanneer data als verouderd geldt.
    # LET OP: placeholderwaarden, aanpassen aan wat de MRU's op de BM010 echt halen.
    tower_fullscale_mm: float = 60.0
    vessel_fullscale_mm: float = 500.0
    stale_after_s: float = 30.0

    # Inloggen via OpenID Connect (Authentik). Staat standaard AAN; uitzetten kan alleen
    # bewust (ALMA_AUTH_ENABLED=false), bedoeld voor lokale ontwikkeling.
    auth_enabled: bool = True
    public_url: str = "http://localhost:8000"  # externe basis-URL, bv. https://alma.example.com
    oidc_issuer: str | None = None  # bv. https://auth.example.com/application/o/alma/
    oidc_client_id: str | None = None
    oidc_client_secret: SecretStr | None = None
    oidc_scopes: str = "openid profile email"
    allowed_groups: Annotated[list[str], NoDecode] = []  # leeg = iedereen die Authentik toelaat
    session_secret: SecretStr | None = None
    session_max_age_s: int = 8 * 3600

    @field_validator("allowed_groups", "udp_allowed_sources", "compensating_modes", mode="before")
    @classmethod
    def _split_list(cls, value):
        if isinstance(value, str):
            return [g.strip() for g in value.split(",") if g.strip()]
        return value

    @model_validator(mode="after")
    def _check_system_modes(self) -> "Settings":
        if not self.system_modes or any(not name.strip() for name in self.system_modes.values()):
            raise ValueError("ALMA_SYSTEM_MODES moet minstens een modus met een naam bevatten")
        unknown = sorted(set(self.compensating_modes) - set(self.system_modes))
        if unknown:
            raise ValueError(
                f"ALMA_COMPENSATING_MODES bevat {unknown}, maar die modenummers staan niet in ALMA_SYSTEM_MODES"
            )
        return self

    @model_validator(mode="after")
    def _check_udp_settings(self) -> "Settings":
        for entry in self.udp_allowed_sources:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError:
                raise ValueError(
                    f"ALMA_UDP_ALLOWED_SOURCES bevat {entry!r}, dat is geen IP-adres of netwerk (bv. 10.0.0.5 of 10.0.1.0/24)"
                ) from None
        try:
            ipaddress.ip_address(self.udp_bind)
        except ValueError:
            raise ValueError("ALMA_UDP_BIND moet een IP-adres zijn, bv. 0.0.0.0") from None
        if self.udp_allow_any_source and self.udp_allowed_sources:
            raise ValueError("Kies ALMA_UDP_ALLOWED_SOURCES of ALMA_UDP_ALLOW_ANY_SOURCE=true, niet allebei")
        if self.source == "udp" and not (self.udp_allowed_sources or self.udp_allow_any_source):
            raise ValueError(
                "ALMA_SOURCE=udp vereist ALMA_UDP_ALLOWED_SOURCES (het IP-adres van de ibaPDA-machine). "
                "ALMA_UDP_ALLOW_ANY_SOURCE=true laat elk bronadres toe en is alleen voor lokale ontwikkeling"
            )
        return self

    @field_validator("public_url")
    @classmethod
    def _check_public_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("moet een basis-URL zijn zoals https://alma.example.com")
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ValueError("mag geen pad, query of fragment bevatten (de app draait op de root)")
        return f"{parts.scheme}://{parts.netloc}"

    @model_validator(mode="after")
    def _check_auth_settings(self) -> "Settings":
        if not self.auth_enabled:
            return self

        required = {
            "ALMA_OIDC_ISSUER": self.oidc_issuer,
            "ALMA_OIDC_CLIENT_ID": self.oidc_client_id,
            "ALMA_OIDC_CLIENT_SECRET": self.oidc_client_secret and self.oidc_client_secret.get_secret_value(),
            "ALMA_SESSION_SECRET": self.session_secret and self.session_secret.get_secret_value(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                f"Inloggen staat aan maar deze instellingen ontbreken: {', '.join(missing)}. "
                "Zet ALMA_AUTH_ENABLED=false alleen voor lokale ontwikkeling zonder login."
            )
        if len(self.session_secret.get_secret_value()) < 32:
            raise ValueError(
                "ALMA_SESSION_SECRET moet minstens 32 tekens zijn; genereer er een met: "
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if "openid" not in self.oidc_scopes.split():
            raise ValueError("ALMA_OIDC_SCOPES moet 'openid' bevatten")
        if not self.public_is_https and urlsplit(self.public_url).hostname not in LOCAL_HOSTS:
            raise ValueError(
                "ALMA_PUBLIC_URL moet https gebruiken (alleen localhost mag http zijn), "
                "anders worden de sessiecookies niet beveiligd"
            )
        issuer = urlsplit(self.oidc_issuer)
        if issuer.scheme != "https" and issuer.hostname not in LOCAL_HOSTS:
            raise ValueError("ALMA_OIDC_ISSUER moet een https-URL zijn")
        return self

    @property
    def public_is_https(self) -> bool:
        return self.public_url.startswith("https://")


def get_settings() -> Settings:
    return Settings()
