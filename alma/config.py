import ipaddress
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# Logical signals used by the rest of the code, decoupled from ibaHD channel IDs.
# Tower/vessel signals are MRU displacements in mm; pressures in bar; system_mode is the integer mode code of the
# PLC. Two skids pull on the tower (skid 1 in X, skid 2 in Y), each with pressure on the bore and rod side.
SKIDS = (1, 2)
SIGNALS = [
    "system_mode",
    *[f"pressure_{side}_skid{n}" for n in SKIDS for side in ("bore", "rod")],
    "tower_x",
    "tower_y",
    "vessel_x",
    "vessel_y",
]

# NOTE: placeholder codes, replace with the real mode numbers of the BM010 PLC (ALMA_SYSTEM_MODES).
DEFAULT_SYSTEM_MODES = {0: "System off", 1: "HPU on", 2: "Position control", 3: "CT low", 4: "Damping"}
# Modes in which the system applies force to the tower; these count as 'compensation on' (hatching, damping percentage).
DEFAULT_COMPENSATING_MODES = [3, 4]

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class CylinderGeometry(BaseModel):
    """Cylinder geometry for the cable tension calculation.

    NOTE: placeholder values, replace with the real BM010 data
    (in Docker e.g. via ALMA_CYLINDER__BORE_DIAMETER_M=0.25).
    """

    bore_diameter_m: float = 0.200
    rod_diameter_m: float = 0.140
    reeving_falls: int = 4  # number of cable falls (reeving ratio)
    efficiency: float = 0.95  # mechanical efficiency of the reeving


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALMA_",
        env_file=ROOT / ".env",
        env_nested_delimiter="__",
        extra="ignore",
        # Without this, pydantic prints the raw input on a validation error, secrets included.
        hide_input_in_errors=True,
    )

    source: str = "mock"  # mock, ibahd or udp
    ibahd_quick_connect: Path = ROOT / "secrets" / "quick_connect.json"
    signal_map: Path = ROOT / "signals.json"
    cylinder: CylinderGeometry = CylinderGeometry()  # both skids: same cylinder until other data is available

    # System mode: the PLC sends a mode number (signal system_mode). This maps each number to a name, and
    # defines which modes count as compensation on. As JSON in the environment: ALMA_SYSTEM_MODES='{"0": "System off", ...}'
    system_modes: dict[int, str] = DEFAULT_SYSTEM_MODES
    compensating_modes: Annotated[list[int], NoDecode] = DEFAULT_COMPENSATING_MODES  # e.g. 3,4

    # Live ingest (ALMA_SOURCE=udp): the ibaHD server does not deliver in real time, so iba (ibaPDA, Generic UDP
    # output module) sends the measurements as UDP telegrams to this port. The server keeps the last
    # cache_minutes minutes in memory to fill the dashboard.
    udp_bind: str = "0.0.0.0"  # in Docker this must stay 0.0.0.0; restrict the port via compose/firewall
    udp_port: int = Field(5005, ge=0, le=65535)  # 0 = any free port, for tests only
    udp_format: Path = ROOT / "udp_format.json"
    # UDP is not authenticated: without a source filter anyone who can reach the port can fabricate measurements.
    udp_allowed_sources: Annotated[list[str], NoDecode] = []  # IPs or networks, e.g. 10.0.0.5,10.0.1.0/24
    udp_allow_any_source: bool = False  # enable deliberately, meant for local development
    cache_minutes: int = Field(10, ge=1, le=60)
    udp_max_samples: int = Field(200_000, ge=1_000)  # hard memory limit, even under a flood of packets

    # Dashboard: scale of the joystick circles (outer ring) and when data counts as stale.
    # NOTE: placeholder values, adjust to what the MRUs on the BM010 actually reach.
    tower_fullscale_mm: float = 60.0
    vessel_fullscale_mm: float = 500.0
    stale_after_s: float = 30.0

    # Login via OpenID Connect (Authentik). ON by default; it can only be turned off
    # deliberately (ALMA_AUTH_ENABLED=false), meant for local development.
    auth_enabled: bool = True
    public_url: str = "http://localhost:8000"  # external base URL, e.g. https://alma.example.com
    oidc_issuer: str | None = None  # e.g. https://auth.example.com/application/o/alma/
    oidc_client_id: str | None = None
    oidc_client_secret: SecretStr | None = None
    oidc_scopes: str = "openid profile email"
    allowed_groups: Annotated[list[str], NoDecode] = []  # empty = everyone Authentik lets in
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
            raise ValueError("ALMA_SYSTEM_MODES must contain at least one mode with a name")
        unknown = sorted(set(self.compensating_modes) - set(self.system_modes))
        if unknown:
            raise ValueError(
                f"ALMA_COMPENSATING_MODES contains {unknown}, but those mode numbers are not in ALMA_SYSTEM_MODES"
            )
        return self

    @model_validator(mode="after")
    def _check_udp_settings(self) -> "Settings":
        for entry in self.udp_allowed_sources:
            try:
                ipaddress.ip_network(entry, strict=False)
            except ValueError:
                raise ValueError(
                    f"ALMA_UDP_ALLOWED_SOURCES contains {entry!r}, which is not an IP address or network (e.g. 10.0.0.5 or 10.0.1.0/24)"
                ) from None
        try:
            ipaddress.ip_address(self.udp_bind)
        except ValueError:
            raise ValueError("ALMA_UDP_BIND must be an IP address, e.g. 0.0.0.0") from None
        if self.udp_allow_any_source and self.udp_allowed_sources:
            raise ValueError("Choose either ALMA_UDP_ALLOWED_SOURCES or ALMA_UDP_ALLOW_ANY_SOURCE=true, not both")
        if self.source == "udp" and not (self.udp_allowed_sources or self.udp_allow_any_source):
            raise ValueError(
                "ALMA_SOURCE=udp requires ALMA_UDP_ALLOWED_SOURCES (the IP address of the ibaPDA machine). "
                "ALMA_UDP_ALLOW_ANY_SOURCE=true allows any source address and is for local development only"
            )
        return self

    @field_validator("public_url")
    @classmethod
    def _check_public_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("must be a base URL such as https://alma.example.com")
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ValueError("must not contain a path, query or fragment (the app runs at the root)")
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
                f"Login is enabled but these settings are missing: {', '.join(missing)}. "
                "Set ALMA_AUTH_ENABLED=false only for local development without login."
            )
        if len(self.session_secret.get_secret_value()) < 32:
            raise ValueError(
                "ALMA_SESSION_SECRET must be at least 32 characters; generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if "openid" not in self.oidc_scopes.split():
            raise ValueError("ALMA_OIDC_SCOPES must contain 'openid'")
        if not self.public_is_https and urlsplit(self.public_url).hostname not in LOCAL_HOSTS:
            raise ValueError(
                "ALMA_PUBLIC_URL must use https (only localhost may use http), "
                "otherwise the session cookies are not protected"
            )
        issuer = urlsplit(self.oidc_issuer)
        if issuer.scheme != "https" and issuer.hostname not in LOCAL_HOSTS:
            raise ValueError("ALMA_OIDC_ISSUER must be an https URL")
        return self

    @property
    def public_is_https(self) -> bool:
        return self.public_url.startswith("https://")


def get_settings() -> Settings:
    return Settings()
