from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# Logische signalen die de rest van de code gebruikt, losgekoppeld van ibaHD-kanaal-ID's.
SIGNALS = ["accel_x", "accel_y", "pressure_bore", "pressure_rod", "system_active"]


class CylinderGeometry(BaseModel):
    """Cilindergeometrie voor de kabelspanningsberekening.

    LET OP: placeholderwaarden, vervangen door de echte BM010-gegevens.
    """

    bore_diameter_m: float = 0.200
    rod_diameter_m: float = 0.140
    reeving_falls: int = 4  # aantal kabelparten (takel-overbrenging)
    efficiency: float = 0.95  # mechanisch rendement takel


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALMA_", env_file=ROOT / ".env", extra="ignore")

    source: str = "mock"
    ibahd_quick_connect: Path = ROOT / "secrets" / "quick_connect.json"
    signal_map: Path = ROOT / "signals.json"
    cylinder: CylinderGeometry = CylinderGeometry()


def get_settings() -> Settings:
    return Settings()
