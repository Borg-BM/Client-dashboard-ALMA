# Start het dashboard lokaal op de live UDP-bron, met telegrammen van een echte iba (ibaPDA).
# Gebruik:  .\demo-udp.ps1 -Sources 10.0.0.5        daarna  http://localhost:8000  (stoppen: Ctrl+C)
#   -Sources  IP-adres van de ibaPDA-machine zoals deze computer het ziet (komma-gescheiden mag ook).
#             Controleer het eerst met:  .\.venv\Scripts\python.exe -m alma.udp_check
#   -Port     UDP-poort waar ibaPDA naartoe stuurt (standaard 5005).
# De cache vult zich vanaf nul: na een minuut staat er een minuut in de grafieken.
# Alleen voor lokaal gebruik: de login staat hier bewust uit en de webpagina luistert alleen op localhost.
# De UDP-poort luistert op alle adressen, maar neemt alleen telegrammen van -Sources aan. Windows vraagt
# bij de eerste keer om UDP-verkeer voor python.exe toe te staan in de firewall.
param(
    [Parameter(Mandatory = $true)][string]$Sources,
    [int]$Port = 5005
)
$env:ALMA_AUTH_ENABLED = "false"
$env:ALMA_SOURCE = "udp"
$env:ALMA_UDP_ALLOWED_SOURCES = $Sources
$env:ALMA_UDP_BIND = "0.0.0.0"
$env:ALMA_UDP_PORT = $Port

Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn alma.main:app_factory --factory --host 127.0.0.1 --port 8000
} finally {
    Pop-Location
}
