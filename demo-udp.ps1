# Start the dashboard locally on the live UDP source, with telegrams from a real iba (ibaPDA).
# Usage:  .\demo-udp.ps1 -Sources 10.0.0.5        then  http://localhost:8000  (stop: Ctrl+C)
#   -Sources  IP address of the ibaPDA machine as this computer sees it (comma-separated is also allowed).
#             Check it first with:  .\.venv\Scripts\python.exe -m alma.udp_check
#   -Port     UDP port that ibaPDA sends to (default 5005).
# The cache fills up from zero: after a minute the charts show a minute of data.
# For local use only: the login is deliberately disabled here and the web page only listens on localhost.
# The UDP port listens on all addresses, but only accepts telegrams from -Sources. The first time, Windows
# asks to allow UDP traffic for python.exe in the firewall.
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
