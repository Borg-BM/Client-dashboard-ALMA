# Start het dashboard lokaal met gesimuleerde data en zonder login, om het ontwerp te bekijken.
# Gebruik:  .\demo.ps1        daarna  http://localhost:8000  (stoppen: Ctrl+C)
# Alleen voor lokaal gebruik: de login staat hier bewust uit en de server luistert alleen op localhost.
$env:ALMA_AUTH_ENABLED = "false"
$env:ALMA_SOURCE = "mock"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn alma.main:app_factory --factory --host 127.0.0.1 --port 8000
