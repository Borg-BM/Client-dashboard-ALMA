# Start the dashboard locally with simulated data and without login, to review the design.
# Usage:  .\demo.ps1        then  http://localhost:8000  (stop: Ctrl+C)
# For local use only: the login is deliberately disabled here and the server only listens on localhost.
$env:ALMA_AUTH_ENABLED = "false"
$env:ALMA_SOURCE = "mock"
& "$PSScriptRoot\.venv\Scripts\python.exe" -m uvicorn alma.main:app_factory --factory --host 127.0.0.1 --port 8000
