# ALMA BM010 client dashboard

Web dashboard for the ALMA system (BM010), which counteracts oscillations in a wind turbine tower
during construction. It shows the tower vibration with the system off and on, the damping, and an
estimate of the cable tension based on the cylinder pressure. The measurements arrive live as UDP
telegrams from iba (ibaPDA); the server keeps the last few minutes in memory. As long as there is no
source, the app runs on simulated data (with a DEMO banner). The ibaHD server (ibaHD-API) can serve as
a source for history, but does not deliver in real time.

Login goes through Authentik (OpenID Connect). The app runs in Docker.

## Running locally (without Docker, without login)

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"          # add ,ibahd for the real ibaHD source
$env:ALMA_AUTH_ENABLED = "false"               # local use only; see 'Security: how it works'
.venv\Scripts\python -m uvicorn alma.main:app_factory --factory --port 8000
```

Open <http://localhost:8000>. Tests: `.venv\Scripts\python -m pytest`.

`.\demo.ps1` does the same with simulated data. `.\demo-udp.ps1 -Sources <ibaPDA IP>` starts the
dashboard on the live UDP source with telegrams from a real iba (see 'Live data via UDP').

## Running in Docker

```powershell
copy .env.example .env      # fill in the Authentik details and ALMA_SESSION_SECRET
docker compose up -d --build
docker compose logs -f alma
```

* The container listens on `127.0.0.1:8000` of the host. Put a **reverse proxy with TLS** in front of it
  (the public URL must be `https`) and never publish the port directly on the internet.
* `ALMA_PUBLIC_URL` must be exactly the URL users open the dashboard with. Only open the app through
  that URL: the session cookie belongs to that hostname.
* The app does not start if the login settings are missing or insecure (secret too short, `http`
  outside localhost, etc.); the error message in `docker compose logs` says what is missing.
* `GET /healthz` is public and is used for the Docker healthcheck.
* The container runs as non-root, with a read-only filesystem and without extra capabilities.

## Setting up Authentik

In the Authentik admin interface (labels can differ slightly per Authentik version, see the
[OAuth2 provider documentation](https://docs.goauthentik.io/add-secure-apps/providers/oauth2/)):

1. Create an **Application** with an **OAuth2/OpenID Provider** (slug for example `alma`).
2. Provider settings:
   * *Client type*: **confidential**. Copy *Client ID* and *Client Secret* to `.env`.
   * *Redirect URIs*: add exactly `https://<ALMA_PUBLIC_URL host>/auth/callback`. Do not leave this
     field empty: Authentik would then remember the first URL that comes by.
   * *Signing key*: **choose a certificate** (for example Authentik's self-signed certificate).
     Without a signing key Authentik signs with HS256 and does not publish a key through JWKS; the
     app rejects such tokens (login then fails with "could not be verified").
   * *Scopes*: `openid`, `profile`, `email` (default). The groups come along in the `profile` scope.
3. **Who gets in?** Bind a *policy* to the Application for the group(s) that may have access.
   Without a binding Authentik admits every logged-in user. If you also want the app to check this
   itself, set `ALMA_ALLOWED_GROUPS=group1,group2` (names from the `groups` claim).
4. `ALMA_OIDC_ISSUER` is `https://<authentik-host>/application/o/<slug>/`. If the provider uses
   *global issuer mode*, use the issuer from the discovery document
   (`.../application/o/<slug>/.well-known/openid-configuration`).

## Configuration

Everything through environment variables or `.env` (see `.env.example`).

| Variable | Meaning |
| --- | --- |
| `ALMA_SOURCE` | `mock` (simulated, DEMO banner), `udp` (live from iba) or `ibahd` |
| `ALMA_UDP_PORT` | UDP port for the telegrams from iba (default 5005); compose publishes the same port |
| `ALMA_UDP_PUBLISH` | Compose only: on which host IP the UDP port is published (default `127.0.0.1`; set the server's tunnel IP) |
| `ALMA_UDP_ALLOWED_SOURCES` | Required with `udp`: IPs or networks (comma-separated) from which telegrams are accepted |
| `ALMA_UDP_ALLOW_ANY_SOURCE` | `true` allows any source address; for local development only, cannot be combined with the filter |
| `ALMA_CACHE_MINUTES` | How many minutes of measurements stay in memory (1-60, default 10); the period selector follows this |
| `ALMA_SYSTEM_MODES` | Which PLC mode number has which name, as JSON, e.g. `{"0": "System off", "1": "HPU on", ...}`. The defaults (0-4: System off, HPU on, Position control, CT low, Damping) are **placeholders** |
| `ALMA_COMPENSATING_MODES` | Mode numbers (comma-separated) that count as 'compensation on': green shading and damping percentage (default `3,4` = CT low and damping) |
| `ALMA_UDP_FORMAT` | Path to the telegram format (default `udp_format.json`) |
| `ALMA_UDP_BIND`, `ALMA_UDP_MAX_SAMPLES` | Listen address (default `0.0.0.0`) and hard memory limit of the cache (default 200,000 samples) |
| `ALMA_PUBLIC_URL` | External base URL, `https`, without a path |
| `ALMA_OIDC_ISSUER` / `_CLIENT_ID` / `_CLIENT_SECRET` | Authentik provider |
| `ALMA_SESSION_SECRET` | At least 32 characters; changing it logs everyone out |
| `ALMA_SESSION_MAX_AGE_S` | Lifetime of a session (default 8 hours, from login) |
| `ALMA_ALLOWED_GROUPS` | Optional, comma-separated groups |
| `ALMA_OIDC_SCOPES` | Default `openid profile email` |
| `ALMA_AUTH_ENABLED` | `false` turns login off; for local development only |
| `ALMA_CYLINDER__BORE_DIAMETER_M` etc. | Cylinder geometry for the cable tension (placeholders for now) |
| `ALMA_IBAHD_QUICK_CONNECT`, `ALMA_SIGNAL_MAP` | Only with `ibahd`; see below |

## Live data via UDP (iba)

The ibaHD server does not deliver data in real time. That is why iba (ibaPDA) sends the measurements
live as UDP telegrams to this dashboard:

```
ibaPDA (Generic UDP output module) --UDP--> receiver -> cache (last minutes) -> API -> browser
```

The server keeps the last `ALMA_CACHE_MINUTES` minutes in memory and fills the charts with them; the
period selector in the dashboard follows that cache. After a restart the cache starts empty and fills
up again. Run the app with **one** uvicorn worker: a second worker would try to open the same UDP
port and the packets would be divided between them.

### The telegram (`udp_format.json`)

Each cycle (at least 50 ms, so at most 20 per second) ibaPDA sends one telegram with all signals at
fixed positions. The file describes that layout with the same terms as the ibaPDA module: offset,
data type (BYTE, WORD, DWORD, SINT, INT, DINT, LINT, FLOAT, DOUBLE) and, for a digital signal, a bit
number. The default layout (big endian, 36 bytes):

| Offset | Type | Bit | Signal |
| --- | --- | --- | --- |
| 0 | FLOAT | | `tower_x` (mm) |
| 4 | FLOAT | | `tower_y` (mm) |
| 8 | FLOAT | | `vessel_x` (mm) |
| 12 | FLOAT | | `vessel_y` (mm) |
| 16 | FLOAT | | `pressure_bore_skid1` (bar) |
| 20 | FLOAT | | `pressure_rod_skid1` (bar) |
| 24 | DWORD | | `system_mode` (mode number, 0-255) |
| 28 | FLOAT | | `pressure_bore_skid2` (bar) |
| 32 | FLOAT | | `pressure_rod_skid2` (bar) |

Two skids pull on the tower: skid 1 in X, skid 2 in Y. The dashboard calculates the cable tension per
skid (cylinder pressure, currently with the same placeholder geometry for both) and shows both in the
same chart. The tensions of the two skids are not added up: they pull in different directions.

Per signal, `scale`, `bias`, `min` and `max` can be given (value = raw × scale + bias; a telegram with
a value outside min/max is rejected). `header_bytes` skips a fixed header (offsets then count from
the end of it) and `checks` requires constant values in every telegram, for example a fixed key word.
A telegram with the wrong length, a wrong check value or a value that is not a number (NaN, infinity)
is rejected and counted.

### System mode

The PLC sends the mode as a single number (`system_mode`). The status bar shows its name. Which number
is which mode is set in `ALMA_SYSTEM_MODES`; the default numbers are placeholders and must match the
PLC. A number that is not in the table is shown as `MODE <number>`. Which modes count as compensation
on (green shading, damping percentage in `/api/overview`) is set in `ALMA_COMPENSATING_MODES`: by
default CT low and damping.

### Period and overlay

The period selector starts at 30 seconds (default 1 minute): a tower oscillation lasts about 6 s, so a
minute shows about ten oscillations. The API (`?minutes=`) accepts 0.25 to 60 minutes. The top chart
overlays the tension of both skids (left, kN) on the tower motion in X and Y (right, mm, dashed). This
shows whether the tension is applied on time and fast enough relative to the motion.

### Setting up ibaPDA

Labels differ per ibaPDA version; see the manual of *ibaPDA-Interface-Generic-UDP* (output modules
are on the *Outputs* tab).

1. Under *Outputs*, add an output module to the *Generic UDP* interface.
2. Destination address: the IP address of the server inside the VPN (the tunnel IP), destination
   port: `ALMA_UDP_PORT` (default 5005).
3. Update time: 50 ms or higher. 100 ms (10 Hz) is plenty for a tower oscillation of ~6 s (0.17 Hz).
4. Byte order: the same as `byte_order` in `udp_format.json`. If it is wrong, nonsense values arrive
   and digital signals end up on the wrong bit.
5. Put the signals at the offsets from the table; `system_mode` as an integer (DWORD) at address 24.
   The telegram length must be `packet_bytes`.

ibaPDA writes disabled signals as 0 in the telegram: a disabled signal cannot be told apart from a
real zero.

### Checking the first connection with iba

There is deliberately no simulator: you test with the real ibaPDA. Start with `alma.udp_check`, then
the dashboard.

* `python -m alma.udp_check` listens on the port for a few seconds and shows who is sending, how often
  and how regularly, whether the telegrams fit the format and which values they contain. With `--dump`
  it shows the raw bytes of the first telegram as a table of float and integer values per offset, in
  both byte orders: that lets you spot a wrong offset or byte order by eye. The tool opens the port
  itself, so stop the dashboard first or use `--port`. In Docker: `docker compose stop alma` and then
  `docker compose run --rm --service-ports alma python -m alma.udp_check`.
* `GET /api/health` shows (for logged-in users) the number of telegrams received, the age of the last
  telegram and how many seconds the cache holds. The details (source addresses, rejected telegrams)
  are in the logs and in `alma.udp_check`.

### The vessel: UDP over Starlink, in a VPN with the server

The ibaPDA is on the vessel and is connected to the server through Starlink in a VPN. The telegram
therefore goes through the tunnel:

```
ibaPDA (vessel) --UDP--> VPN tunnel over Starlink --> server (tunnel interface, e.g. 10.8.0.1:5005) --> receiver
```

* The tunnel encrypts and authenticates. Only members of the VPN can reach the port, and the source
  address inside the tunnel cannot be forged. That makes `ALMA_UDP_ALLOWED_SOURCES` real protection
  here and not a formality.
* Put the address the server actually sees in `ALMA_UDP_ALLOWED_SOURCES`. That is the tunnel IP of the
  ibaPDA machine, or that of the vessel's router if it translates the traffic (NAT) or builds the
  tunnel itself. Check it with `alma.udp_check`: it shows the source address of incoming telegrams.
* **Publish the UDP port only on the tunnel interface**: `ALMA_UDP_PUBLISH=<tunnel IP of the server>`
  in `.env`. The default is `127.0.0.1` only: without this setting nothing arrives, but nothing is
  open on the internet either. In the server's firewall, only allow UDP to this port via the tunnel
  interface. The tunnel interface must exist when the container starts: Docker refuses to publish a
  port on an IP address that does not exist yet. So start the VPN service before Docker.
* ibaPDA sends to the tunnel IP of the server, not to the public address.
* The VPN must cope with the vessel's changing public address (Starlink uses CGNAT and changes address
  on a new connection). WireGuard does this by itself; set `PersistentKeepalive = 25` on the vessel
  side. With OpenVPN or IPsec: keepalive and fast reconnection.

Starlink also affects the data itself:

* **Packets get lost.** Starlink regularly loses packets and drops out briefly (obstruction by crane or
  tower, handover between satellites). The dashboard shows such a gap as a gap and estimates the loss
  from the arrival times (`loss_percent` in `/api/health`, at least 20 telegrams needed). An exact
  measurement with a counter in the telegram does not exist yet.
* The timestamp is the receive time on the server. That is good enough here: Starlink jitter of tens
  of milliseconds is small compared to a tower period of about 6 s, and the vessel's clock does not
  matter.

If the VPN ever drops out as an option, the port is open on the internet. The source filter is then
unusable (changing address), so the app refuses `ALMA_UDP_ALLOW_ANY_SOURCE=true` on a non-local
address unless `udp_format.json` contains a `checks` key word. That key word keeps out accidental and
blind traffic, but travels unencrypted over the line. A better option then is a small forwarder on
board that sends the telegram to the server with authentication (HTTPS) and re-sends the missed
measurements after an outage.

### Other security of the UDP port

* `ALMA_UDP_ALLOWED_SOURCES` is required; the app does not start without it. The filter trusts the
  source address, which is only safe inside the tunnel or on a private network.
* **Docker Desktop (Windows/macOS) shows every sender as `172.17.0.1`** (measured, also for packets
  from the host itself): the filter cannot tell senders apart there. On a Linux host the source address
  of packets arriving through the tunnel interface is normally preserved; check this with
  `alma.udp_check` in the container. So run production on a Linux host.
* Memory is bounded (`ALMA_UDP_MAX_SAMPLES`): a flood of packets displaces older measurements at most.
  Rejected packets are counted and logged at most once a minute, without content.
* `/api/health` shows logged-in users only counters, never addresses.

### Splitting into frontend and backend

With the VPN the UDP port is not public, and then a single process is simplest: receiver, cache, API
and page run together in one container, with one uvicorn worker. Customers only reach the web page,
through a reverse proxy with TLS and the Authentik login; the VPN carries only the measurements.

The customer page is just a web page: static files (`frontend/`) that only talk to the backend
(`alma/`) through `/api/...`. That is already separated. A separate *deployment* of the page only
pays off if you want to host it elsewhere (then CORS and session cookies come into play).

The receiver and cache (`alma/ingest/`) do not depend on FastAPI. That makes it possible later to run
the receiver separately, for example to be able to restart the page without losing the cache, or to
use the same decoding layer in a forwarder on board.

## ibaHD server as a source (history, not real time)

Requires the **ibaHD-Server-API-Read** licence on the ibaHD server.

1. In ibaHD Manager, export the quick-connect details to `secrets/quick_connect.json`
   (Host, Port, ApiKey, TlsServerCertificate). The `secrets/` folder is in `.gitignore` and
   `.dockerignore`. The certificate is only valid for the names it contains: use one of those as
   `Host`.
2. **Check the connection:** `.venv\Scripts\python -m alma.ibahd_check` (in Docker:
   `docker compose run --rm alma python -m alma.ibahd_check`). It goes step by step through the file,
   the network, TLS and ApiKey, the stores and channels, the signal map, the latest values and a
   measurement series, and says what is probably wrong in case of an error. The ApiKey never appears
   in the output.
3. Look up the channels with `--list` or `--grep <text>` and fill in the IDs of the nine signals in
   `signals.json` (the value from the first column, of the form `<store>\<channel>`). Run the check
   again: it tests each ID against the schema and reads the data.
4. Set `ALMA_SOURCE=ibahd` and restart. If the server is unreachable, the API returns a 503 and the
   dashboard shows NO DATA; the reason is in the logs (at most once a minute).

Times in the ibaHD API are microseconds since 1970. The tests (`tests/test_ibahd.py`) run against a
simulated server (`tests/fake_ibahd.py`: real gRPC, TLS and API key) and prove that our client is
consistent with our reading of the iba documentation. Whether a real server behaves that way is only
shown by `alma.ibahd_check`. The dashboard reads *aggregated* data (average per bucket, at most 40,000
points per window); the check reports the effective sampling rate and warns below 2 Hz, where raw
data (`GetRawChannelData`) would be better for the frequency analysis.

## Security: how it works

* **Everything closed, except `/healthz` and `/auth/*`.** New routes are therefore protected
  automatically.
* **Authorization code flow with PKCE**, `state` and `nonce`. The id_token is checked for signature
  (RS256 via JWKS), issuer, audience and validity. If that fails, nobody is logged in.
* **Session cookie**: signed (not encrypted), `HttpOnly`, `Secure`, `SameSite=Lax` and, on https,
  with the `__Host-` prefix. It contains only name, email, username and `sub`; no tokens are stored. A
  session is limited in absolute time.
* **Logging out** ends the session in the app, not the Authentik session (no single logout). On a
  shared computer also log out in Authentik.
* Strict `Content-Security-Policy` (all scripts and styles come from the app itself; uPlot is in
  `frontend/vendor/`), `Cache-Control: no-store`, HSTS on https.

## Maintenance

* Refreshing dependencies (pinned with hashes in `requirements.txt`):
  `uv pip compile pyproject.toml --extra ibahd --universal --python-version 3.13 --generate-hashes -o requirements.txt`
  and then `docker compose up -d --build`.
* **Licence:** the `ibahd-api` package is partly proprietary and may only be used with a valid iba
  licence. So do not publish the Docker image in a public registry.
* Known limitation: two login attempts at the same time in the same browser (two tabs) make one of
  them fail; logging in again fixes that.
