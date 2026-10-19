# ALMA BM010 klantdashboard

Webdashboard voor het ALMA-systeem (BM010) dat oscillaties in een windmolenpaal tijdens de bouw
tegengaat. Het toont de trilling van de toren met het systeem uit en aan, de demping, en een
schatting van de kabelspanning op basis van de cilinderdruk. De metingen komen live binnen als
UDP-telegrammen van iba (ibaPDA); de server bewaart de laatste minuten in het geheugen. Zolang er
geen bron is draait de app op gesimuleerde data (met een DEMO-banner). De ibaHD-server (ibaHD-API)
kan als bron voor historie dienen, maar levert niet in realtime.

Inloggen gaat via Authentik (OpenID Connect). De app draait in Docker.

## Lokaal draaien (zonder Docker, zonder login)

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"          # voeg ,ibahd toe voor de echte ibaHD-bron
$env:ALMA_AUTH_ENABLED = "false"               # alleen voor lokaal; zie 'Beveiliging'
.venv\Scripts\python -m uvicorn alma.main:app_factory --factory --port 8000
```

Open <http://localhost:8000>. Tests: `.venv\Scripts\python -m pytest`.

`.\demo.ps1` doet hetzelfde met gesimuleerde data. `.\demo-udp.ps1 -Sources <IP van ibaPDA>` start
het dashboard op de live UDP-bron met telegrammen van een echte iba (zie 'Live data via UDP').

## Draaien in Docker

```powershell
copy .env.example .env      # vul de Authentik-gegevens en ALMA_SESSION_SECRET in
docker compose up -d --build
docker compose logs -f alma
```

* De container luistert op `127.0.0.1:8000` van de host. Zet er een **reverse proxy met TLS** voor
  (de publieke URL moet `https` zijn) en publiceer de poort nooit rechtstreeks op internet.
* `ALMA_PUBLIC_URL` moet exact de URL zijn waarmee gebruikers het dashboard openen. Open de app
  alleen via die URL: de sessiecookie hoort bij die hostnaam.
* De app start niet als de logininstellingen ontbreken of onveilig zijn (te kort geheim, `http`
  buiten localhost, enz.); de foutmelding in `docker compose logs` zegt wat er mist.
* `GET /healthz` is publiek en wordt gebruikt voor de Docker-healthcheck.
* De container draait als niet-root, met read-only bestandssysteem en zonder extra capabilities.

## Authentik instellen

In de Authentik-adminomgeving (labels kunnen per Authentik-versie iets verschillen, zie de
[OAuth2-providerdocumentatie](https://docs.goauthentik.io/add-secure-apps/providers/oauth2/)):

1. Maak een **Application** met een **OAuth2/OpenID Provider** (slug bijvoorbeeld `alma`).
2. Provider-instellingen:
   * *Client type*: **confidential**. Kopieer *Client ID* en *Client Secret* naar `.env`.
   * *Redirect URIs*: voeg exact `https://<ALMA_PUBLIC_URL host>/auth/callback` toe. Laat dit veld
     niet leeg: dan onthoudt Authentik de eerste URL die erlangs komt.
   * *Signing key*: **kies een certificaat** (bijvoorbeeld het zelfondertekende certificaat van
     Authentik). Zonder signing key tekent Authentik met HS256 en publiceert het geen sleutel via
     JWKS; de app weigert zulke tokens (inloggen mislukt dan met "kon niet worden gecontroleerd").
   * *Scopes*: `openid`, `profile`, `email` (standaard). De groepen komen mee in de `profile`-scope.
3. **Wie mag erin?** Koppel op de Application een *policy binding* met de groep(en) die toegang
   mogen hebben. Zonder binding laat Authentik elke ingelogde gebruiker toe. Wil je dat de app dit
   ook zelf controleert, zet dan `ALMA_ALLOWED_GROUPS=groep1,groep2` (namen uit de `groups`-claim).
4. `ALMA_OIDC_ISSUER` is `https://<authentik-host>/application/o/<slug>/`. Staat de provider op
   *global issuer mode*, gebruik dan de issuer uit het discovery-document
   (`.../application/o/<slug>/.well-known/openid-configuration`).

## Configuratie

Alles via omgevingsvariabelen of `.env` (zie `.env.example`).

| Variabele | Betekenis |
| --- | --- |
| `ALMA_SOURCE` | `mock` (gesimuleerd, DEMO-banner), `udp` (live van iba) of `ibahd` |
| `ALMA_UDP_PORT` | UDP-poort voor de telegrammen van iba (standaard 5005); compose publiceert dezelfde poort |
| `ALMA_UDP_PUBLISH` | Alleen compose: op welk IP van de host de UDP-poort wordt gepubliceerd (standaard `127.0.0.1`; zet het tunnel-IP van de server) |
| `ALMA_UDP_ALLOWED_SOURCES` | Verplicht bij `udp`: IP's of netwerken (komma-gescheiden) waarvan telegrammen worden aangenomen |
| `ALMA_UDP_ALLOW_ANY_SOURCE` | `true` laat elk bronadres toe; alleen voor lokale ontwikkeling, niet samen met het filter |
| `ALMA_CACHE_MINUTES` | Hoeveel minuten metingen in het geheugen blijven (1-60, standaard 10); de periodekeuze volgt dit |
| `ALMA_UDP_FORMAT` | Pad naar het telegramformaat (standaard `udp_format.json`) |
| `ALMA_UDP_BIND`, `ALMA_UDP_MAX_SAMPLES` | Luisteradres (standaard `0.0.0.0`) en harde geheugengrens van de cache (standaard 200.000 metingen) |
| `ALMA_PUBLIC_URL` | Externe basis-URL, `https`, zonder pad |
| `ALMA_OIDC_ISSUER` / `_CLIENT_ID` / `_CLIENT_SECRET` | Authentik-provider |
| `ALMA_SESSION_SECRET` | Minstens 32 tekens; wijzigen logt iedereen uit |
| `ALMA_SESSION_MAX_AGE_S` | Levensduur van een sessie (standaard 8 uur, vanaf inloggen) |
| `ALMA_ALLOWED_GROUPS` | Optioneel, komma-gescheiden groepen |
| `ALMA_OIDC_SCOPES` | Standaard `openid profile email` |
| `ALMA_AUTH_ENABLED` | `false` zet inloggen uit; alleen voor lokale ontwikkeling |
| `ALMA_CYLINDER__BORE_DIAMETER_M` e.a. | Cilindergeometrie voor de kabelspanning (nu placeholders) |
| `ALMA_IBAHD_QUICK_CONNECT`, `ALMA_SIGNAL_MAP` | Alleen bij `ibahd`; zie hieronder |

## Live data via UDP (iba)

De ibaHD-server levert de data niet in realtime. Daarom stuurt iba (ibaPDA) de metingen live als
UDP-telegrammen naar dit dashboard:

```
ibaPDA (Generic UDP-uitvoermodule) --UDP--> ontvanger -> cache (laatste minuten) -> API -> browser
```

De server bewaart de laatste `ALMA_CACHE_MINUTES` minuten in het geheugen en vult daarmee de
grafieken; de periodekeuze in het dashboard volgt die cache. Na een herstart begint de cache leeg
en loopt hij weer vol. Draai de app met **één** uvicorn-worker: een tweede worker zou dezelfde
UDP-poort willen openen en de pakketten onderling verdelen.

### Het telegram (`udp_format.json`)

ibaPDA stuurt per cyclus (minimaal 50 ms, dus hooguit 20 per seconde) één telegram met alle
signalen op vaste plaatsen. Het bestand beschrijft die indeling met dezelfde begrippen als de
ibaPDA-module: offset, datatype (BYTE, WORD, DWORD, SINT, INT, DINT, LINT, FLOAT, DOUBLE) en voor
een digitaal signaal een bitnummer. De standaardindeling (big endian, 28 bytes):

| Offset | Type | Bit | Signaal |
| --- | --- | --- | --- |
| 0 | FLOAT | | `tower_x` (mm) |
| 4 | FLOAT | | `tower_y` (mm) |
| 8 | FLOAT | | `vessel_x` (mm) |
| 12 | FLOAT | | `vessel_y` (mm) |
| 16 | FLOAT | | `pressure_bore` (bar) |
| 20 | FLOAT | | `pressure_rod` (bar) |
| 24 | DWORD | 0 | `compensation_active` (0 of 1) |

Per signaal kunnen `scale`, `bias`, `min` en `max` worden opgegeven (waarde = ruw × scale + bias;
een telegram met een waarde buiten min/max wordt geweigerd). `header_bytes` slaat een vaste kop
over (offsets tellen dan vanaf het einde ervan) en `checks` eist constante waarden in elk telegram,
bijvoorbeeld een vast sleutelwoord. Een telegram met de verkeerde lengte, een verkeerde controlewaarde
of een waarde die geen getal is (NaN, oneindig) wordt geweigerd en geteld.

### ibaPDA instellen

Labels verschillen per ibaPDA-versie; zie de handleiding van *ibaPDA-Interface-Generic-UDP*
(uitvoermodules staan op het tabblad *Outputs*).

1. Voeg onder *Outputs* bij de interface *Generic UDP* een uitvoermodule toe.
2. Doeladres: het IP-adres van de server binnen het VPN (het tunnel-IP), doelpoort: `ALMA_UDP_PORT`
   (standaard 5005).
3. Updatetijd: 50 ms of hoger. 100 ms (10 Hz) is ruim genoeg voor een torenfrequentie rond 0,3 Hz.
4. Byte-volgorde: gelijk aan `byte_order` in `udp_format.json`. Staat deze verkeerd, dan komen
   er onzinwaarden binnen en staan digitale signalen op de verkeerde bit.
5. Zet de signalen op de offsets uit de tabel; `compensation_active` als digitaal signaal op adres
   24, bit 0. De telegramlengte moet `packet_bytes` zijn.

ibaPDA schrijft uitgeschakelde signalen als 0 in het telegram: een uitgeschakeld signaal is dus niet
te onderscheiden van een echte nul.

### De eerste verbinding met iba controleren

Er is bewust geen simulator: je test met de echte ibaPDA. Begin met `alma.udp_check`, dan het dashboard.

* `python -m alma.udp_check` luistert een paar seconden op de poort en laat zien wie er stuurt,
  hoe vaak en hoe regelmatig, of de telegrammen bij het formaat passen en welke waarden erin zitten.
  Met `--dump` toont het de ruwe bytes van het eerste telegram als tabel van float- en
  integerwaarden per offset, in beide byte-volgordes: daarmee vind je een verkeerde offset of
  byte-volgorde met het oog. De tool opent de poort zelf, dus stop het dashboard eerst of gebruik
  `--port`. In Docker: `docker compose stop alma` en daarna
  `docker compose run --rm --service-ports alma python -m alma.udp_check`.
* `GET /api/health` toont (voor ingelogde gebruikers) het aantal ontvangen telegrammen, de leeftijd
  van het laatste telegram en hoeveel seconden de cache bevat. De details (bronadressen, geweigerde
  telegrammen) staan in de logs en in `alma.udp_check`.

### Het schip: UDP over Starlink, in een VPN met de server

De ibaPDA staat op het schip en zit via Starlink in een VPN met de server. Het telegram gaat dus
door de tunnel:

```
ibaPDA (schip) --UDP--> VPN-tunnel over Starlink --> server (tunnelinterface, bv. 10.8.0.1:5005) --> ontvanger
```

* De tunnel versleutelt en authenticeert. Alleen wie bij het VPN hoort bereikt de poort, en het
  bronadres binnen de tunnel is niet te vervalsen. Daardoor is `ALMA_UDP_ALLOWED_SOURCES` hier een
  echte beveiliging en geen formaliteit.
* Zet in `ALMA_UDP_ALLOWED_SOURCES` het adres dat de server werkelijk ziet. Dat is het tunnel-IP van
  de ibaPDA-machine, of dat van de router van het schip als die het verkeer vertaalt (NAT) of de
  tunnel zelf opbouwt. Controleer het met `alma.udp_check`: dat toont het bronadres van binnenkomende
  telegrammen.
* **Publiceer de UDP-poort alleen op de tunnelinterface**: `ALMA_UDP_PUBLISH=<tunnel-IP van de server>`
  in `.env`. De standaard is alleen `127.0.0.1`: zonder deze instelling komt er dus niets binnen, maar
  er staat ook niets open op internet. Laat in de firewall van de server UDP naar deze poort alleen
  toe via de tunnelinterface. De tunnelinterface moet bestaan als de container start: Docker
  weigert een poort te publiceren op een IP-adres dat er nog niet is. Zet de VPN-service dus vóór
  Docker in de opstartvolgorde.
* ibaPDA stuurt naar het tunnel-IP van de server, niet naar het publieke adres.
* Het VPN moet het wisselende publieke adres van het schip aankunnen (Starlink gebruikt CGNAT en
  wisselt van adres bij een nieuwe verbinding). WireGuard doet dat vanzelf; zet daar
  `PersistentKeepalive = 25` aan de scheepskant. Bij OpenVPN of IPsec: keepalive en snel herverbinden.

Starlink beïnvloedt ook de data zelf:

* **Pakketten gaan verloren.** Starlink verliest regelmatig pakketten en valt kort uit (obstructie
  door kraan of toren, overdracht tussen satellieten). Het dashboard toont zo'n gat als gat en
  schat het verlies uit de aankomsttijden (`loss_percent` in `/api/health`, minimaal 20 telegrammen
  nodig). Een exacte meting met een teller in het telegram is er nog niet.
* De tijdstempel is de ontvangsttijd op de server. Dat is hier goed genoeg: Starlink-jitter van
  tientallen milliseconden is klein tegen een torenperiode van ongeveer 3,5 s, en de klok van het
  schip doet er niet toe.

Valt het VPN ooit weg als optie, dan staat de poort open op internet. Het bronfilter is dan niet
bruikbaar (wisselend adres), dus de app weigert `ALMA_UDP_ALLOW_ANY_SOURCE=true` op een niet-lokaal
adres, tenzij `udp_format.json` een `checks`-sleutelwoord bevat. Dat sleutelwoord houdt toevallig en
blind verkeer tegen, maar gaat onversleuteld over de lijn. Beter is dan een kleine doorstuurder aan
boord die het telegram met authenticatie (HTTPS) naar de server stuurt en na een uitval de gemiste
metingen nastuurt.

### Overige beveiliging van de UDP-poort

* `ALMA_UDP_ALLOWED_SOURCES` is verplicht; de app start niet zonder. Het filter vertrouwt het
  bronadres, en dat is alleen veilig binnen de tunnel of een eigen netwerk.
* **Docker Desktop (Windows/macOS) toont elke afzender als `172.17.0.1`** (gemeten, ook voor pakketten
  van de host zelf): daar kan het filter afzenders niet onderscheiden. Op een Linux-host blijft het
  bronadres van pakketten die via de tunnelinterface binnenkomen normaal behouden; controleer dat met
  `alma.udp_check` in de container. Draai productie dus op een Linux-host.
* Het geheugen is begrensd (`ALMA_UDP_MAX_SAMPLES`): een stortvloed aan pakketten verdringt hooguit
  oudere metingen. Geweigerde pakketten worden geteld en hooguit eens per minuut gelogd, zonder inhoud.
* `/api/health` toont aan ingelogde gebruikers alleen tellers, nooit adressen.

### Splitsen in frontend en backend

Met het VPN is de UDP-poort niet publiek, en dan is één proces het eenvoudigst: ontvanger, cache,
API en pagina draaien samen in één container, met één uvicorn-worker. Klanten bereiken alleen de
webpagina, via een reverse proxy met TLS en de Authentik-login; het VPN voert alleen de metingen.

De klantpagina is gewoon een webpagina: statische bestanden (`frontend/`) die alleen via `/api/...`
met de backend (`alma/`) praten. Dat is al gescheiden. Een aparte *deployment* van de pagina levert
alleen iets op als je hem elders wilt hosten (dan komen CORS en sessiecookies erbij).

De ontvanger en cache (`alma/ingest/`) hangen niet van FastAPI af. Dat maakt het later mogelijk de
ontvanger los te draaien, bijvoorbeeld om de pagina te kunnen herstarten zonder de cache te verliezen,
of om dezelfde decodeerlaag in een doorstuurder aan boord te gebruiken.

## ibaHD-server als bron (historie, niet realtime)

Vereist de licentie **ibaHD-Server-API-Read** op de ibaHD-server.

1. Exporteer in ibaHD Manager de quick-connect-gegevens naar `secrets/quick_connect.json`
   (Host, Port, ApiKey, TlsServerCertificate). De map `secrets/` staat in `.gitignore` en
   `.dockerignore`. Het certificaat is alleen geldig voor de namen die erin staan: gebruik die
   als `Host`.
2. **Controleer de verbinding:** `.venv\Scripts\python -m alma.ibahd_check` (in Docker:
   `docker compose run --rm alma python -m alma.ibahd_check`). Het loopt stap voor stap langs
   het bestand, het netwerk, TLS en ApiKey, de stores en kanalen, de signaalmap, de laatste
   waarden en een meetreeks, en zegt bij een fout wat er waarschijnlijk mis is. De ApiKey komt
   nooit in de uitvoer.
3. Zoek de kanalen op met `--list` of `--grep <tekst>` en vul in `signals.json` de ID's van de
   zeven signalen in (de waarde uit de eerste kolom, vorm `<store>\<kanaal>`). Draai de check
   opnieuw: die toetst elk ID tegen het schema en leest de data uit.
4. Zet `ALMA_SOURCE=ibahd` en herstart. Is de server onbereikbaar, dan geeft de API een 503 en
   toont het dashboard NO DATA; de reden staat (hooguit eens per minuut) in de logs.

Tijden in de ibaHD-API zijn microseconden sinds 1970. De tests (`tests/test_ibahd.py`) draaien
tegen een nagebootste server (`tests/fake_ibahd.py`: echte gRPC, TLS en API-key) en bewijzen dat
onze client consistent is met onze lezing van de iba-documentatie. Of een echte server zich zo
gedraagt, toont alleen `alma.ibahd_check`. Het dashboard leest *aggregated* data (gemiddelde per
bucket, maximaal 40.000 punten per venster); de check meldt de effectieve meetfrequentie en
waarschuwt onder 2 Hz, waar ruwe data (`GetRawChannelData`) beter zou zijn voor de frequentieanalyse.

## Beveiliging: hoe het werkt

* **Alles dicht, behalve `/healthz` en `/auth/*`.** Nieuwe routes zijn dus automatisch beschermd.
* **Authorization code flow met PKCE**, `state` en `nonce`. De id_token wordt gecontroleerd op
  handtekening (RS256 via JWKS), issuer, audience en geldigheid. Mislukt dat, dan wordt er niet
  ingelogd.
* **Sessiecookie**: ondertekend (niet versleuteld), `HttpOnly`, `Secure`, `SameSite=Lax` en op
  https met het `__Host-`-voorvoegsel. Hij bevat alleen naam, e-mail, gebruikersnaam en `sub`;
  er worden geen tokens bewaard. Een sessie is absoluut beperkt in tijd.
* **Uitloggen** beëindigt de sessie in de app, niet de Authentik-sessie (geen single logout).
  Op een gedeelde computer ook in Authentik uitloggen.
* Strikte `Content-Security-Policy` (alle scripts en stijlen komen van de app zelf; uPlot staat in
  `frontend/vendor/`), `Cache-Control: no-store`, HSTS bij https.

## Onderhoud

* Afhankelijkheden vernieuwen (vastgepind met hashes in `requirements.txt`):
  `uv pip compile pyproject.toml --extra ibahd --universal --python-version 3.13 --generate-hashes -o requirements.txt`
  en daarna `docker compose up -d --build`.
* **Licentie:** het pakket `ibahd-api` is deels proprietary en mag alleen met een geldige iba-licentie
  worden gebruikt. Publiceer het Docker-image dus niet in een openbaar register.
* Bekende beperking: twee inlogpogingen tegelijk in dezelfde browser (twee tabbladen) laten er één
  mislukken; opnieuw inloggen lost dat op.
