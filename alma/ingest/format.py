"""Het formaat van een UDP-telegram van ibaPDA (Generic UDP-uitvoermodule): welke bytes welk signaal zijn.

Een telegram is een momentopname van alle signalen. De signalen staan op vaste offsets, de byte-
volgorde is instelbaar en een digitaal signaal is een bit in een woord. Het formaat staat in een
JSON-bestand, zodat een wijziging in ibaPDA geen codewijziging vraagt:

    {
      "byte_order": "big",
      "packet_bytes": 14,
      "header_bytes": 4,
      "fields": [
        {"signal": "compensation_active", "offset": 0, "type": "WORD", "bit": 0},
        {"signal": "pressure_bore", "offset": 2, "type": "FLOAT"},
        {"signal": "tower_x", "offset": 6, "type": "INT", "scale": 0.1, "min": -500, "max": 500}
      ],
      "checks": [{"offset": 8, "type": "WORD", "equals": 43981}],
      "timestamp": {"source": "receive"}
    }

Offsets tellen vanaf het einde van de header. Een bitveld is bit `bit` (0 = minst significant) van het
woord op `offset`, gelezen in de byte-volgorde van het formaat; zo adresseert ook ibaPDA digitale
signalen. Waarde = ruw * scale + bias, en min/max gelden na het schalen. `checks` zijn constante
woorden die in elk telegram moeten staan: een goedkope poort tegen losse of blind ingespoten pakketten.

Het formaat is bewust streng. Een onbekende sleutel, een byte-volgorde zonder keuze of overlappende
bytes is een fout bij het laden, want dubbelzinnige bytes geven stil verkeerde waarden. decode() weigert
op dezelfde manier alles wat niet exact past, en doet per telegram niets zwaarders dan een struct-aanroep
en wat rekenwerk: het draait voor elk datagram, ook van wie het netwerk bestookt.
"""

import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple

__all__ = ["MAX_PACKET_BYTES", "Check", "Field", "FormatError", "PacketError", "PacketFormat"]

MAX_PACKET_BYTES = 65507  # grootste UDP-payload over IPv4: 65535 - 8 (UDP-kop) - 20 (IP-kop)

_TOP_KEYS = ("byte_order", "packet_bytes", "header_bytes", "fields", "checks", "timestamp")
_FIELD_KEYS = ("signal", "offset", "type", "bit", "scale", "bias", "min", "max")
_CHECK_KEYS = ("offset", "type", "equals")
_NUMBER_KEYS = ("scale", "bias", "min", "max")  # horen bij een analoog signaal, niet bij een bitveld
_BYTE_ORDERS = ("big", "little")

# Bewust zonder getallen en zonder verwachte waarde: de melding belandt in logs en mag de sleutel van
# de controle niet verraden.
_CHECK_MISMATCH = "controlewoord klopt niet: geen telegram van de verwachte afzender of een verkeerd formaat"


class FormatError(ValueError):
    """Het formaat is ongeldig. De tekst noemt het signaal, de sleutel of de index; nooit de waarde van een check."""


class PacketError(ValueError):
    """Een datagram past niet bij het formaat. De tekst is bedoeld voor de logs en bevat geen pakketinhoud."""


@dataclass(frozen=True)
class _Type:
    name: str
    code: str  # struct-code
    size: int
    integer: bool
    signed: bool
    low: int | None  # grenzen van een geheel type; None bij floats
    high: int | None


def _integer(name: str, code: str, size: int, signed: bool) -> _Type:
    bits = 8 * size
    low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return _Type(name, code, size, True, signed, low, high)


_TYPES = {
    kind.name: kind
    for kind in (
        _integer("uint8", "B", 1, False),
        _integer("int8", "b", 1, True),
        _integer("uint16", "H", 2, False),
        _integer("int16", "h", 2, True),
        _integer("uint32", "I", 4, False),
        _integer("int32", "i", 4, True),
        _integer("int64", "q", 8, True),  # bewust geen uint64: de uitkomst is een float en die houdt 2**64 niet vast
        _Type("float32", "f", 4, False, True, None, None),
        _Type("float64", "d", 8, False, True, None, None),
    )
}
# Zoals ibaPDA de types noemt (IEC 61131-3).
_ALIASES = {
    "byte": "uint8",
    "sint": "int8",
    "word": "uint16",
    "int": "int16",
    "dword": "uint32",
    "dint": "int32",
    "lint": "int64",
    "float": "float32",
    "double": "float64",
}
_TYPE_HELP = ", ".join(f"{canonical}/{alias.upper()}" for alias, canonical in _ALIASES.items())
_FLOAT32 = struct.Struct("<f")  # alleen om overloop bij encode() te melden


def _find_type(name: str) -> _Type | None:
    key = name.lower()  # lower() en geen casefold(): casefold laat ook een lange s (U+017F) voor een s doorgaan
    return _TYPES.get(_ALIASES.get(key, key))


@dataclass(frozen=True)
class Field:
    """Een signaal in het telegram. `type` is de canonieke naam; bit, min en max zijn None als ze ontbreken."""

    signal: str
    offset: int
    type: str
    bit: int | None = None
    scale: float = 1.0
    bias: float = 0.0
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class Check:
    """Een constant woord dat in elk telegram moet staan."""

    offset: int
    type: str
    equals: int = field(repr=False)  # nooit in logs of tracebacks: het is de sleutel tegen losse pakketten


# ---- het formaat inlezen ------------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)  # een JSON-boolean is geen getal


def _finite(value: Any) -> float | None:
    """value als float, of None als het geen getal is (ook geen boolean) of niet eindig."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # een int boven het float-bereik
        return None
    return number if math.isfinite(number) else None


def _ascii(text: str) -> str:
    """Voor in een melding: alles buiten ASCII als escape, zodat een Windows-console er niet over struikelt."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _shorten(value: Any, limit: int = 40) -> str:
    """Voor in een melding: een sleutel of typenaam uit het bestand, ingekort en in ASCII."""
    text = ascii(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _reject_unknown_keys(entry: Mapping, allowed: tuple[str, ...], where: str) -> None:
    for key in entry:
        if key not in allowed:
            raise FormatError(f"{where}: onbekende sleutel {_shorten(key)} (toegestaan: {', '.join(allowed)})")


def _require(entry: Mapping, key: str, where: str) -> Any:
    if key not in entry:
        raise FormatError(f"{where}: {key!r} ontbreekt")
    return entry[key]


def _parse_offset(where: str, value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise FormatError(f"{where}: offset moet een geheel getal van 0 of meer zijn")
    return value


def _parse_type(where: str, value: Any) -> _Type:
    kind = _find_type(value) if isinstance(value, str) else None
    if kind is None:
        raise FormatError(f"{where}: onbekend type {_shorten(value)} (kies uit: {_TYPE_HELP})")
    return kind


def _parse_field(index: int, entry: Any) -> Field:
    where = f"fields[{index}]"
    if not isinstance(entry, Mapping):
        raise FormatError(f"{where} moet een JSON-object zijn, geen {type(entry).__name__}")
    name = entry.get("signal")
    if isinstance(name, str) and name:
        where += f" ({name!a})"
    _reject_unknown_keys(entry, _FIELD_KEYS, where)
    for key in ("signal", "offset", "type"):
        _require(entry, key, where)
    if not isinstance(name, str) or not name:
        raise FormatError(f"{where}: signal moet een niet-lege tekst zijn")
    offset = _parse_offset(where, entry["offset"])
    kind = _parse_type(where, entry["type"])

    bit = None
    if "bit" in entry:
        bit = entry["bit"]
        if not kind.integer:
            raise FormatError(f"{where}: bit hoort bij een geheel type, niet bij {kind.name}")
        if not _is_int(bit) or not 0 <= bit < 8 * kind.size:
            last = 8 * kind.size - 1
            raise FormatError(f"{where}: bit moet een geheel getal van 0 tot en met {last} zijn ({kind.name})")
        extras = [key for key in _NUMBER_KEYS if key in entry]
        if extras:
            raise FormatError(f"{where}: een bitveld mag geen {' of '.join(extras)} hebben")

    settings: dict[str, float] = {}
    for key in _NUMBER_KEYS:
        if key in entry:
            number = _finite(entry[key])
            if number is None:
                raise FormatError(f"{where}: {key} moet een eindig getal zijn")
            settings[key] = number
    if settings.get("scale") == 0:
        raise FormatError(f"{where}: scale mag niet 0 zijn")
    low, high = settings.get("min"), settings.get("max")
    if low is not None and high is not None and low > high:
        raise FormatError(f"{where}: min ({low:g}) is groter dan max ({high:g})")
    return Field(name, offset, kind.name, bit, settings.get("scale", 1.0), settings.get("bias", 0.0), low, high)


def _parse_check(index: int, entry: Any) -> Check:
    where = f"checks[{index}]"
    if not isinstance(entry, Mapping):
        raise FormatError(f"{where} moet een JSON-object zijn, geen {type(entry).__name__}")
    _reject_unknown_keys(entry, _CHECK_KEYS, where)
    for key in _CHECK_KEYS:
        _require(entry, key, where)
    offset = _parse_offset(where, entry["offset"])
    kind = _parse_type(where, entry["type"])
    if not kind.integer:
        raise FormatError(f"{where}: alleen gehele types kunnen een check zijn, niet {kind.name}")
    equals = entry["equals"]
    if not _is_int(equals) or not kind.low <= equals <= kind.high:
        # De waarde zelf blijft uit de melding: het is de sleutel van de controle.
        raise FormatError(f"{where}: equals moet een geheel getal zijn dat in {kind.name} past")
    return Check(offset, kind.name, equals)


def _parse_timestamp(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise FormatError('timestamp moet een JSON-object zijn, bijvoorbeeld {"source": "receive"}')
    _reject_unknown_keys(value, ("source",), "timestamp")
    if _require(value, "source", "timestamp") != "receive":
        raise FormatError('timestamp: source moet "receive" zijn (de enige ondersteunde bron)')


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json.loads houdt bij een dubbele sleutel stil de laatste; bij byte_order is dat levensgevaarlijk."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormatError(f"sleutel {_shorten(key)} komt twee keer voor in hetzelfde JSON-object")
        result[key] = value
    return result


# ---- de indeling controleren en tot een struct omzetten ---------------------------------------------


def _check_bounds(label: str, offset: int, kind: _Type, body: int, header: int) -> None:
    if offset + kind.size > body:
        hint = " (offsets tellen vanaf het einde van de header)" if header else ""
        raise FormatError(
            f"{label}: {kind.size} bytes ({kind.name}) op offset {offset} passen niet in de {body} bytes "
            f"na de header{hint}"
        )


def _wire_order(
    fields: tuple[Field, ...], checks: tuple[Check, ...], body: int, header: int
) -> list[tuple[int, str]]:
    """Controleer namen, grenzen en overlap; geef de gelezen woorden (offset, type) in volgorde van offset.

    Bitvelden met hetzelfde offset en type delen een woord. Elke andere overlap is een fout.
    """
    first_with_name: dict[str, int] = {}
    claims: list[tuple[int, int, str, tuple[int, str]]] = []  # (begin, einde, label, woord)
    bit_words: set[tuple[int, str]] = set()
    for index, f in enumerate(fields):
        label = f"fields[{index}] ({f.signal!a})"
        if f.signal in first_with_name:
            raise FormatError(f"{label}: signaal komt al voor in fields[{first_with_name[f.signal]}]")
        first_with_name[f.signal] = index
        kind = _TYPES[f.type]
        _check_bounds(label, f.offset, kind, body, header)
        word = (f.offset, f.type)
        if f.bit is not None:
            if word in bit_words:
                continue
            bit_words.add(word)
        claims.append((f.offset, f.offset + kind.size, label, word))
    for index, c in enumerate(checks):
        label = f"checks[{index}]"
        kind = _TYPES[c.type]
        _check_bounds(label, c.offset, kind, body, header)
        claims.append((c.offset, c.offset + kind.size, label, (c.offset, c.type)))

    claims.sort(key=lambda claim: claim[:2])
    for before, after in zip(claims, claims[1:]):
        if after[0] < before[1]:  # op begin gesorteerd: zijn de buren heel, dan overlapt niets
            raise FormatError(f"{after[2]} overlapt met {before[2]}")
    return [claim[3] for claim in claims]


def _struct_for(byte_order: str, header: int, body: int, order: list[tuple[int, str]]) -> struct.Struct:
    """Het hele telegram als een struct: header en leemtes als pad-bytes, de woorden op volgorde van offset."""
    parts = ["<" if byte_order == "little" else ">"]  # standaardgroottes, geen uitlijning
    if header:
        parts.append(f"{header}x")
    cursor = 0
    for offset, type_name in order:
        kind = _TYPES[type_name]
        if offset > cursor:
            parts.append(f"{offset - cursor}x")
        parts.append(kind.code)
        cursor = offset + kind.size
    if body > cursor:
        parts.append(f"{body - cursor}x")
    return struct.Struct("".join(parts))


class _Read(NamedTuple):
    """Hoe decode() een signaal uit het uitgepakte tuple haalt."""

    signal: str
    place: int  # plaats in het uitgepakte tuple
    mask: int  # bitmasker van een bitveld; 0 bij een analoog signaal
    scale: float
    bias: float
    low: float  # -inf zonder min: zonder grens geldt geen grens, en dat scheelt een vergelijking
    high: float


@dataclass(frozen=True)
class _Compiled:
    """Alles wat decode() en encode() per telegram nodig hebben, vooraf berekend."""

    layout: struct.Struct  # het hele telegram in een keer: header en leemtes als pad-bytes
    slots: dict[tuple[int, str], int]  # woord (offset, type) -> plaats in het uitgepakte tuple
    checks: tuple[tuple[int, int], ...]  # (plaats, verwachte waarde)
    reads: tuple[_Read, ...]
    template: tuple[int, ...]  # voor encode(): nullen, met de checkwoorden ingevuld


@dataclass(frozen=True)
class PacketFormat:
    """Hoe een telegram eruitziet en hoe het gelezen en geschreven wordt.

    Bouw het met from_dict() of load(): die controleren het formaat helemaal. Een fout formaat moet bij het
    laden stuklopen, niet pas als er stille onzin in de grafieken staat.
    """

    byte_order: str  # "big" of "little"
    packet_bytes: int  # exacte lengte van een datagram, inclusief de header
    header_bytes: int  # genegeerde kop; alle offsets tellen vanaf het einde ervan
    fields: tuple[Field, ...]
    checks: tuple[Check, ...] = ()
    signals: tuple[str, ...] = field(init=False, repr=False, compare=False)  # in de volgorde van fields
    _compiled: _Compiled = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.byte_order not in _BYTE_ORDERS:
            raise FormatError('byte_order moet "big" of "little" zijn')
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "checks", tuple(self.checks))
        if not self.fields:
            raise FormatError("fields mag niet leeg zijn")
        body = self.packet_bytes - self.header_bytes
        order = _wire_order(self.fields, self.checks, body, self.header_bytes)
        slots = {word: place for place, word in enumerate(order)}
        layout = _struct_for(self.byte_order, self.header_bytes, body, order)
        assert layout.size == self.packet_bytes

        check_words = {(c.offset, c.type): c.equals for c in self.checks}
        compiled = _Compiled(
            layout=layout,
            slots=slots,
            checks=tuple((slots[(c.offset, c.type)], c.equals) for c in self.checks),
            reads=tuple(
                _Read(
                    f.signal,
                    slots[(f.offset, f.type)],
                    0 if f.bit is None else 1 << f.bit,
                    f.scale,
                    f.bias,
                    -math.inf if f.min is None else f.min,
                    math.inf if f.max is None else f.max,
                )
                for f in self.fields
            ),
            template=tuple(check_words.get(word, 0) for word in order),
        )
        object.__setattr__(self, "signals", tuple(f.signal for f in self.fields))
        object.__setattr__(self, "_compiled", compiled)

    def __reduce__(self):
        # struct.Struct is niet te pickelen; de definitie is genoeg, de rest wordt opnieuw berekend
        return (type(self), (self.byte_order, self.packet_bytes, self.header_bytes, self.fields, self.checks))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, required_signals: Iterable[str] | None = None) -> "PacketFormat":
        """Bouw het formaat uit de JSON-structuur. Alles wat niet klopt is een FormatError met een Nederlandse
        melding die het signaal, de sleutel of de index noemt.

        required_signals: de signalen in "fields" moeten precies deze verzameling zijn.
        """
        if not isinstance(data, Mapping):
            raise FormatError(f"formaat moet een JSON-object zijn, geen {type(data).__name__}")
        _reject_unknown_keys(data, _TOP_KEYS, "formaat")

        if "byte_order" not in data:
            raise FormatError('formaat: byte_order ontbreekt; kies "big" of "little" (er is bewust geen standaard)')
        byte_order = data["byte_order"]
        if not isinstance(byte_order, str) or byte_order not in _BYTE_ORDERS:
            raise FormatError('formaat: byte_order moet "big" of "little" zijn')
        packet_bytes = _require(data, "packet_bytes", "formaat")
        if not _is_int(packet_bytes) or not 1 <= packet_bytes <= MAX_PACKET_BYTES:
            raise FormatError(f"formaat: packet_bytes moet een geheel getal van 1 tot en met {MAX_PACKET_BYTES} zijn")
        header_bytes = data.get("header_bytes", 0)
        if not _is_int(header_bytes) or header_bytes < 0:
            raise FormatError("formaat: header_bytes moet een geheel getal van 0 of meer zijn")
        if header_bytes >= packet_bytes:
            raise FormatError(
                f"formaat: header_bytes ({header_bytes}) laat geen ruimte over in packet_bytes ({packet_bytes})"
            )

        raw_fields = _require(data, "fields", "formaat")
        if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
            raise FormatError("formaat: fields moet een lijst met minstens een veld zijn")
        fields = tuple(_parse_field(index, entry) for index, entry in enumerate(raw_fields))
        raw_checks = data.get("checks", [])
        if not isinstance(raw_checks, (list, tuple)):
            raise FormatError("formaat: checks moet een lijst zijn")
        checks = tuple(_parse_check(index, entry) for index, entry in enumerate(raw_checks))
        if "timestamp" in data:
            _parse_timestamp(data["timestamp"])

        fmt = cls(byte_order, packet_bytes, header_bytes, fields, checks)
        if required_signals is not None:
            wanted, present = {str(s) for s in required_signals}, set(fmt.signals)
            if wanted != present:
                parts = []
                if wanted - present:
                    parts.append("ontbrekend: " + ", ".join(sorted(wanted - present)))
                if present - wanted:
                    parts.append("onbekend: " + ", ".join(sorted(present - wanted)))
                raise FormatError("de signalen in fields komen niet overeen met de verwachte; " + "; ".join(parts))
        return fmt

    @classmethod
    def load(cls, path: str | PathLike[str], *, required_signals: Iterable[str] | None = None) -> "PacketFormat":
        """Lees het formaat uit een JSON-bestand (UTF-8, met of zonder BOM).

        De meldingen noemen het pad en nooit de inhoud van het bestand: het kan per ongeluk het verkeerde bestand
        zijn, bijvoorbeeld een met een ApiKey erin. Bij onleesbare JSON staat er alleen de plek van de fout in; bij
        een fout formaat noemt de melding sleutels, signaal- en typenamen en indexen, en nooit een check-waarde.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8-sig")  # Kladblok zet er graag een BOM voor
        except FileNotFoundError:
            raise FormatError(f"Formaatbestand niet gevonden: {path}") from None
        except UnicodeDecodeError:
            raise FormatError(f"Formaatbestand {path} is geen UTF-8; sla het op als UTF-8") from None
        except (OSError, ValueError) as err:
            raise FormatError(f"Formaatbestand {path} is niet te lezen ({type(err).__name__})") from None
        try:
            data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except FormatError as err:  # van de hook; FormatError is een ValueError en moet dus eerst
            raise FormatError(f"Formaatbestand {path}: {err}") from None
        except json.JSONDecodeError as err:
            where = f"regel {err.lineno}, kolom {err.colno}"  # de plek, niet de tekst eromheen
            raise FormatError(f"Formaatbestand {path} is geen geldige JSON ({where})") from None
        except (ValueError, RecursionError) as err:  # bv. een getal van duizenden cijfers, of absurde nesting
            raise FormatError(f"Formaatbestand {path} is geen geldige JSON ({type(err).__name__})") from None
        try:
            return cls.from_dict(data, required_signals=required_signals)
        except FormatError as err:
            raise FormatError(f"Formaatbestand {path}: {err}") from None

    def decode(self, packet: bytes | bytearray | memoryview) -> dict[str, float]:
        """Lees alle signalen uit een datagram. Gooit PacketError en niets anders, wat er ook binnenkomt.

        De waarden staan in de volgorde van `fields`: een bitveld als 1.0 of 0.0, de rest als ruw * scale + bias.
        """
        plan = self._compiled
        try:
            words = plan.layout.unpack(packet)  # controleert ook de lengte
        except Exception:  # struct.error bij een verkeerde lengte, anders een vreemd object (geen bytes, vrijgegeven)
            raise PacketError(self._misfit(packet)) from None
        for place, expected in plan.checks:
            if words[place] != expected:
                raise PacketError(_CHECK_MISMATCH)
        values: dict[str, float] = {}
        for signal, place, mask, scale, bias, low, high in plan.reads:
            word = words[place]
            if mask:
                values[signal] = 1.0 if word & mask else 0.0  # ook goed voor het tekenbit van een getal met teken
                continue
            value = word * scale + bias
            if not math.isfinite(value):
                raise PacketError(f"{_ascii(signal)}: waarde is NaN of oneindig")
            if not low <= value <= high:  # de waarde zelf blijft uit de melding: die belandt in de logs
                raise PacketError(f"{_ascii(signal)}: waarde valt buiten het bereik ({_bounds_text(low, high)})")
            values[signal] = value
        return values

    def _misfit(self, packet: Any) -> str:
        try:
            actual = memoryview(packet).nbytes
        except ValueError:
            return "telegram is niet meer te lezen (vrijgegeven memoryview)"
        except Exception:  # meestal een TypeError: geen bytes-achtig object
            return f"telegram moet bytes zijn, kreeg {type(packet).__name__}"
        if actual != self.packet_bytes:
            return f"telegram is {actual} bytes, verwacht {self.packet_bytes} bytes"
        return "telegram is niet als aaneengesloten bytes te lezen"

    def encode(self, values: Mapping[str, float]) -> bytes:
        """Het omgekeerde van decode(), voor de simulator en de tests.

        De header, de leemtes en de bits die geen signaal zijn blijven nul; de checkwoorden worden ingevuld.
        Bitvelden in hetzelfde woord worden samengevoegd (bit gezet bij een waarde boven 0.5). min en max worden
        hier niet afgedwongen, zodat een test ook een telegram buiten het bereik kan maken.

        Gooit ValueError bij een ontbrekend of onbekend signaal, een niet-eindige waarde of een waarde die niet
        in het type past. decode(encode(v)) geeft v terug op de nauwkeurigheid van het type: exact bij float64
        zonder scale en bias, anders binnen een halve stap (scale/2) bij gehele types en float32-precisie.
        """
        plan = self._compiled
        unknown = sorted(_shorten(name) for name in values if name not in self.signals)
        missing = [ascii(name) for name in self.signals if name not in values]
        if unknown or missing:
            problems = []
            if missing:
                problems.append("signaal ontbreekt: " + ", ".join(missing))
            if unknown:
                problems.append("onbekend signaal: " + ", ".join(unknown))
            raise ValueError("; ".join(problems))

        raw: list[int | float] = list(plan.template)
        bit_words: dict[int, tuple[int, _Type]] = {}
        for f in self.fields:
            kind = _TYPES[f.type]
            place = plan.slots[(f.offset, f.type)]
            number = _as_finite_float(f.signal, values[f.signal])
            if f.bit is not None:
                word = bit_words[place][0] if place in bit_words else 0
                if number > 0.5:
                    word |= 1 << f.bit
                bit_words[place] = (word, kind)
                continue
            scaled = (number - f.bias) / f.scale
            if not math.isfinite(scaled):
                raise _does_not_fit(f, number)
            if kind.integer:
                raw[place] = round(scaled)
                if not kind.low <= raw[place] <= kind.high:
                    raise _does_not_fit(f, number)
            else:
                if kind.code == "f":
                    try:
                        _FLOAT32.pack(scaled)
                    except OverflowError:  # buiten het float32-bereik; struct meldt dat zonder signaalnaam
                        raise _does_not_fit(f, number) from None
                raw[place] = scaled
        for place, (word, kind) in bit_words.items():
            if kind.signed and word >= 1 << (8 * kind.size - 1):
                word -= 1 << (8 * kind.size)  # struct wil een getal met teken als de code dat zegt
            raw[place] = word
        return plan.layout.pack(*raw)


def _bounds_text(low: float, high: float) -> str:
    parts = []
    if low != -math.inf:
        parts.append(f"min {low:g}")
    if high != math.inf:
        parts.append(f"max {high:g}")
    return ", ".join(parts)


def _does_not_fit(f: Field, number: float) -> ValueError:
    return ValueError(f"{_ascii(f.signal)}: waarde {number:g} past niet in {f.type}")


def _as_finite_float(signal: str, value: Any) -> float:
    """Alles wat float() als getal leest (ook bool en numpy-scalars), behalve tekst: float("1.5") slaagt, maar
    dan heeft de aanroeper een fout gemaakt."""
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{_ascii(signal)}: waarde moet een getal zijn, geen {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{_ascii(signal)}: waarde moet een getal zijn, geen {type(value).__name__}") from None
    except OverflowError:  # een gigantische int
        raise ValueError(f"{_ascii(signal)}: waarde is te groot") from None
    if not math.isfinite(number):
        raise ValueError(f"{_ascii(signal)}: waarde is NaN of oneindig")
    return number
