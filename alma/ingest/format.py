"""The format of a UDP telegram from ibaPDA (Generic UDP output module): which bytes make up which signal.

A telegram is a snapshot of all signals. The signals sit at fixed offsets, the byte order is configurable
and a digital signal is a bit in a word. The format lives in a JSON file, so a change in ibaPDA does not
require a code change:

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

Offsets count from the end of the header. A bit field is bit `bit` (0 = least significant) of the word
at `offset`, read in the byte order of the format; this is also how ibaPDA addresses digital signals.
Value = raw * scale + bias, and min/max apply after scaling. `checks` are constant words that must be
present in every telegram: a cheap gate against stray or blindly injected packets.

The format is deliberately strict. An unknown key, a missing byte order or overlapping bytes is an error
at load time, because ambiguous bytes silently produce wrong values. decode() rejects everything that
does not fit exactly in the same way, and does nothing heavier per telegram than a struct call and a bit
of arithmetic: it runs for every datagram, including those from whoever is flooding the network.
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

MAX_PACKET_BYTES = 65507  # largest UDP payload over IPv4: 65535 - 8 (UDP header) - 20 (IP header)

_TOP_KEYS = ("byte_order", "packet_bytes", "header_bytes", "fields", "checks", "timestamp")
_FIELD_KEYS = ("signal", "offset", "type", "bit", "scale", "bias", "min", "max")
_CHECK_KEYS = ("offset", "type", "equals")
_NUMBER_KEYS = ("scale", "bias", "min", "max")  # belong to an analog signal, not to a bit field
_BYTE_ORDERS = ("big", "little")

# Deliberately without numbers and without the expected value: the message ends up in logs and must not
# reveal the key of the check.
_CHECK_MISMATCH = "check word mismatch: not a telegram from the expected sender, or a wrong format"


class FormatError(ValueError):
    """The format is invalid. The text names the signal, the key or the index; never the value of a check."""


class PacketError(ValueError):
    """A datagram does not match the format. The text is meant for the logs and contains no packet contents."""


@dataclass(frozen=True)
class _Type:
    name: str
    code: str  # struct-code
    size: int
    integer: bool
    signed: bool
    low: int | None  # limits of an integer type; None for floats
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
        _integer("int64", "q", 8, True),  # deliberately no uint64: the result is a float and that cannot hold 2**64
        _Type("float32", "f", 4, False, True, None, None),
        _Type("float64", "d", 8, False, True, None, None),
    )
}
# The type names as ibaPDA uses them (IEC 61131-3).
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
_FLOAT32 = struct.Struct("<f")  # only to report overflow in encode()


def _find_type(name: str) -> _Type | None:
    key = name.lower()  # lower() and not casefold(): casefold also lets a long s (U+017F) pass for an s
    return _TYPES.get(_ALIASES.get(key, key))


@dataclass(frozen=True)
class Field:
    """A signal in the telegram. `type` is the canonical name; bit, min and max are None when absent."""

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
    """A constant word that must be present in every telegram."""

    offset: int
    type: str
    equals: int = field(repr=False)  # never in logs or tracebacks: it is the key against stray packets


# ---- reading the format -------------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)  # a JSON boolean is not a number


def _finite(value: Any) -> float | None:
    """value as a float, or None if it is not a number (a boolean does not count) or not finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:  # an int above the float range
        return None
    return number if math.isfinite(number) else None


def _ascii(text: str) -> str:
    """For use in a message: everything outside ASCII as an escape, so a Windows console does not choke on it."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _shorten(value: Any, limit: int = 40) -> str:
    """For use in a message: a key or type name from the file, shortened and in ASCII."""
    text = ascii(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _reject_unknown_keys(entry: Mapping, allowed: tuple[str, ...], where: str) -> None:
    for key in entry:
        if key not in allowed:
            raise FormatError(f"{where}: unknown key {_shorten(key)} (allowed: {', '.join(allowed)})")


def _require(entry: Mapping, key: str, where: str) -> Any:
    if key not in entry:
        raise FormatError(f"{where}: {key!r} is missing")
    return entry[key]


def _parse_offset(where: str, value: Any) -> int:
    if not _is_int(value) or value < 0:
        raise FormatError(f"{where}: offset must be an integer of 0 or more")
    return value


def _parse_type(where: str, value: Any) -> _Type:
    kind = _find_type(value) if isinstance(value, str) else None
    if kind is None:
        raise FormatError(f"{where}: unknown type {_shorten(value)} (choose from: {_TYPE_HELP})")
    return kind


def _parse_field(index: int, entry: Any) -> Field:
    where = f"fields[{index}]"
    if not isinstance(entry, Mapping):
        raise FormatError(f"{where} must be a JSON object, not {type(entry).__name__}")
    name = entry.get("signal")
    if isinstance(name, str) and name:
        where += f" ({name!a})"
    _reject_unknown_keys(entry, _FIELD_KEYS, where)
    for key in ("signal", "offset", "type"):
        _require(entry, key, where)
    if not isinstance(name, str) or not name:
        raise FormatError(f"{where}: signal must be a non-empty string")
    offset = _parse_offset(where, entry["offset"])
    kind = _parse_type(where, entry["type"])

    bit = None
    if "bit" in entry:
        bit = entry["bit"]
        if not kind.integer:
            raise FormatError(f"{where}: bit belongs to an integer type, not to {kind.name}")
        if not _is_int(bit) or not 0 <= bit < 8 * kind.size:
            last = 8 * kind.size - 1
            raise FormatError(f"{where}: bit must be an integer from 0 to {last} inclusive ({kind.name})")
        extras = [key for key in _NUMBER_KEYS if key in entry]
        if extras:
            raise FormatError(f"{where}: a bit field cannot have {' or '.join(extras)}")

    settings: dict[str, float] = {}
    for key in _NUMBER_KEYS:
        if key in entry:
            number = _finite(entry[key])
            if number is None:
                raise FormatError(f"{where}: {key} must be a finite number")
            settings[key] = number
    if settings.get("scale") == 0:
        raise FormatError(f"{where}: scale must not be 0")
    low, high = settings.get("min"), settings.get("max")
    if low is not None and high is not None and low > high:
        raise FormatError(f"{where}: min ({low:g}) is greater than max ({high:g})")
    return Field(name, offset, kind.name, bit, settings.get("scale", 1.0), settings.get("bias", 0.0), low, high)


def _parse_check(index: int, entry: Any) -> Check:
    where = f"checks[{index}]"
    if not isinstance(entry, Mapping):
        raise FormatError(f"{where} must be a JSON object, not {type(entry).__name__}")
    _reject_unknown_keys(entry, _CHECK_KEYS, where)
    for key in _CHECK_KEYS:
        _require(entry, key, where)
    offset = _parse_offset(where, entry["offset"])
    kind = _parse_type(where, entry["type"])
    if not kind.integer:
        raise FormatError(f"{where}: only integer types can be a check, not {kind.name}")
    equals = entry["equals"]
    if not _is_int(equals) or not kind.low <= equals <= kind.high:
        # The value itself stays out of the message: it is the key of the check.
        raise FormatError(f"{where}: equals must be an integer that fits in {kind.name}")
    return Check(offset, kind.name, equals)


def _parse_timestamp(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise FormatError('timestamp must be a JSON object, for example {"source": "receive"}')
    _reject_unknown_keys(value, ("source",), "timestamp")
    if _require(value, "source", "timestamp") != "receive":
        raise FormatError('timestamp: source must be "receive" (the only supported source)')


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """json.loads silently keeps the last of a duplicate key; for byte_order that is dangerous."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FormatError(f"key {_shorten(key)} occurs twice in the same JSON object")
        result[key] = value
    return result


# ---- checking the layout and turning it into a struct -----------------------------------------------


def _check_bounds(label: str, offset: int, kind: _Type, body: int, header: int) -> None:
    if offset + kind.size > body:
        hint = " (offsets count from the end of the header)" if header else ""
        raise FormatError(
            f"{label}: {kind.size} bytes ({kind.name}) at offset {offset} do not fit in the {body} bytes "
            f"after the header{hint}"
        )


def _wire_order(
    fields: tuple[Field, ...], checks: tuple[Check, ...], body: int, header: int
) -> list[tuple[int, str]]:
    """Check names, bounds and overlap; return the words read (offset, type) in order of offset.

    Bit fields with the same offset and type share a word. Any other overlap is an error.
    """
    first_with_name: dict[str, int] = {}
    claims: list[tuple[int, int, str, tuple[int, str]]] = []  # (start, end, label, word)
    bit_words: set[tuple[int, str]] = set()
    for index, f in enumerate(fields):
        label = f"fields[{index}] ({f.signal!a})"
        if f.signal in first_with_name:
            raise FormatError(f"{label}: signal already occurs in fields[{first_with_name[f.signal]}]")
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
        if after[0] < before[1]:  # sorted by start: if the neighbours are clear, nothing overlaps
            raise FormatError(f"{after[2]} overlaps with {before[2]}")
    return [claim[3] for claim in claims]


def _struct_for(byte_order: str, header: int, body: int, order: list[tuple[int, str]]) -> struct.Struct:
    """The whole telegram as one struct: header and gaps as pad bytes, the words in order of offset."""
    parts = ["<" if byte_order == "little" else ">"]  # standard sizes, no alignment
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
    """How decode() gets a signal out of the unpacked tuple."""

    signal: str
    place: int  # position in the unpacked tuple
    mask: int  # bit mask of a bit field; 0 for an analog signal
    scale: float
    bias: float
    low: float  # -inf without min: no bound means no limit, and that saves a comparison
    high: float


@dataclass(frozen=True)
class _Compiled:
    """Everything decode() and encode() need per telegram, precomputed."""

    layout: struct.Struct  # the whole telegram in one go: header and gaps as pad bytes
    slots: dict[tuple[int, str], int]  # word (offset, type) -> position in the unpacked tuple
    checks: tuple[tuple[int, int], ...]  # (position, expected value)
    reads: tuple[_Read, ...]
    template: tuple[int, ...]  # for encode(): zeros, with the check words filled in


@dataclass(frozen=True)
class PacketFormat:
    """What a telegram looks like and how it is read and written.

    Build it with from_dict() or load(): those check the format completely. A wrong format must fail at load
    time, not only once silent nonsense shows up in the charts.
    """

    byte_order: str  # "big" or "little"
    packet_bytes: int  # exact length of a datagram, including the header
    header_bytes: int  # ignored header; all offsets count from the end of it
    fields: tuple[Field, ...]
    checks: tuple[Check, ...] = ()
    signals: tuple[str, ...] = field(init=False, repr=False, compare=False)  # in the order of fields
    _compiled: _Compiled = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.byte_order not in _BYTE_ORDERS:
            raise FormatError('byte_order must be "big" or "little"')
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "checks", tuple(self.checks))
        if not self.fields:
            raise FormatError("fields must not be empty")
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
        # struct.Struct cannot be pickled; the definition is enough, the rest is recomputed
        return (type(self), (self.byte_order, self.packet_bytes, self.header_bytes, self.fields, self.checks))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, required_signals: Iterable[str] | None = None) -> "PacketFormat":
        """Build the format from the JSON structure. Anything that is wrong is a FormatError with an English
        message that names the signal, the key or the index.

        required_signals: the signals in "fields" must be exactly this set.
        """
        if not isinstance(data, Mapping):
            raise FormatError(f"format must be a JSON object, not {type(data).__name__}")
        _reject_unknown_keys(data, _TOP_KEYS, "format")

        if "byte_order" not in data:
            raise FormatError('format: byte_order is missing; choose "big" or "little" (deliberately no default)')
        byte_order = data["byte_order"]
        if not isinstance(byte_order, str) or byte_order not in _BYTE_ORDERS:
            raise FormatError('format: byte_order must be "big" or "little"')
        packet_bytes = _require(data, "packet_bytes", "format")
        if not _is_int(packet_bytes) or not 1 <= packet_bytes <= MAX_PACKET_BYTES:
            raise FormatError(f"format: packet_bytes must be an integer from 1 to {MAX_PACKET_BYTES} inclusive")
        header_bytes = data.get("header_bytes", 0)
        if not _is_int(header_bytes) or header_bytes < 0:
            raise FormatError("format: header_bytes must be an integer of 0 or more")
        if header_bytes >= packet_bytes:
            raise FormatError(
                f"format: header_bytes ({header_bytes}) leaves no room in packet_bytes ({packet_bytes})"
            )

        raw_fields = _require(data, "fields", "format")
        if not isinstance(raw_fields, (list, tuple)) or not raw_fields:
            raise FormatError("format: fields must be a list with at least one field")
        fields = tuple(_parse_field(index, entry) for index, entry in enumerate(raw_fields))
        raw_checks = data.get("checks", [])
        if not isinstance(raw_checks, (list, tuple)):
            raise FormatError("format: checks must be a list")
        checks = tuple(_parse_check(index, entry) for index, entry in enumerate(raw_checks))
        if "timestamp" in data:
            _parse_timestamp(data["timestamp"])

        fmt = cls(byte_order, packet_bytes, header_bytes, fields, checks)
        if required_signals is not None:
            wanted, present = {str(s) for s in required_signals}, set(fmt.signals)
            if wanted != present:
                parts = []
                if wanted - present:
                    parts.append("missing: " + ", ".join(sorted(wanted - present)))
                if present - wanted:
                    parts.append("unknown: " + ", ".join(sorted(present - wanted)))
                raise FormatError("the signals in fields do not match the expected ones; " + "; ".join(parts))
        return fmt

    @classmethod
    def load(cls, path: str | PathLike[str], *, required_signals: Iterable[str] | None = None) -> "PacketFormat":
        """Read the format from a JSON file (UTF-8, with or without BOM).

        The messages name the path and never the contents of the file: it may be the wrong file by accident,
        for example one with an ApiKey in it. For unreadable JSON the message only gives the location of the
        error; for a wrong format it names keys, signal and type names and indexes, and never a check value.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8-sig")  # Notepad likes to put a BOM in front
        except FileNotFoundError:
            raise FormatError(f"Format file not found: {path}") from None
        except UnicodeDecodeError:
            raise FormatError(f"Format file {path} is not UTF-8; save it as UTF-8") from None
        except (OSError, ValueError) as err:
            raise FormatError(f"Format file {path} cannot be read ({type(err).__name__})") from None
        try:
            data = json.loads(text, object_pairs_hook=_no_duplicate_keys)
        except FormatError as err:  # from the hook; FormatError is a ValueError and so must come first
            raise FormatError(f"Format file {path}: {err}") from None
        except json.JSONDecodeError as err:
            where = f"line {err.lineno}, column {err.colno}"  # the location, not the text around it
            raise FormatError(f"Format file {path} is not valid JSON ({where})") from None
        except (ValueError, RecursionError) as err:  # e.g. a number of thousands of digits, or absurd nesting
            raise FormatError(f"Format file {path} is not valid JSON ({type(err).__name__})") from None
        try:
            return cls.from_dict(data, required_signals=required_signals)
        except FormatError as err:
            raise FormatError(f"Format file {path}: {err}") from None

    def decode(self, packet: bytes | bytearray | memoryview) -> dict[str, float]:
        """Read all signals from a datagram. Raises PacketError and nothing else, whatever comes in.

        The values are in the order of `fields`: a bit field as 1.0 or 0.0, the rest as raw * scale + bias.
        """
        plan = self._compiled
        try:
            words = plan.layout.unpack(packet)  # also checks the length
        except Exception:  # struct.error for a wrong length, otherwise a strange object (not bytes, released)
            raise PacketError(self._misfit(packet)) from None
        for place, expected in plan.checks:
            if words[place] != expected:
                raise PacketError(_CHECK_MISMATCH)
        values: dict[str, float] = {}
        for signal, place, mask, scale, bias, low, high in plan.reads:
            word = words[place]
            if mask:
                values[signal] = 1.0 if word & mask else 0.0  # also right for the sign bit of a signed number
                continue
            value = word * scale + bias
            if not math.isfinite(value):
                raise PacketError(f"{_ascii(signal)}: value is NaN or infinite")
            if not low <= value <= high:  # the value itself stays out of the message: it ends up in the logs
                raise PacketError(f"{_ascii(signal)}: value is outside the range ({_bounds_text(low, high)})")
            values[signal] = value
        return values

    def _misfit(self, packet: Any) -> str:
        try:
            actual = memoryview(packet).nbytes
        except ValueError:
            return "telegram can no longer be read (released memoryview)"
        except Exception:  # usually a TypeError: not a bytes-like object
            return f"telegram must be bytes, got {type(packet).__name__}"
        if actual != self.packet_bytes:
            return f"telegram is {actual} bytes, expected {self.packet_bytes} bytes"
        return "telegram cannot be read as contiguous bytes"

    def encode(self, values: Mapping[str, float]) -> bytes:
        """The inverse of decode(), for the simulator and the tests.

        The header, the gaps and the bits that are not a signal stay zero; the check words are filled in.
        Bit fields in the same word are merged (bit set for a value above 0.5). min and max are not enforced
        here, so a test can also build a telegram outside the range.

        Raises ValueError for a missing or unknown signal, a non-finite value or a value that does not fit
        in the type. decode(encode(v)) returns v to the precision of the type: exact for float64 without
        scale and bias, otherwise within half a step (scale/2) for integer types and float32 precision.
        """
        plan = self._compiled
        unknown = sorted(_shorten(name) for name in values if name not in self.signals)
        missing = [ascii(name) for name in self.signals if name not in values]
        if unknown or missing:
            problems = []
            if missing:
                problems.append("signal missing: " + ", ".join(missing))
            if unknown:
                problems.append("unknown signal: " + ", ".join(unknown))
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
                    except OverflowError:  # outside the float32 range; struct reports that without a signal name
                        raise _does_not_fit(f, number) from None
                raw[place] = scaled
        for place, (word, kind) in bit_words.items():
            if kind.signed and word >= 1 << (8 * kind.size - 1):
                word -= 1 << (8 * kind.size)  # struct wants a signed number when the code says so
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
    return ValueError(f"{_ascii(f.signal)}: value {number:g} does not fit in {f.type}")


def _as_finite_float(signal: str, value: Any) -> float:
    """Anything float() reads as a number (including bool and numpy scalars), except text: float("1.5") succeeds,
    but then the caller has made a mistake."""
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{_ascii(signal)}: value must be a number, not {type(value).__name__}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{_ascii(signal)}: value must be a number, not {type(value).__name__}") from None
    except OverflowError:  # a gigantic int
        raise ValueError(f"{_ascii(signal)}: value is too large") from None
    if not math.isfinite(number):
        raise ValueError(f"{_ascii(signal)}: value is NaN or infinite")
    return number
