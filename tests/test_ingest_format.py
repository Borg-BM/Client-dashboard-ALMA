"""Tests van alma.ingest.format: het formaat van een ibaPDA-UDP-telegram, decode() en encode().

De verwachte bytes staan hier als letterlijke waarden (of via int.to_bytes), niet via struct of encode(),
zodat decode() niet alleen tegen zijn eigen spiegelbeeld wordt gecontroleerd. Alles is deterministisch:
willekeur komt uit een vaste seed, en er is geen netwerk, klok of echte iba.
"""

import copy
import dataclasses
import itertools
import json
import math
import pickle
import random
import re
import struct
from fractions import Fraction

import numpy as np
import pytest

from alma.config import SIGNALS
from alma.ingest import format as format_module
from alma.ingest.format import MAX_PACKET_BYTES, Check, Field, FormatError, PacketError, PacketFormat

ORDERS = ("big", "little")
SIZES = {
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "uint32": 4,
    "int32": 4,
    "int64": 8,
    "float32": 4,
    "float64": 8,
}
INTEGER_TYPES = [name for name in SIZES if not name.startswith("float")]
ALIASES = {
    "BYTE": "uint8",
    "SINT": "int8",
    "WORD": "uint16",
    "INT": "int16",
    "DWORD": "uint32",
    "DINT": "int32",
    "LINT": "int64",
    "FLOAT": "float32",
    "DOUBLE": "float64",
}
INT_LIMITS = {
    "uint8": (0, 255),
    "int8": (-128, 127),
    "uint16": (0, 65535),
    "int16": (-32768, 32767),
    "uint32": (0, 2**32 - 1),
    "int32": (-(2**31), 2**31 - 1),
    "int64": (-(2**63), 2**63 - 1),
}
# type -> (kleinste en grootste float die nog past, de eerstvolgende die net niet meer past aan beide kanten)
LIMIT_FLOATS = {
    "uint8": (0.0, 255.0, -1.0, 256.0),
    "int8": (-128.0, 127.0, -129.0, 128.0),
    "uint16": (0.0, 65535.0, -1.0, 65536.0),
    "int16": (-32768.0, 32767.0, -32769.0, 32768.0),
    "uint32": (0.0, 4294967295.0, -1.0, 4294967296.0),
    "int32": (-2147483648.0, 2147483647.0, -2147483649.0, 2147483648.0),
    # float(2**63 - 1) is al 2**63; de grootste float eronder ligt 1024 lager, de eerste onder -2**63 ligt 2048 lager
    "int64": (-(2.0**63), 2.0**63 - 1024, -(2.0**63) - 2048, 2.0**63),
}
# type -> [(waarde, bytes op de draad bij big endian als hex)]. Little endian is dezelfde reeks omgekeerd.
KNOWN = {
    "uint8": [(0, "00"), (1, "01"), (255, "ff")],
    "int8": [(-128, "80"), (-1, "ff"), (0, "00"), (127, "7f")],
    "uint16": [(0, "0000"), (4660, "1234"), (65535, "ffff")],
    "int16": [(-32768, "8000"), (-2, "fffe"), (258, "0102"), (32767, "7fff")],
    "uint32": [(0, "00000000"), (16909060, "01020304"), (4294967295, "ffffffff")],
    "int32": [(-2147483648, "80000000"), (-2, "fffffffe"), (305419896, "12345678"), (2147483647, "7fffffff")],
    "int64": [
        (0, "0000000000000000"),
        (-2, "fffffffffffffffe"),
        (1108152157446, "0000010203040506"),
        (2**62, "4000000000000000"),
        (-(2**63), "8000000000000000"),
    ],
    "float32": [(1.5, "3fc00000"), (-2.0, "c0000000"), (0.0, "00000000"), (0.10000000149011612, "3dcccccd")],
    "float64": [
        (1.5, "3ff8000000000000"),
        (-2.0, "c000000000000000"),
        (0.0, "0000000000000000"),
        (0.1, "3fb999999999999a"),
    ],
}


def wire(hexbytes: str, order: str) -> bytes:
    """De bytes zoals ze op de draad staan: de big-endian tekst, omgekeerd bij little endian."""
    data = bytes.fromhex(hexbytes)
    return data if order == "big" else data[::-1]


def make(byte_order: str, *fields: dict, packet_bytes: int, **extra) -> PacketFormat:
    return PacketFormat.from_dict(
        {"byte_order": byte_order, "packet_bytes": packet_bytes, "fields": list(fields), **extra}
    )


def single(byte_order: str, type_name: str, signal: str = "x", **field_extra) -> PacketFormat:
    """Een telegram met een enkel signaal op offset 0. type_name mag ook een alias zijn, in elke schrijfwijze."""
    entry = {"signal": signal, "offset": 0, "type": type_name, **field_extra}
    canonical = ALIASES.get(type_name.upper(), type_name.lower())
    return make(byte_order, entry, packet_bytes=SIZES[canonical])


def format_error(data, **kwargs) -> str:
    with pytest.raises(FormatError) as caught:
        PacketFormat.from_dict(data, **kwargs)
    return str(caught.value)


# ---- types en bytes -------------------------------------------------------------------------------------


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("type_name", KNOWN)
def test_every_type_reads_and_writes_its_documented_bytes(type_name, order):
    fmt = single(order, type_name)
    for value, hexbytes in KNOWN[type_name]:
        assert fmt.decode(wire(hexbytes, order)) == {"x": value}
        assert fmt.encode({"x": value}) == wire(hexbytes, order)


def test_decode_gives_python_floats_for_every_type():
    for type_name, samples in KNOWN.items():
        fmt = single("big", type_name)
        for _, hexbytes in samples:
            assert type(fmt.decode(wire(hexbytes, "big"))["x"]) is float


SPELLINGS = [
    (spelling, canonical)
    for alias, canonical in ALIASES.items()
    for spelling in (alias, alias.lower(), alias.capitalize())
] + [(spelling, name) for name in SIZES for spelling in (name, name.upper(), name.capitalize())]


@pytest.mark.parametrize("spelling, canonical", SPELLINGS, ids=[f"{s}={c}" for s, c in SPELLINGS])
def test_type_names_are_case_insensitive_and_stored_canonically(spelling, canonical):
    assert single("big", spelling).fields[0].type == canonical


def test_an_alias_reads_exactly_like_its_canonical_type():
    for alias, canonical in ALIASES.items():
        for _, hexbytes in KNOWN[canonical]:
            packet = wire(hexbytes, "big")
            assert single("big", alias).decode(packet) == single("big", canonical).decode(packet)


@pytest.mark.parametrize(
    "name",
    ["uint64", "UINT64", "ULINT", "UINT", "REAL", "bool", "", " WORD", "WORD ", "uint 8", "int128", "\u017fint"],
    ids=ascii,
)
def test_other_type_names_are_rejected_and_named(name):
    # "\u017fint" is een lange s: casefold() zou er "sint" (SINT) van maken, lower() niet
    data = valid()
    data["fields"][1]["type"] = name
    message = format_error(data)
    assert ascii(name) in message and "level" in message  # de verkeerde naam en het signaal
    assert "uint32" in message  # en wat wel mag


@pytest.mark.parametrize("header", [0, 4])
@pytest.mark.parametrize("type_name", SIZES)
def test_a_field_must_fit_exactly_up_to_the_end_of_the_packet(type_name, header):
    size = SIZES[type_name]

    def build(offset, body):
        entry = {"signal": "x", "offset": offset, "type": type_name}
        return {"byte_order": "big", "packet_bytes": header + body, "header_bytes": header, "fields": [entry]}

    PacketFormat.from_dict(build(0, size))  # past precies
    PacketFormat.from_dict(build(3, size + 3))
    assert "x" in format_error(build(1, size))  # een byte te ver
    if size > 1:
        assert "x" in format_error(build(0, size - 1))  # een byte te kort


# ---- decode: lengte en invoer ----------------------------------------------------------------------------


def test_decode_returns_the_signals_in_field_order_not_offset_order():
    fmt = make(
        "big",
        {"signal": "late", "offset": 2, "type": "uint16"},
        {"signal": "early", "offset": 0, "type": "uint16"},
        packet_bytes=4,
    )
    assert fmt.signals == ("late", "early")
    assert list(fmt.decode(b"\x00\x01\x00\x02").items()) == [("late", 2.0), ("early", 1.0)]


def test_each_decode_returns_a_dict_of_its_own():
    fmt = single("big", "uint8")
    first = fmt.decode(b"\x01")
    first["x"] = 99.0
    first["extra"] = 1.0
    assert fmt.decode(b"\x01") == {"x": 1.0}


def test_decode_accepts_bytes_bytearray_and_memoryview():
    fmt = single("big", "uint16")
    for packet in (b"\x01\x02", bytearray(b"\x01\x02"), memoryview(b"\x01\x02"), memoryview(bytearray(b"\x01\x02"))):
        assert fmt.decode(packet) == {"x": 258.0}
    buffer = bytearray(b"\xff\xff\x01\x02\xff")  # recv_into: het telegram is een stuk van een grotere buffer
    assert fmt.decode(memoryview(buffer)[2:4]) == {"x": 258.0}


@pytest.mark.parametrize("length", [0, 1, 27, 29, 56, 70_000, 5_000_000])
def test_a_telegram_of_the_wrong_length_is_rejected_stating_both_lengths(length):
    fmt = make("big", {"signal": "x", "offset": 0, "type": "float32"}, packet_bytes=28)
    with pytest.raises(PacketError) as caught:
        fmt.decode(bytes(length))
    assert f"{length} bytes" in str(caught.value) and "28 bytes" in str(caught.value)


@pytest.mark.parametrize(
    "junk", [None, "ab", 258, 2.5, [1, 2], (1, 2), {"a": 1}, object(), range(2)], ids=lambda junk: type(junk).__name__
)
def test_decode_rejects_input_that_is_not_bytes_like(junk):
    with pytest.raises(PacketError, match="bytes"):
        single("big", "uint16").decode(junk)


def test_decode_rejects_memoryviews_it_cannot_read():
    fmt = single("big", "uint16")
    released = memoryview(bytes(2))
    released.release()
    for view in (released, memoryview(bytes(4))[::2]):  # vrijgegeven, en niet aaneengesloten
        with pytest.raises(PacketError):
            fmt.decode(view)


def test_decode_turns_every_failure_of_a_strange_object_into_a_packet_error():
    class Hostile:
        def __buffer__(self, flags):  # vanaf Python 3.12 kan een klasse zelf een buffer leveren
            raise RuntimeError("geen pakket")

        def __len__(self):
            raise RuntimeError("geen pakket")

    with pytest.raises(PacketError):
        single("big", "uint16").decode(Hostile())


def test_header_is_ignored_and_offsets_count_from_the_end_of_it():
    for order in ORDERS:
        fmt = make(order, {"signal": "x", "offset": 0, "type": "uint16"}, packet_bytes=6, header_bytes=4)
        body = wire("0102", order)
        assert fmt.decode(b"\x00\x00\x00\x00" + body) == {"x": 258.0}
        assert fmt.decode(b"\xde\xad\xbe\xef" + body) == {"x": 258.0}  # wat er in de header staat telt niet


def test_bytes_that_no_field_or_check_covers_are_ignored():
    fmt = make(
        "big",
        {"signal": "a", "offset": 1, "type": "uint8"},
        {"signal": "b", "offset": 5, "type": "uint8"},
        packet_bytes=8,
    )
    assert fmt.decode(bytes([0xAA, 1, 0xBB, 0xCC, 0xDD, 2, 0xEE, 0xFF])) == {"a": 1.0, "b": 2.0}
    assert fmt.decode(bytes([0, 1, 0, 0, 0, 2, 0, 0])) == {"a": 1.0, "b": 2.0}


# ---- bitvelden --------------------------------------------------------------------------------------------


def bit_fields(type_name: str, count: int) -> list[dict]:
    return [{"signal": f"b{n}", "offset": 0, "type": type_name, "bit": n} for n in range(count)]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("type_name", INTEGER_TYPES)
def test_bit_n_is_the_nth_least_significant_bit_of_the_word(type_name, order):
    size = SIZES[type_name]
    fmt = make(order, *bit_fields(type_name, 8 * size), packet_bytes=size)
    for n in range(8 * size):
        word = (1 << n).to_bytes(size, order)  # het woord in de byte-volgorde van het formaat
        decoded = fmt.decode(word)
        assert [name for name, value in decoded.items() if value == 1.0] == [f"b{n}"]
        assert all(value in (0.0, 1.0) and type(value) is float for value in decoded.values())
        assert fmt.encode({f"b{k}": float(k == n) for k in range(8 * size)}) == word


@pytest.mark.parametrize("order", ORDERS)
def test_bit_7_of_a_byte(order):
    fmt = make(order, {"signal": "top", "offset": 0, "type": "BYTE", "bit": 7}, packet_bytes=1)
    assert fmt.decode(b"\x80") == {"top": 1.0}
    assert fmt.decode(b"\x7f") == {"top": 0.0}
    assert fmt.encode({"top": 1}) == b"\x80"


@pytest.mark.parametrize("order", ORDERS)
def test_bit_31_of_a_dword(order):
    fmt = make(order, {"signal": "top", "offset": 0, "type": "DWORD", "bit": 31}, packet_bytes=4)
    assert fmt.decode(wire("80000000", order)) == {"top": 1.0}
    assert fmt.decode(wire("7fffffff", order)) == {"top": 0.0}
    assert fmt.encode({"top": 1}) == wire("80000000", order)


def test_the_byte_order_decides_which_bit_a_byte_holds():
    fields = [
        {"signal": "low", "offset": 0, "type": "WORD", "bit": 0},
        {"signal": "high", "offset": 0, "type": "WORD", "bit": 8},
    ]
    assert make("big", *fields, packet_bytes=2).decode(b"\x01\x00") == {"low": 0.0, "high": 1.0}
    assert make("little", *fields, packet_bytes=2).decode(b"\x01\x00") == {"low": 1.0, "high": 0.0}


@pytest.mark.parametrize("order", ORDERS)
def test_bit_fields_in_one_word_are_or_ed_together_by_encode(order):
    fmt = make(
        order,
        {"signal": "a", "offset": 0, "type": "WORD", "bit": 0},
        {"signal": "b", "offset": 0, "type": "uint16", "bit": 15},  # alias en canonieke naam zijn hetzelfde type
        {"signal": "c", "offset": 0, "type": "WORD", "bit": 4},
        {"signal": "d", "offset": 2, "type": "BYTE", "bit": 7},
        packet_bytes=3,
    )
    for a, b, c, d in itertools.product((0, 1), repeat=4):
        word = a | (c << 4) | (b << 15)
        expected = word.to_bytes(2, order) + bytes([d << 7])
        assert fmt.encode({"a": a, "b": b, "c": c, "d": d}) == expected
        assert fmt.decode(expected) == {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}


def test_a_bit_is_set_for_values_above_one_half():
    fmt = single("big", "uint8", bit=0)
    for value, expected in [(0, 0), (0.0, 0), (0.5, 0), (-1.0, 0), (0.5000001, 1), (1, 1), (2.5, 1)]:
        assert fmt.encode({"x": value}) == bytes([expected]), value
    assert fmt.encode({"x": True}) == b"\x01" and fmt.encode({"x": False}) == b"\x00"


def test_a_bit_field_ignores_the_other_bits_of_its_word_when_decoding():
    fmt = single("big", "uint16", bit=3)
    assert fmt.decode(b"\xff\xf7") == {"x": 0.0}  # alle andere bits staan aan
    assert fmt.decode(b"\x00\x08") == {"x": 1.0}


# ---- schaal, bias, min en max -----------------------------------------------------------------------------

SCALED = [  # type, bytes bij big endian, scale, bias, verwachte waarde
    ("uint16", "03e8", 0.1, -50.0, 50.0),  # ruw 1000
    ("int16", "fffb", -2.0, 1.0, 11.0),  # ruw -5
    ("uint8", "ff", 0.5, 0.0, 127.5),
    ("int32", "fffffff6", 0.001, 100.0, 99.99),  # ruw -10
    ("float32", "40200000", 2.0, 1.0, 6.0),  # 2.5 * 2 + 1
    ("float64", "c000000000000000", 0.5, 0.25, -0.75),  # -2 * 0.5 + 0.25
]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("type_name, hexbytes, scale, bias, expected", SCALED)
def test_decode_is_raw_times_scale_plus_bias(type_name, hexbytes, scale, bias, expected, order):
    fmt = single(order, type_name, scale=scale, bias=bias)
    assert fmt.decode(wire(hexbytes, order))["x"] == pytest.approx(expected, rel=1e-12)


def test_scale_and_bias_default_to_one_and_zero():
    fmt = single("big", "int16")
    assert (fmt.fields[0].scale, fmt.fields[0].bias) == (1.0, 0.0)
    assert fmt.decode(b"\xff\xfe") == {"x": -2.0}


def test_min_and_max_are_inclusive_and_apply_after_scaling():
    fmt = single("big", "int16", signal="level", scale=0.5, bias=10.0, min=9.5, max=20.0)

    def raw(n):
        return n.to_bytes(2, "big", signed=True)

    assert fmt.decode(raw(-1)) == {"level": 9.5}  # precies op de ondergrens
    assert fmt.decode(raw(20)) == {"level": 20.0}  # precies op de bovengrens
    for outside in (-2, 21):  # 9.0 en 20.5
        with pytest.raises(PacketError, match="level"):
            fmt.decode(raw(outside))


def test_min_and_max_can_be_given_separately():
    only_min = single("big", "int8", min=0)
    only_max = single("big", "int8", max=0)
    assert only_min.decode(b"\x00") == {"x": 0.0} and only_min.decode(b"\x7f") == {"x": 127.0}
    assert only_max.decode(b"\x00") == {"x": 0.0} and only_max.decode(b"\xff") == {"x": -1.0}
    with pytest.raises(PacketError, match="x"):
        only_min.decode(b"\xff")
    with pytest.raises(PacketError, match="x"):
        only_max.decode(b"\x01")


def test_a_rejected_range_names_only_its_own_signal_and_not_the_received_value():
    fmt = make(
        "big",
        {"signal": "tower_x", "offset": 0, "type": "int16", "max": 10},
        {"signal": "tower_y", "offset": 2, "type": "int16", "max": 10},
        packet_bytes=4,
    )
    with pytest.raises(PacketError) as caught:
        fmt.decode((5).to_bytes(2, "big") + (7777).to_bytes(2, "big"))
    message = str(caught.value)
    assert "tower_y" in message and "tower_x" not in message
    assert "7777" not in message  # de melding belandt in de logs: geen telegraminhoud
    assert "10" in message  # wel de grens waar het aan ligt


NOT_FINITE = {
    "float32": ["7fc00000", "7f800000", "ff800000"],  # NaN, +oneindig, -oneindig
    "float64": ["7ff8000000000000", "7ff0000000000000", "fff0000000000000"],
}


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("type_name", NOT_FINITE)
def test_nan_and_infinity_are_rejected_naming_the_signal(type_name, order):
    fmt = single(order, type_name, signal="tower_x")
    for hexbytes in NOT_FINITE[type_name]:
        with pytest.raises(PacketError, match="tower_x"):
            fmt.decode(wire(hexbytes, order))


def test_nan_is_rejected_even_when_min_and_max_are_set():
    fmt = single("big", "float32", signal="tower_x", min=-1e30, max=1e30)
    for hexbytes in NOT_FINITE["float32"]:
        with pytest.raises(PacketError, match="tower_x"):
            fmt.decode(wire(hexbytes, "big"))


def test_a_value_that_overflows_when_scaled_is_rejected():
    # struct maakt hier de grensgevallen: 1.7e308 * 10 en 3e38 * 1e300 zijn in float64 oneindig
    big = single("big", "float64", signal="vessel_x", scale=10.0)
    with pytest.raises(PacketError, match="vessel_x"):
        big.decode(struct.pack(">d", 1.7e308))
    small = single("big", "float32", signal="vessel_x", scale=1e300)
    with pytest.raises(PacketError, match="vessel_x"):
        small.decode(struct.pack(">f", 3e38))


# ---- checks ---------------------------------------------------------------------------------------------

CHECK = 0xBEEF  # 48879: past in een WORD en komt niet toevallig voor als lengte of offset


def checked(order: str, check_type: str = "WORD", equals: int = CHECK) -> PacketFormat:
    return make(
        order,
        {"signal": "x", "offset": 2, "type": "uint16"},
        packet_bytes=4,
        checks=[{"offset": 0, "type": check_type, "equals": equals}],
    )


@pytest.mark.parametrize("order", ORDERS)
def test_matching_check_words_are_accepted_and_are_not_part_of_the_result(order):
    assert checked(order).decode(wire("beef", order) + wire("0102", order)) == {"x": 258.0}


@pytest.mark.parametrize("order", ORDERS)
def test_a_wrong_check_word_is_rejected(order):
    other = "little" if order == "big" else "big"
    body = wire("0102", order)
    for wrong in (wire("beef", other), wire("beee", order), wire("0000", order), wire("ffff", order)):
        with pytest.raises(PacketError, match="controlewoord"):
            checked(order).decode(wrong + body)


@pytest.mark.parametrize("equals", [0, 1, 7, 255, 4660, 48879, 65535])
def test_the_message_for_a_wrong_check_never_contains_the_expected_value(equals):
    fmt = checked("big", equals=equals)
    received = (equals + 1) % 65536
    with pytest.raises(PacketError) as caught:
        fmt.decode(received.to_bytes(2, "big") + b"\x01\x02")
    message = str(caught.value).lower()
    assert str(equals) not in message  # ook niet bij kleine waarden: er staan geen getallen in
    assert str(received) not in message  # en ook niet wat er wel binnenkwam
    if equals >= 4096:
        assert format(equals, "x") not in message


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("type_name", INTEGER_TYPES)
def test_check_words_of_every_integer_type_work_at_their_limits(type_name, order):
    size = SIZES[type_name]
    low, high = INT_LIMITS[type_name]
    for equals in (low, high, 1):
        fmt = make(
            order,
            {"signal": "x", "offset": size, "type": "uint8"},
            packet_bytes=size + 1,
            checks=[{"offset": 0, "type": type_name, "equals": equals}],
        )
        packet = fmt.encode({"x": 9})
        assert packet == equals.to_bytes(size, order, signed=low < 0) + b"\x09"
        assert fmt.decode(packet) == {"x": 9.0}
        broken = bytearray(packet)
        broken[0] ^= 0x01
        with pytest.raises(PacketError, match="controlewoord"):
            fmt.decode(bytes(broken))


def test_a_signed_check_word_matches_its_twos_complement_bytes():
    fmt = checked("big", check_type="INT", equals=-2)
    assert fmt.decode(b"\xff\xfe\x00\x01") == {"x": 1.0}
    with pytest.raises(PacketError):
        fmt.decode(b"\xff\xff\x00\x01")


def test_every_check_must_match_not_just_the_first():
    fmt = make(
        "big",
        {"signal": "x", "offset": 4, "type": "uint8"},
        packet_bytes=5,
        checks=[{"offset": 0, "type": "WORD", "equals": 0x1234}, {"offset": 2, "type": "WORD", "equals": 0x5678}],
    )
    assert fmt.decode(b"\x12\x34\x56\x78\x01") == {"x": 1.0}
    for broken in (b"\x12\x35\x56\x78\x01", b"\x12\x34\x56\x79\x01"):
        with pytest.raises(PacketError, match="controlewoord"):
            fmt.decode(broken)


def test_a_wrong_check_is_reported_before_the_values_are_looked_at():
    fmt = make(
        "big",
        {"signal": "tower_x", "offset": 2, "type": "float32"},
        packet_bytes=6,
        checks=[{"offset": 0, "type": "WORD", "equals": CHECK}],
    )
    with pytest.raises(PacketError, match="controlewoord"):
        fmt.decode(b"\x00\x00\x7f\xc0\x00\x00")  # verkeerd controlewoord en een NaN


def test_repr_never_shows_a_check_value():
    fmt = checked("big")
    assert "48879" not in repr(fmt) and "48879" not in repr(fmt.checks[0])
    assert fmt.checks[0].equals == CHECK  # maar de waarde zelf is gewoon bereikbaar


# ---- encode -------------------------------------------------------------------------------------------


def padded(order: str) -> PacketFormat:
    return PacketFormat.from_dict(
        {
            "byte_order": order,
            "packet_bytes": 10,
            "header_bytes": 3,
            "fields": [{"signal": "x", "offset": 1, "type": "uint16"}],
            "checks": [{"offset": 0, "type": "BYTE", "equals": 0xA5}, {"offset": 4, "type": "WORD", "equals": 0x1234}],
        }
    )


def test_encode_zeroes_header_and_padding_and_fills_in_the_check_words():
    assert padded("big").encode({"x": 258}) == bytes.fromhex("000000" "a5" "0102" "00" "1234" "00")
    assert padded("little").encode({"x": 258}) == bytes.fromhex("000000" "a5" "0201" "00" "3412" "00")
    for order in ORDERS:
        fmt = padded(order)
        assert fmt.decode(fmt.encode({"x": 258})) == {"x": 258.0}


def test_encode_returns_bytes_of_exactly_packet_bytes():
    for order, type_name in itertools.product(ORDERS, SIZES):
        packet = single(order, type_name).encode({"x": 1})
        assert type(packet) is bytes and len(packet) == SIZES[type_name]


def test_encode_inverts_scale_and_bias_and_rounds_to_the_nearest_integer():
    fmt = single("big", "int16", scale=0.1, bias=-50.0)
    cases = [(50.0, 1000), (50.04, 1000), (50.06, 1001), (-49.9, 1), (-50.0, 0), (-60.0, -100), (-60.06, -101)]
    for value, raw in cases:
        assert fmt.encode({"x": value}) == raw.to_bytes(2, "big", signed=True), value
    plain = single("big", "int8")
    for value, raw in [(1.4, 1), (1.6, 2), (-1.4, -1), (-1.6, -2)]:
        assert plain.encode({"x": value}) == raw.to_bytes(1, "big", signed=True), value


def test_encode_names_missing_and_unknown_signals():
    fmt = make(
        "big",
        {"signal": "tower_x", "offset": 0, "type": "float32"},
        {"signal": "tower_y", "offset": 4, "type": "float32"},
        packet_bytes=8,
    )
    with pytest.raises(ValueError, match="tower_y") as missing:
        fmt.encode({"tower_x": 1.0})
    with pytest.raises(ValueError, match="colour") as unknown:
        fmt.encode({"tower_x": 1.0, "tower_y": 2.0, "colour": 3.0})
    for caught in (missing, unknown):
        assert type(caught.value) is ValueError  # geen FormatError of PacketError: dat zijn andere fouten
    assert "ontbreekt" in str(missing.value) and "onbekend" in str(unknown.value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("type_name", [*SIZES, "bit"])
def test_encode_refuses_non_finite_values_naming_the_signal(type_name, value):
    if type_name == "bit":
        fmt = single("big", "uint16", signal="tower_x", bit=3)
    else:
        fmt = single("big", type_name, signal="tower_x")
    with pytest.raises(ValueError, match="tower_x"):
        fmt.encode({"tower_x": value})


@pytest.mark.parametrize("type_name", INTEGER_TYPES)
def test_encode_accepts_the_limits_of_an_integer_type_and_refuses_one_step_beyond(type_name):
    low, high, below, above = LIMIT_FLOATS[type_name]
    fmt = single("big", type_name, signal="level")
    for value in (low, high):
        assert fmt.decode(fmt.encode({"level": value})) == {"level": value}
    for value in (below, above):
        with pytest.raises(ValueError, match="level") as caught:
            fmt.encode({"level": value})
        assert type_name in str(caught.value)


def test_encode_rounds_to_the_nearest_integer_before_checking_the_range():
    fmt = single("big", "uint8")
    assert fmt.encode({"x": 255.4}) == b"\xff" and fmt.encode({"x": -0.4}) == b"\x00"
    for value in (255.6, -0.6):
        with pytest.raises(ValueError, match="x"):
            fmt.encode({"x": value})


def test_encode_checks_the_range_after_scale_and_bias_are_undone():
    fmt = single("big", "uint8", signal="level", scale=0.5, bias=10.0)
    assert fmt.encode({"level": 137.5}) == b"\xff"  # (137.5 - 10) / 0.5 = 255
    for value in (138.0, 9.0):  # 256 en -2
        with pytest.raises(ValueError, match="level"):
            fmt.encode({"level": value})


def test_encode_refuses_a_float32_overflow_but_not_float32_extremes():
    fmt = single("big", "float32", signal="level")
    assert fmt.decode(fmt.encode({"level": 3.4e38}))["level"] == pytest.approx(3.4e38, rel=1e-7)
    for value in (1e39, -1e39, 3.4028235677973366e38):  # de laatste rondt in float32 af naar oneindig
        with pytest.raises(ValueError, match="level") as caught:
            fmt.encode({"level": value})
        assert "float32" in str(caught.value)
    scaled = single("big", "float32", signal="level", scale=0.01)
    with pytest.raises(ValueError, match="level"):
        scaled.encode({"level": 1e38})  # past zelf wel, maar de ruwe waarde (1e40) niet


def test_encode_refuses_a_float64_that_overflows_after_unscaling():
    fmt = single("big", "float64", signal="level", scale=0.5)
    with pytest.raises(ValueError, match="level"):
        fmt.encode({"level": 1.7976931348623157e308})  # ruw is 3.6e308


@pytest.mark.parametrize("junk", [None, "1.5", b"1", [1.0], 1j, object(), {}], ids=lambda junk: type(junk).__name__)
def test_encode_refuses_values_that_are_not_numbers(junk):
    with pytest.raises(ValueError, match="x") as caught:
        single("big", "float64").encode({"x": junk})
    assert type(caught.value) is ValueError


def test_encode_refuses_an_absurdly_large_integer_with_a_value_error():
    with pytest.raises(ValueError, match="x") as caught:
        single("big", "float64").encode({"x": 10**400})
    assert type(caught.value) is ValueError


def test_encode_accepts_anything_that_is_a_number():
    fmt = single("big", "float64")
    for value in (3, True, Fraction(3, 2), np.float32(1.5), np.float64(2.5), np.int16(7), np.bool_(True)):
        assert fmt.decode(fmt.encode({"x": value})) == {"x": float(value)}


def test_encode_does_not_enforce_min_and_max_so_tests_can_build_bad_telegrams():
    fmt = single("big", "int16", signal="tower_x", max=10)
    packet = fmt.encode({"tower_x": 500})
    with pytest.raises(PacketError, match="tower_x"):
        fmt.decode(packet)


def test_encode_works_with_the_signals_in_any_order():
    fmt = make(
        "big",
        {"signal": "a", "offset": 0, "type": "uint8"},
        {"signal": "b", "offset": 1, "type": "uint8"},
        packet_bytes=2,
    )
    assert fmt.encode({"b": 2, "a": 1}) == b"\x01\x02"


# ---- heen en terug ---------------------------------------------------------------------------------------

SCALINGS = [(1.0, 0.0), (0.1, 0.0), (0.01, -100.0), (-0.5, 3.0)]


def raw_limits(type_name: str) -> tuple[int, int]:
    """Het bereik waarin we ruwe waarden trekken; bij int64 ruim binnen wat een float exact vasthoudt."""
    return (-(2**40), 2**40) if type_name == "int64" else INT_LIMITS[type_name]


@pytest.mark.parametrize("scale, bias", SCALINGS)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("type_name", INTEGER_TYPES)
def test_integer_types_round_trip_within_half_a_step(type_name, order, scale, bias):
    rng = random.Random(f"{type_name}/{order}/{scale}/{bias}")
    fmt = single(order, type_name, scale=scale, bias=bias)
    low, high = raw_limits(type_name)
    ends = sorted((low * scale + bias, high * scale + bias))
    values = [*ends, *(rng.uniform(*ends) for _ in range(60))]
    values += [rng.randint(low, high) * scale + bias for _ in range(20)]  # precies op het raster
    for value in values:
        packet = fmt.encode({"x": value})
        decoded = fmt.decode(packet)["x"]
        assert abs(decoded - value) <= abs(scale) / 2 + 4 * math.ulp(max(abs(value), 1.0))
        assert fmt.encode({"x": decoded}) == packet  # en nog eens: dezelfde bytes


@pytest.mark.parametrize("order", ORDERS)
def test_float64_round_trips_exactly(order):
    rng = random.Random(f"float64/{order}")
    fmt = single(order, "float64")
    values = [0.0, 1.0, -1.0, 0.1, math.pi, 1e-300, 5e-324, 1e300, 1.7976931348623157e308, -1.7976931348623157e308]
    values += [rng.uniform(-1, 1) * 10 ** rng.uniform(-300, 300) for _ in range(300)]
    for value in values:
        packet = fmt.encode({"x": value})
        assert fmt.decode(packet) == {"x": value}
        assert fmt.encode(fmt.decode(packet)) == packet


@pytest.mark.parametrize("order", ORDERS)
def test_float32_round_trips_within_float32_precision(order):
    rng = random.Random(f"float32/{order}")
    fmt = single(order, "float32")
    values = [0.0, 1.0, -1.0, 0.1, math.pi, 3.4e38, -3.4e38, 1.5e-30]
    values += [rng.uniform(-1, 1) * 10 ** rng.uniform(-30, 38) for _ in range(300)]
    for value in values:
        packet = fmt.encode({"x": value})
        decoded = fmt.decode(packet)["x"]
        assert abs(decoded - value) <= abs(value) * 2**-24
        assert fmt.encode({"x": decoded}) == packet


@pytest.mark.parametrize("scale, bias", [(0.5, 3.0), (0.001, -20.0), (-4.0, 0.25)])
@pytest.mark.parametrize("type_name", ["float32", "float64"])
def test_float_types_with_scale_and_bias_round_trip_approximately(type_name, scale, bias):
    rng = random.Random(f"{type_name}/{scale}/{bias}")
    fmt = single("big", type_name, scale=scale, bias=bias)
    for _ in range(200):
        value = rng.uniform(-1000, 1000)
        packet = fmt.encode({"x": value})
        decoded = fmt.decode(packet)["x"]
        if type_name == "float32":  # de afronding zit in de ruwe waarde, die door de bias groter is dan de uitkomst
            assert abs(decoded - value) <= abs(value - bias) * 2**-24 + 2 * math.ulp(value)
            assert fmt.encode({"x": decoded}) == packet
        else:
            assert abs(decoded - value) <= 4 * math.ulp(max(abs(value), abs(bias)))


ALMA_VALUES = {
    "system_mode": 3.0,
    "pressure_bore_skid1": 123.5,
    "pressure_rod_skid1": 40.25,
    "pressure_bore_skid2": 98.25,
    "pressure_rod_skid2": 55.5,
    "tower_x": 1.2345678901234,
    "tower_y": -3.5,
    "vessel_x": 100.25,
    "vessel_y": -250.5,
}


def alma_definition(order: str) -> dict:
    """Een telegram zoals het er op de BM010 uit kan zien: header, gemengde types, een bit en een controlewoord."""
    return {
        "byte_order": order,
        "packet_bytes": 52,
        "header_bytes": 4,
        "fields": [
            {"signal": "system_mode", "offset": 0, "type": "DWORD", "min": 0, "max": 255},
            {"signal": "pressure_bore_skid1", "offset": 4, "type": "FLOAT", "min": 0, "max": 400},
            {"signal": "pressure_rod_skid1", "offset": 8, "type": "INT", "scale": 0.01},
            {"signal": "pressure_bore_skid2", "offset": 12, "type": "FLOAT"},
            {"signal": "pressure_rod_skid2", "offset": 16, "type": "DINT", "scale": 0.001},
            {"signal": "tower_x", "offset": 20, "type": "DOUBLE"},
            {"signal": "tower_y", "offset": 28, "type": "DINT", "scale": 0.001},
            {"signal": "vessel_x", "offset": 32, "type": "FLOAT", "bias": 10.0},
            {"signal": "vessel_y", "offset": 36, "type": "LINT", "scale": 0.1},
        ],
        "checks": [{"offset": 44, "type": "DWORD", "equals": 0xCAFEBABE}],
        "timestamp": {"source": "receive"},
    }


@pytest.mark.parametrize("order", ORDERS)
def test_a_realistic_telegram_has_the_documented_layout_and_round_trips(order):
    fmt = PacketFormat.from_dict(alma_definition(order), required_signals=SIGNALS)
    # De indeling met de hand uitgeschreven: 4 header, DWORD, FLOAT, INT, 2 leeg, FLOAT, DINT, DOUBLE, DINT, FLOAT,
    # LINT, DWORD
    prefix = ">" if order == "big" else "<"
    expected = struct.pack(
        prefix + "4xIfh2xfidifqI", 3, 123.5, 4025, 98.25, 55500, 1.2345678901234, -3500, 90.25, -2505, 0xCAFEBABE
    )
    assert len(expected) == 52
    packet = fmt.encode(ALMA_VALUES)
    assert packet == expected
    decoded = fmt.decode(packet)
    assert list(decoded) == list(SIGNALS) and decoded == pytest.approx(ALMA_VALUES)
    assert fmt.encode(decoded) == packet


# ---- een ongeldig formaat ----------------------------------------------------------------------------


def valid() -> dict:
    """Een geldig formaat met een bitveld, een geschaald float-signaal, een DINT, een check en ruimte over."""
    return {
        "byte_order": "big",
        "packet_bytes": 20,
        "header_bytes": 2,
        "fields": [
            {"signal": "flag", "offset": 0, "type": "WORD", "bit": 3},
            {"signal": "level", "offset": 2, "type": "FLOAT", "scale": 2.0, "bias": 1.0, "min": -10, "max": 10},
            {"signal": "count", "offset": 6, "type": "DINT"},
        ],
        "checks": [{"offset": 10, "type": "WORD", "equals": 48879}],
        "timestamp": {"source": "receive"},
    }


def top(**changes):
    return lambda data: data.update(changes)


def without(key):
    return lambda data: data.pop(key)


def entry(index, **changes):
    return lambda data: data["fields"][index].update(changes)


def entry_without(index, key):
    return lambda data: data["fields"][index].pop(key)


def check_entry(index, **changes):
    return lambda data: data["checks"][index].update(changes)


def check_without(index, key):
    return lambda data: data["checks"][index].pop(key)


def add_field(**new):
    return lambda data: data["fields"].append(new)


def add_check(**new):
    return lambda data: data["checks"].append(new)


def replace(path, value):
    def edit(data):
        target = data
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value

    return edit


def chain(*edits):
    def edit(data):
        for each in edits:
            each(data)

    return edit


def family(edit_for, key, values, *fragments):
    """Een case per ongeldige waarde; de id is de sleutel met de (ingekorte) waarde."""
    return [pytest.param(edit_for(value), fragments, id=f"{key}={value!r}"[:40]) for value in values]


NOT_A_NUMBER = [True, False, "2", None, [], {}, math.nan, math.inf, -math.inf, 10**400]
NOT_A_COUNT = [-1, 1.0, 2.5, "2", True, None, [], {}]


def bad(edit, *fragments, id):
    return pytest.param(edit, fragments, id=id)


def check_values_outside(type_name, values, limit_type):
    """Check-waarden die niet in het type van de check passen."""

    def edit_for(value):
        return check_entry(0, type=type_name, equals=value)

    return family(edit_for, f"{type_name} equals", values, "checks[0]", "equals", limit_type)


INVALID = [
    # het bovenste niveau
    bad(top(colour=1), "colour", id="unknown key at the top"),
    bad(without("byte_order"), "byte_order", id="byte_order missing"),
    *family(
        lambda v: top(byte_order=v),
        "byte_order",
        ["middle", "Big", "BIG", "", 1, None, True, ["big"]],
        "byte_order",
    ),
    bad(without("packet_bytes"), "packet_bytes", id="packet_bytes missing"),
    *family(
        lambda v: top(packet_bytes=v),
        "packet_bytes",
        [0, -1, MAX_PACKET_BYTES + 1, "20", 20.0, True, None, [20]],
        "packet_bytes",
    ),
    *family(lambda v: top(header_bytes=v), "header_bytes", [-1, "2", 2.5, True, None, []], "header_bytes"),
    *family(lambda v: top(header_bytes=v), "header_bytes", [20, 21], "header_bytes", "packet_bytes"),
    bad(without("fields"), "fields", id="fields missing"),
    *family(lambda v: top(fields=v), "fields", [[], {}, "x", None, 5], "fields"),
    *family(lambda v: top(checks=v), "checks", [{}, "x", None, 5], "checks"),
    *family(lambda v: top(timestamp=v), "timestamp", ["receive", [], None, 5], "timestamp"),
    bad(top(timestamp={}), "timestamp", "source", id="timestamp without source"),
    *family(
        lambda v: top(timestamp={"source": v}),
        "timestamp.source",
        ["device", "RECEIVE", "", None, 1],
        "timestamp",
        "source",
    ),
    bad(top(timestamp={"source": "receive", "extra": 1}), "timestamp", "extra", id="unknown key in timestamp"),
    # velden
    *family(lambda v: replace(("fields", 1), v), "fields[1]", [5, "level", None, [], True], "fields[1]"),
    bad(entry(1, colour="red"), "fields[1]", "level", "colour", id="unknown key in a field"),
    bad(entry(0, colour="red"), "fields[0]", "flag", "colour", id="unknown key in a bit field"),
    bad(entry_without(1, "signal"), "fields[1]", "signal", id="signal missing"),
    *family(lambda v: entry(1, signal=v), "signal", [5, None, "", ["a"], True], "fields[1]", "signal"),
    bad(entry(2, signal="flag"), "flag", "fields[2]", "fields[0]", id="duplicate signal"),
    bad(entry_without(1, "offset"), "fields[1]", "level", "offset", id="offset missing"),
    *family(lambda v: entry(1, offset=v), "offset", NOT_A_COUNT, "fields[1]", "level", "offset"),
    bad(entry_without(1, "type"), "fields[1]", "level", "type", id="type missing"),
    *family(
        lambda v: entry(1, type=v),
        "type",
        [5, None, ["WORD"], {}, "", "FLOT", "uint64"],
        "fields[1]",
        "level",
        "type",
    ),
    bad(entry(1, offset=15), "fields[1]", "level", id="field one byte beyond the packet"),
    bad(entry(2, offset=15), "fields[2]", "count", id="dint beyond the packet"),
    bad(entry(1, offset=10**30), "fields[1]", "level", id="absurd offset"),
    # bitvelden
    bad(entry(1, bit=1), "fields[1]", "level", "bit", id="bit on a float"),
    bad(entry(1, type="DOUBLE", bit=1), "fields[1]", "level", "bit", id="bit on a double"),
    bad(add_field(signal="fbit", offset=12, type="FLOAT", bit=1), "fbit", "geheel type", id="bit on a bare float"),
    bad(add_field(signal="dbit", offset=12, type="float64", bit=0), "dbit", "geheel type", id="bit on a bare double"),
    *family(lambda v: entry(0, bit=v), "bit", [-1, 16, 100, 1.0, "3", True, None, []], "fields[0]", "flag", "bit"),
    bad(entry(0, type="BYTE", bit=8), "flag", "bit", id="bit 8 of a byte"),
    bad(entry(0, type="DWORD", bit=32), "flag", "bit", id="bit 32 of a dword"),
    bad(entry(0, type="LINT", bit=64), "flag", "bit", id="bit 64 of a lint"),
    *[bad(entry(0, **{key: 1}), "flag", key, id=f"bit field with {key}") for key in ("scale", "bias", "min", "max")],
    # schaal, bias, min en max
    *family(lambda v: entry(1, scale=v), "scale", [*NOT_A_NUMBER, 0, 0.0], "fields[1]", "level", "scale"),
    *family(lambda v: entry(1, bias=v), "bias", NOT_A_NUMBER, "fields[1]", "level", "bias"),
    *family(lambda v: entry(1, min=v), "min", NOT_A_NUMBER, "fields[1]", "level", "min"),
    *family(lambda v: entry(1, max=v), "max", NOT_A_NUMBER, "fields[1]", "level", "max"),
    bad(entry(1, min=5, max=3), "fields[1]", "level", "min", "max", id="min above max"),
    # overlap
    bad(entry(2, offset=4), "count", "level", id="two fields overlap"),
    bad(entry(2, offset=1), "count", "flag", id="a field overlaps a bit field"),
    bad(check_entry(0, offset=8), "checks[0]", "count", id="a check overlaps a field"),
    bad(add_check(offset=10, type="WORD", equals=1), "checks[1]", "checks[0]", id="identical checks"),
    bad(add_check(offset=11, type="WORD", equals=1), "checks[1]", "checks[0]", id="checks overlap partly"),
    bad(add_field(signal="other", offset=2, type="WORD", bit=0), "other", "level", id="a bit field overlaps a field"),
    bad(
        add_field(signal="b2", offset=0, type="BYTE", bit=1),
        "b2",
        "flag",
        id="bit fields on the same offset but another type",
    ),
    bad(
        add_field(signal="plain", offset=0, type="WORD"), "plain", "flag", id="a plain field on the word of a bit field"
    ),
    bad(add_check(offset=0, type="WORD", equals=1), "checks[1]", "flag", id="a check on the word of a bit field"),
    # checks
    *family(lambda v: replace(("checks", 0), v), "checks[0]", [5, "x", None, [], True], "checks[0]"),
    bad(check_entry(0, colour=1), "checks[0]", "colour", id="unknown key in a check"),
    *[bad(check_without(0, key), "checks[0]", key, id=f"check without {key}") for key in ("offset", "type", "equals")],
    *family(lambda v: check_entry(0, offset=v), "check offset", NOT_A_COUNT, "checks[0]", "offset"),
    *family(
        lambda v: check_entry(0, type=v),
        "check type",
        [5, None, ["WORD"], "", "FLOT", "uint64"],
        "checks[0]",
        "type",
    ),
    *family(
        lambda v: check_entry(0, type=v),
        "check type",
        ["FLOAT", "DOUBLE", "float32", "float64"],
        "checks[0]",
        "gehele",
    ),
    bad(check_entry(0, offset=17), "checks[0]", id="check one byte beyond the packet"),
    *family(
        lambda v: check_entry(0, equals=v),
        "equals",
        [True, False, 1.5, 2.0, "1", None, [], {}],
        "checks[0]",
        "equals",
    ),
    *family(
        lambda v: check_entry(0, equals=v), "equals", [65536, -1, 10**30, 123456], "checks[0]", "equals", "uint16"
    ),
    *check_values_outside("INT", [32768, -32769], "int16"),
    *check_values_outside("BYTE", [256, -1], "uint8"),
    *check_values_outside("SINT", [128, -129], "int8"),
    *check_values_outside("DWORD", [2**32, -1], "uint32"),
    *check_values_outside("DINT", [2**31, -(2**31) - 1], "int32"),
    *check_values_outside("LINT", [2**63, -(2**63) - 1], "int64"),
]


def distinctive_check_values(data) -> list[int]:
    """De check-waarden in de (vervormde) definitie die onmiskenbaar genoeg zijn om als tekst terug te zoeken."""
    checks = data.get("checks")
    found = []
    for check in checks if isinstance(checks, list) else []:
        value = check.get("equals") if isinstance(check, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 1000:
            found.append(abs(value))
    return found


@pytest.mark.parametrize("edit, fragments", INVALID)
def test_an_invalid_format_is_rejected_naming_the_signal_key_or_index(edit, fragments):
    data = valid()
    edit(data)
    message = format_error(data)
    for fragment in fragments:
        assert fragment in message, message
    for value in distinctive_check_values(data):  # de verwachte waarde van een check lekt nooit, ook niet als hex
        assert str(value) not in message and format(value, "x") not in message.lower(), message


def test_the_base_definition_used_for_the_invalid_cases_is_itself_valid():
    PacketFormat.from_dict(valid())
    assert MAX_PACKET_BYTES == 65507


@pytest.mark.parametrize("data", [[], [valid()], "text", None, 42, 1.5, True], ids=lambda data: type(data).__name__)
def test_the_top_level_must_be_a_json_object(data):
    assert "JSON-object" in format_error(data)


@pytest.mark.parametrize("value", [-(2**63), -1, 256, 2**63])
def test_a_rejected_check_value_is_never_echoed(value):
    data = valid()
    data["checks"][0]["type"] = "BYTE"
    data["checks"][0]["equals"] = value
    assert str(value) not in format_error(data)


@pytest.mark.parametrize("packet_bytes", [0, MAX_PACKET_BYTES + 1])
def test_packet_bytes_outside_the_udp_range_is_refused_with_the_range_in_the_message(packet_bytes):
    data = valid()
    data["packet_bytes"] = packet_bytes
    assert "1 tot en met 65507" in format_error(data)


def test_messages_stay_ascii_for_names_with_other_characters():
    data = valid()
    data["fields"][1].update(signal="druk\xe9\u4e2d", colour="x")
    data["caf\xe9"] = 1
    assert format_error(data).isascii() and "caf\\xe9" in format_error(data)  # de onbekende sleutel
    del data["caf\xe9"]
    message = format_error(data)
    assert message.isascii() and "druk\\xe9\\u4e2d" in message  # het signaal waar de fout in zit

    fmt = single("big", "int8", signal="druk\xe9", max=0)
    with pytest.raises(PacketError) as decode_error:
        fmt.decode(b"\x05")
    assert decode_error.value.args[0].isascii() and "druk\\xe9" in str(decode_error.value)
    for values in ({}, {"druk\xe9": math.nan}, {"druk\xe9": "5"}, {"druk\xe9": 1000}, {"druk\xe9": 0, "\xe9x": 1}):
        with pytest.raises(ValueError) as encode_error:
            fmt.encode(values)
        assert str(encode_error.value).isascii() and "\\xe9" in str(encode_error.value)


def test_a_very_long_key_or_type_name_is_shortened_in_the_message():
    data = valid()
    data["k" * 5000] = 1
    assert len(format_error(data)) < 500 and "..." in format_error(data)
    data = valid()
    data["fields"][1]["type"] = "T" * 5000
    assert len(format_error(data)) < 500


VALID_VARIANTS = [
    pytest.param(lambda data: None, id="the base definition"),
    pytest.param(without("header_bytes"), id="no header_bytes"),
    pytest.param(without("checks"), id="no checks"),
    pytest.param(top(checks=[]), id="empty checks"),
    pytest.param(without("timestamp"), id="no timestamp"),
    pytest.param(entry(1, offset=14), id="a field that ends exactly at the end of the packet"),
    pytest.param(top(packet_bytes=MAX_PACKET_BYTES), id="the largest udp payload"),
    pytest.param(entry(0, bit=0), id="bit 0 of a word"),
    pytest.param(entry(0, bit=15), id="bit 15 of a word"),
    pytest.param(add_field(signal="hi", offset=12, type="BYTE", bit=7), id="bit 7 of a byte"),
    pytest.param(add_field(signal="top", offset=12, type="DWORD", bit=31), id="bit 31 of a dword"),
    pytest.param(
        chain(top(packet_bytes=30), add_field(signal="top", offset=12, type="LINT", bit=63)), id="bit 63 of a lint"
    ),
    pytest.param(
        add_field(signal="flag2", offset=0, type="uint16", bit=0), id="two bit fields in a word, alias and canonical"
    ),
    pytest.param(
        chain(
            add_field(signal="flag2", offset=0, type="WORD", bit=4),
            add_field(signal="flag3", offset=0, type="WORD", bit=5),
        ),
        id="three bit fields in a word",
    ),
    pytest.param(entry(1, min=3, max=3), id="min equal to max"),
    pytest.param(entry(1, scale=-1), id="negative scale"),
    pytest.param(entry(1, scale=2, bias=1, min=-5, max=5), id="integers for scale, bias, min and max"),
    pytest.param(entry(1, scale=1e-300), id="tiny scale"),
    pytest.param(entry(1, scale=1e300), id="huge scale"),
    pytest.param(entry(1, signal="Druk bore [bar]"), id="any non-empty text as signal name"),
    pytest.param(lambda data: data["fields"].reverse(), id="fields not in offset order"),
    pytest.param(add_check(offset=12, type="WORD", equals=1), id="a second check right behind the first"),
    pytest.param(
        chain(top(packet_bytes=30), add_check(offset=12, type="INT", equals=-1)), id="a check with a negative value"
    ),
]


@pytest.mark.parametrize("edit", VALID_VARIANTS)
def test_valid_definitions_that_look_like_edge_cases_are_accepted(edit):
    data = valid()
    edit(data)
    assert PacketFormat.from_dict(data).signals == tuple(f["signal"] for f in data["fields"])


def test_a_one_byte_telegram_and_a_header_that_leaves_one_byte_work():
    assert single("big", "uint8").decode(b"\x07") == {"x": 7.0}
    fmt = PacketFormat.from_dict(
        {
            "byte_order": "little",
            "packet_bytes": MAX_PACKET_BYTES,
            "header_bytes": MAX_PACKET_BYTES - 1,
            "fields": [{"signal": "x", "offset": 0, "type": "uint8"}],
        }
    )
    assert fmt.decode(bytes(MAX_PACKET_BYTES - 1) + b"\x07") == {"x": 7.0}


# ---- wat from_dict teruggeeft ------------------------------------------------------------------------


def test_from_dict_exposes_the_definition():
    fmt = PacketFormat.from_dict(valid())
    assert (fmt.byte_order, fmt.packet_bytes, fmt.header_bytes) == ("big", 20, 2)
    assert fmt.signals == ("flag", "level", "count")
    assert fmt.fields == (
        Field("flag", 0, "uint16", 3, 1.0, 0.0, None, None),
        Field("level", 2, "float32", None, 2.0, 1.0, -10.0, 10.0),
        Field("count", 6, "int32", None, 1.0, 0.0, None, None),
    )
    assert fmt.checks == (Check(10, "uint16", 48879),)
    assert isinstance(fmt.fields, tuple) and isinstance(fmt.checks, tuple) and isinstance(fmt.signals, tuple)


def test_optional_parts_get_their_defaults():
    fmt = single("little", "BYTE")
    assert (fmt.header_bytes, fmt.checks) == (0, ())
    assert fmt.fields == (Field("x", 0, "uint8"),)  # bit, min en max zijn None; scale 1 en bias 0


def test_a_format_built_directly_gets_the_same_layout_checks():
    word = Field("x", 0, "uint16")
    assert PacketFormat("big", 2, 0, [word]).fields == (word,)  # een lijst wordt een tuple
    for args, message in [
        (("middle", 2, 0, (word,)), "byte_order"),
        (("big", 2, 0, ()), "fields"),
        (("big", 3, 0, (word, Field("y", 1, "uint16"))), "overlapt"),
        (("big", 2, 0, (word, Field("x", 0, "uint16"))), "komt al voor"),
        (("big", 2, 0, (Field("y", 1, "uint16"),)), "passen niet"),
        (("big", 2, 0, (word,), (Check(0, "uint16", 1),)), "overlapt"),
    ]:
        with pytest.raises(FormatError, match=message):
            PacketFormat(*args)


def test_numbers_in_the_definition_are_stored_as_floats():
    data = valid()
    data["fields"][1].update(scale=2, bias=1, min=-10, max=10)
    level = PacketFormat.from_dict(data).fields[1]
    assert all(type(v) is float for v in (level.scale, level.bias, level.min, level.max))


def test_signals_follow_the_order_of_the_json_not_the_offsets():
    data = valid()
    data["fields"].reverse()
    assert PacketFormat.from_dict(data).signals == ("count", "level", "flag")


def test_from_dict_neither_changes_its_input_nor_keeps_a_reference_to_it():
    data = valid()
    before = copy.deepcopy(data)
    fmt = PacketFormat.from_dict(data)
    assert data == before
    data["fields"][0]["offset"] = 99
    data["fields"].clear()
    assert fmt.fields[0].offset == 0 and len(fmt.fields) == 3


def test_definitions_are_frozen():
    fmt = PacketFormat.from_dict(valid())
    changes = [(fmt, "byte_order", "little"), (fmt.fields[0], "offset", 5), (fmt.checks[0], "equals", 1)]
    for target, name, value in changes:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, name, value)


def test_a_format_survives_copy_deepcopy_and_pickle():
    fmt = PacketFormat.from_dict(alma_definition("little"))
    packet = fmt.encode(ALMA_VALUES)
    for clone in (copy.copy(fmt), copy.deepcopy(fmt), pickle.loads(pickle.dumps(fmt))):
        assert clone == fmt and clone.decode(packet) == fmt.decode(packet)


def test_formats_with_the_same_definition_are_equal_and_hashable():
    a, b = PacketFormat.from_dict(valid()), PacketFormat.from_dict(valid())
    assert a == b and hash(a) == hash(b)
    other = valid()
    other["byte_order"] = "little"
    assert a != PacketFormat.from_dict(other)


def test_the_example_in_the_module_docstring_is_a_valid_format():
    doc = format_module.__doc__
    example = json.loads(doc[doc.index("{") : doc.rindex("}") + 1])
    fmt = PacketFormat.from_dict(example)
    values = {"compensation_active": 1.0, "pressure_bore": 3.5, "tower_x": 12.3}
    assert fmt.decode(fmt.encode(values)) == pytest.approx(values)


# ---- required_signals ----------------------------------------------------------------------------------


def test_required_signals_must_equal_the_signals_in_fields_in_any_order_and_container():
    names = ["flag", "level", "count"]
    for required in (names, tuple(reversed(names)), set(names), frozenset(names), iter(names), dict.fromkeys(names)):
        assert PacketFormat.from_dict(valid(), required_signals=required).signals == ("flag", "level", "count")


def test_required_signals_lists_the_missing_and_the_unknown_signals():
    message = format_error(valid(), required_signals=["flag", "level", "pressure_bore", "pressure_rod"])
    assert "ontbrekend: pressure_bore, pressure_rod" in message
    assert "onbekend: count" in message
    assert "flag" not in message and "level" not in message  # wat wel klopt wordt niet genoemd


def test_required_signals_with_only_missing_or_only_unknown_ones():
    only_missing = format_error(valid(), required_signals=["flag", "level", "count", "tower_x"])
    assert "ontbrekend: tower_x" in only_missing and "onbekend" not in only_missing
    only_unknown = format_error(valid(), required_signals=["flag", "level"])
    assert "onbekend: count" in only_unknown and "ontbrekend" not in only_unknown


def test_an_empty_required_set_makes_every_signal_unknown():
    assert "onbekend: count, flag, level" in format_error(valid(), required_signals=[])


def test_without_required_signals_any_signal_names_are_fine():
    assert PacketFormat.from_dict(valid(), required_signals=None).signals == ("flag", "level", "count")


def test_required_signals_for_the_alma_signals():
    assert PacketFormat.from_dict(alma_definition("big"), required_signals=SIGNALS).signals == tuple(SIGNALS)
    data = alma_definition("big")
    data["fields"][1]["signal"] = "pressure_bore_skid"  # een typefout
    message = format_error(data, required_signals=SIGNALS)
    assert "ontbrekend: pressure_bore_skid1" in message and "onbekend: pressure_bore_skid" in message
    del data["fields"][1]
    assert "ontbrekend: pressure_bore_skid1" in format_error(data, required_signals=SIGNALS)


# ---- load ------------------------------------------------------------------------------------------------


def test_load_reads_a_json_file_with_or_without_a_bom(tmp_path):
    text = json.dumps(valid())
    plain, with_bom = tmp_path / "plain.json", tmp_path / "bom.json"
    plain.write_text(text, encoding="utf-8")
    with_bom.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))  # Kladblok zet er een BOM voor
    expected = PacketFormat.from_dict(valid())
    assert PacketFormat.load(plain) == expected
    assert PacketFormat.load(with_bom) == expected
    assert PacketFormat.load(str(plain)) == expected  # ook een gewone tekst als pad


def test_load_passes_required_signals_on(tmp_path):
    path = tmp_path / "format.json"
    path.write_text(json.dumps(valid()), encoding="utf-8")
    assert PacketFormat.load(path, required_signals=["flag", "level", "count"]).signals == ("flag", "level", "count")
    with pytest.raises(FormatError, match="tower_x"):
        PacketFormat.load(path, required_signals=["tower_x"])


def test_load_reports_a_missing_file_by_its_path(tmp_path):
    path = tmp_path / "does not exist" / "format.json"
    with pytest.raises(FormatError) as caught:
        PacketFormat.load(path)
    assert str(path) in str(caught.value)


def test_load_reports_a_directory_by_its_path(tmp_path):
    with pytest.raises(FormatError) as caught:
        PacketFormat.load(tmp_path)
    assert str(tmp_path) in str(caught.value)


def test_load_reports_a_path_the_operating_system_refuses(tmp_path):
    with pytest.raises(FormatError, match="niet te lezen"):
        PacketFormat.load(str(tmp_path / "format\0.json"))  # een NUL in het pad is een ValueError, geen OSError


@pytest.mark.parametrize(
    "content",
    [
        b'{"ApiKey": "geheim-123", oops',
        b"",
        b"   \n",
        b"not json at all geheim-123",
        b'{"a": 1} geheim-123',
        b"[" * 100_000,  # absurde nesting
    ],
    ids=["truncated", "empty", "blank", "text", "trailing data", "deep nesting"],
)
def test_load_reports_invalid_json_by_its_path_without_echoing_the_contents(tmp_path, content):
    path = tmp_path / "format.json"
    path.write_bytes(content)
    with pytest.raises(FormatError) as caught:
        PacketFormat.load(path)
    message = str(caught.value)
    assert str(path) in message and "JSON" in message
    assert "geheim-123" not in message and "oops" not in message


def test_load_reports_a_number_of_thousands_of_digits_as_a_format_error(tmp_path):
    path = tmp_path / "format.json"
    path.write_bytes(b'{"packet_bytes": ' + b"9" * 5000 + b"}")
    with pytest.raises(FormatError) as caught:
        PacketFormat.load(path)
    assert str(path) in str(caught.value)


def test_load_reports_a_file_that_is_not_utf8(tmp_path):
    path = tmp_path / "format.json"
    path.write_bytes(json.dumps(valid()).encode("utf-16"))  # "Unicode" in Kladblok
    with pytest.raises(FormatError, match="UTF-8") as caught:
        PacketFormat.load(path)
    assert str(path) in str(caught.value)


def test_load_names_the_file_when_the_format_itself_is_wrong(tmp_path):
    path = tmp_path / "format.json"
    data = valid()
    data["fields"][1]["scale"] = 0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FormatError) as caught:
        PacketFormat.load(path)
    assert str(path) in str(caught.value) and "level" in str(caught.value) and "scale" in str(caught.value)


def test_load_reports_a_wrong_file_by_key_name_without_echoing_its_values(tmp_path):
    path = tmp_path / "quick_connect.json"  # per ongeluk het verkeerde bestand
    secrets = {"Host": "ibahd.example", "Port": 61937, "ApiKey": "geheim-123", "TlsServerCertificate": "pem-data"}
    path.write_text(json.dumps(secrets), encoding="utf-8")
    with pytest.raises(FormatError) as caught:
        PacketFormat.load(path)
    message = str(caught.value)
    assert "Host" in message
    assert all(secret not in message for secret in ("ibahd.example", "61937", "geheim-123", "pem-data"))


def test_load_refuses_a_key_that_occurs_twice(tmp_path):
    path = tmp_path / "format.json"
    path.write_text(
        '{"byte_order": "big", "byte_order": "little", "packet_bytes": 2,'
        ' "fields": [{"signal": "x", "offset": 0, "type": "WORD"}]}',
        encoding="utf-8",
    )
    with pytest.raises(FormatError, match="byte_order") as caught:  # stil de laatste houden draait de byte-volgorde om
        PacketFormat.load(path)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_load_refuses_numbers_that_python_reads_from_json_but_are_not_finite(tmp_path, number):
    path = tmp_path / "format.json"
    path.write_text(
        '{"byte_order": "big", "packet_bytes": 4, "fields": [{"signal": "x", "offset": 0, "type": "FLOAT", "bias": '
        + number
        + "}]}",
        encoding="utf-8",
    )
    with pytest.raises(FormatError, match="bias"):
        PacketFormat.load(path)


# ---- decode kan niet worden gecrasht --------------------------------------------------------------------


def every_type_format(order: str) -> PacketFormat:
    """Alle typen, schaal, grenzen en bitvelden door elkaar, zodat een willekeurig telegram op veel manieren afvalt."""
    fields = [
        {"signal": "a", "offset": 0, "type": "uint8"},
        {"signal": "b", "offset": 1, "type": "int8", "scale": -2.0, "bias": 5.0},
        {"signal": "c", "offset": 2, "type": "uint16", "scale": 0.5, "min": 100, "max": 30000},
        {"signal": "d", "offset": 4, "type": "int16", "bias": -1.0},
        {"signal": "e", "offset": 6, "type": "uint32", "scale": 1e-3},
        {"signal": "f", "offset": 10, "type": "int32"},
        {"signal": "g", "offset": 14, "type": "int64", "scale": 0.5, "bias": 1.0},
        {"signal": "h", "offset": 22, "type": "float32", "min": -1e30, "max": 1e30},
        {"signal": "i", "offset": 26, "type": "float64", "scale": 2.0},
        {"signal": "bit_a", "offset": 34, "type": "WORD", "bit": 0},
        {"signal": "bit_b", "offset": 34, "type": "WORD", "bit": 15},
        {"signal": "bit_c", "offset": 36, "type": "DWORD", "bit": 31},
        {"signal": "bit_d", "offset": 40, "type": "BYTE", "bit": 7},
    ]
    # 3 byte header, 41 bytes met velden en 1 leeg byte aan het eind
    return make(order, *fields, packet_bytes=3 + 41 + 1, header_bytes=3)


@pytest.mark.parametrize("order", ORDERS)
def test_decode_of_random_bytes_gives_a_result_or_a_packet_error_and_nothing_else(order):
    fmt = every_type_format(order)
    rng = random.Random(f"fuzz/{order}")
    accepted = rejected = 0
    for _ in range(6000):
        try:
            result = fmt.decode(rng.randbytes(fmt.packet_bytes))
        except PacketError:
            rejected += 1
            continue
        accepted += 1
        assert list(result) == list(fmt.signals)
        assert all(type(v) is float and math.isfinite(v) for v in result.values())
    assert accepted > 100 and rejected > 100  # beide uitkomsten zijn echt geraakt


@pytest.mark.parametrize("order", ORDERS)
def test_decode_of_every_wrong_length_is_a_packet_error(order):
    fmt = every_type_format(order)
    rng = random.Random(f"lengths/{order}")
    lengths = [n for n in range(0, 3 * fmt.packet_bytes) if n != fmt.packet_bytes] + [1_000, 65_507, 70_000, 5_000_000]
    for length in lengths:
        with pytest.raises(PacketError) as caught:
            fmt.decode(rng.randbytes(length) if length < 1_000_000 else bytes(length))
        assert str(length) in str(caught.value) and str(fmt.packet_bytes) in str(caught.value)


def layout_format(order: str) -> PacketFormat:
    """2 bytes header, dan: check (2), count, leeg, level (2), flag (alleen bit 7), leeg, depth (4)."""
    return PacketFormat.from_dict(
        {
            "byte_order": order,
            "packet_bytes": 14,
            "header_bytes": 2,
            "fields": [
                {"signal": "count", "offset": 2, "type": "BYTE"},
                {"signal": "level", "offset": 4, "type": "INT", "scale": 0.5, "min": -100, "max": 100},
                {"signal": "flag", "offset": 6, "type": "BYTE", "bit": 7},
                {"signal": "depth", "offset": 8, "type": "FLOAT"},
            ],
            "checks": [{"offset": 0, "type": "WORD", "equals": 0xC0DE}],
        }
    )


@pytest.mark.parametrize("order", ORDERS)
def test_changing_any_single_byte_of_a_valid_telegram_is_handled_as_documented(order):
    fmt = layout_format(order)
    good = fmt.encode({"count": 7, "level": 12.5, "flag": 1, "depth": 1.25})
    original = fmt.decode(good)
    assert original == {"count": 7.0, "level": 12.5, "flag": 1.0, "depth": 1.25}
    unused = {0, 1, 5, 9}  # de header en de twee lege bytes
    check_bytes, flag_byte = {2, 3}, 8  # het controlewoord, en het byte waarvan alleen bit 7 telt
    for position in range(len(good)):
        for new in range(256):
            changed = bytearray(good)
            changed[position] = new
            try:
                result = fmt.decode(bytes(changed))
            except PacketError:
                assert position not in unused and position != flag_byte and new != good[position]
                continue
            assert list(result) == list(fmt.signals)
            if position in unused:
                assert result == original  # wat nergens in zit, doet niets
            if position in check_bytes:
                assert new == good[position]  # een ander controlewoord komt er nooit door
            if position == flag_byte:  # alleen bit 7 telt
                assert result == {**original, "flag": float(new & 0x80 != 0)}


@pytest.mark.parametrize("order", ORDERS)
def test_a_non_finite_value_never_gets_through(order):
    fmt = layout_format(order)
    good = fmt.encode({"count": 1, "level": 1.0, "flag": 0, "depth": 1.0})
    for hexbytes in ("7fc00000", "7f800000", "ff800000", "7fffffff", "ffc00001"):
        with pytest.raises(PacketError, match="depth"):
            fmt.decode(good[:10] + wire(hexbytes, order))


def test_decode_does_not_compile_or_parse_anything_per_telegram(monkeypatch):
    fmt = layout_format("big")
    good = fmt.encode({"count": 7, "level": 12.5, "flag": 1, "depth": 1.25})

    def forbidden(*args, **kwargs):
        raise AssertionError("decode() mag dit niet voor elk telegram doen")

    with monkeypatch.context() as patch:  # struct-formaten, JSON en reguliere expressies horen bij het laden
        for module, name in [
            (struct, "Struct"),
            (struct, "unpack"),
            (struct, "unpack_from"),
            (struct, "calcsize"),
            (json, "loads"),
            (re, "compile"),
            (re, "match"),
            (re, "search"),
        ]:
            patch.setattr(module, name, forbidden)
        assert fmt.decode(good)["count"] == 7.0
        for bad in (good[:-1], b"", None, good[:2] + b"\x00\x00" + good[4:]):
            try:
                fmt.decode(bad)
            except PacketError:
                pass
            else:
                raise AssertionError("dit telegram had geweigerd moeten worden")


# ---- from_dict kan niet worden gecrasht -------------------------------------------------------------------

JUNK = [
    *(None, True, False, 0, -1, 1, 2, 7, 15, 16, 255, 2**31, 2**70, -(2**70), 10**400),  # getallen
    *(0.0, -0.0, 1.5, 1e308, math.nan, math.inf, -math.inf),
    *("", "x", "big", "little", "WORD", "float", "receive"),  # tekst
    *([], {}, [1], ["a"], {"a": 1}, {"source": "receive"}, (1, 2), b"bytes"),  # samengesteld
]


def paths(node, prefix=()):
    yield prefix
    if isinstance(node, dict):
        for key, value in node.items():
            yield from paths(value, (*prefix, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from paths(value, (*prefix, index))


def at(node, path):
    for step in path:
        node = node[step]
    return node


def vandalise(rng: random.Random, data: dict) -> None:
    """Een paar willekeurige ingrepen: een waarde vervangen, een sleutel weghalen of toevoegen, iets invoegen."""
    for _ in range(rng.choice([1, 1, 2, 3])):
        path = rng.choice([p for p in paths(data) if p])
        parent = at(data, path[:-1])
        action = rng.random()
        if action < 0.6:
            parent[path[-1]] = copy.deepcopy(rng.choice(JUNK) if rng.random() < 0.9 else valid()["fields"][0])
        elif action < 0.8 and isinstance(parent, dict):
            del parent[path[-1]]
        elif action < 0.9 and isinstance(parent, dict):
            key = rng.choice(["extra", "bit", "scale", "min", "max", "equals", "signal", 5, None])
            parent[key] = copy.deepcopy(rng.choice(JUNK))
        elif isinstance(parent, list):
            parent.insert(rng.randrange(len(parent) + 1), copy.deepcopy(rng.choice(JUNK)))


def test_from_dict_of_vandalised_definitions_gives_a_usable_format_or_a_format_error_and_nothing_else():
    rng = random.Random("from_dict fuzz")
    usable = refused = 0
    for _ in range(4000):
        data = valid()
        vandalise(rng, data)
        try:
            fmt = PacketFormat.from_dict(data)
        except FormatError:
            refused += 1
            continue
        usable += 1
        try:  # een formaat dat het laden overleeft moet ook werken
            fmt.decode(bytes(fmt.packet_bytes))
        except PacketError:
            pass
        assert fmt.signals == tuple(f.signal for f in fmt.fields)
    assert usable > 100 and refused > 100
