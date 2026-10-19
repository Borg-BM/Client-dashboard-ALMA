"""Settings of the UDP source: safe by default, and unsafe only deliberately."""

import pytest
from pydantic import ValidationError

from alma.config import Settings
from alma.sources import create_source


def make(**kwargs):
    return Settings(_env_file=None, auth_enabled=False, **kwargs)


def test_udp_source_requires_a_source_filter():
    with pytest.raises(ValidationError, match="ALMA_UDP_ALLOWED_SOURCES"):
        make(source="udp")


def test_udp_source_with_an_allowlist_or_the_explicit_opt_out():
    assert make(source="udp", udp_allowed_sources=["10.0.0.5"]).udp_allowed_sources == ["10.0.0.5"]
    assert make(source="udp", udp_allow_any_source=True).udp_allow_any_source is True


def test_allowlist_and_allow_any_exclude_each_other():
    with pytest.raises(ValidationError, match="not both"):
        make(source="udp", udp_allowed_sources=["10.0.0.5"], udp_allow_any_source=True)


def test_other_sources_do_not_need_a_filter():
    assert make(source="mock").udp_allowed_sources == []


def test_allowlist_comes_from_a_comma_separated_variable(monkeypatch):
    monkeypatch.setenv("ALMA_UDP_ALLOWED_SOURCES", "10.0.0.5, 10.0.1.0/24 ,fe80::/10")
    assert make(source="udp").udp_allowed_sources == ["10.0.0.5", "10.0.1.0/24", "fe80::/10"]


@pytest.mark.parametrize("entry", ["not-an-ip", "10.0.0.256", "10.0.0.0/33", "10.0.0.5:5005"])
def test_allowlist_rejects_nonsense_even_when_the_source_is_not_udp(entry):
    with pytest.raises(ValidationError, match="not an IP address or network"):
        make(source="mock", udp_allowed_sources=[entry])


def test_bind_must_be_an_ip_address():
    assert make(udp_bind="127.0.0.1").udp_bind == "127.0.0.1"
    with pytest.raises(ValidationError, match="ALMA_UDP_BIND"):
        make(udp_bind="localhost")


@pytest.mark.parametrize("port", [-1, 65536])
def test_port_must_be_valid(port):
    with pytest.raises(ValidationError):
        make(udp_port=port)


@pytest.mark.parametrize("minutes", [0, 61])
def test_cache_minutes_stay_within_what_the_api_serves(minutes):
    with pytest.raises(ValidationError):
        make(cache_minutes=minutes)


def test_defaults_are_conservative():
    settings = make()
    assert settings.udp_port == 5005 and settings.cache_minutes == 10
    assert settings.udp_allow_any_source is False and settings.udp_allowed_sources == []


def test_unknown_source_names_the_choices():
    with pytest.raises(ValueError, match="udp"):
        create_source(make(source="nonsense"))
