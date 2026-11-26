import numpy as np
import pytest

from bss_sim.config import ConfigError, SimConfig
from bss_sim.rng import Streams


def test_defaults_follow_the_spec():
    c = SimConfig()
    assert c.mode == "dockless"
    assert c.fleet.size == 800
    assert (c.users.walk_beta_m, c.users.walk_radius_m, c.users.walking_speed_mps) == (150.0, 400.0, 1.4)
    assert c.energy.dead_ebike_rideable is False


def test_dict_round_trip_and_hash():
    c = SimConfig.from_dict({"mode": "docked", "fleet": {"size": 300, "initial_battery": [0.5, 0.9]},
                             "energy": {"battery_bands": [0.1, 0.4, 0.8]}})
    assert c.fleet.initial_battery == (0.5, 0.9)
    assert c.energy.battery_bands == (0.1, 0.4, 0.8)
    again = SimConfig.from_dict(c.to_dict())
    assert again == c
    assert again.config_hash() == c.config_hash()
    assert c.config_hash() != SimConfig().config_hash()


def test_ints_accepted_for_floats_and_replace():
    c = SimConfig.from_dict({"users": {"walk_beta_m": 200}})
    assert c.users.walk_beta_m == 200.0 and isinstance(c.users.walk_beta_m, float)
    d = c.replace(**{"users.walk_radius_m": 300, "mode": "docked"})
    assert d.users.walk_radius_m == 300.0 and d.mode == "docked" and d.users.walk_beta_m == 200.0


@pytest.mark.parametrize("data, msg", [
    ({"fleet": {"sise": 3}}, "fleet.sise"),
    ({"mode": "teleport"}, "mode"),
    ({"fleet": {"size": -1}}, "fleet.size"),
    ({"fleet": {"size": 1.5}}, "fleet.size"),
    ({"fleet": {"initial_battery": [0.9, 0.2]}}, "initial_battery"),
    ({"energy": {"battery_bands": [0.5, 0.2]}}, "battery_bands"),
    ({"users": {"walk_beta_m": 0}}, "walk_beta_m"),
    ({"debug": "yes"}, "debug"),
    ({"users": 3}, "users"),
])
def test_invalid_values_rejected(data, msg):
    with pytest.raises(ConfigError, match=msg):
        SimConfig.from_dict(data)


def test_from_yaml(tmp_path):
    p = tmp_path / "sim.yaml"
    p.write_text("mode: docked\nexternal:\n  reservoir: 10\n")
    c = SimConfig.from_yaml(p)
    assert c.mode == "docked" and c.external.reservoir == 10


def test_streams_are_independent_of_creation_order():
    a, b = Streams(5), Streams(5)
    b.generator("forecast").random(100)  # using another stream first changes nothing
    np.testing.assert_array_equal(a.generator("users").random(10), b.generator("users").random(10))
    assert a.int_seed("demand") == b.int_seed("demand")
    assert not np.array_equal(a.generator("users").random(10), a.generator("fleet").random(10))
    assert not np.array_equal(Streams(5).generator("users").random(10), Streams(6).generator("users").random(10))
    with pytest.raises(ValueError):
        Streams(-1)
