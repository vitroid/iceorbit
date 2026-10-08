import numpy as np
import pytest

from iceorbit import automorphism_group, builtin_graph, classify, enumerate_ice_configs
from iceorbit.polarization import (
    ZERO_TOL,
    bond_directions,
    magnitude_distribution,
    polarization,
    polarization_direct,
)


@pytest.fixture(scope="module")
def dodeca():
    graph = builtin_graph("dodecahedron")
    configs = enumerate_ice_configs(graph)
    res = classify(configs, automorphism_group(graph))
    mag = np.linalg.norm(polarization(configs, graph), axis=1)
    return graph, configs, res, mag


@pytest.mark.parametrize("name", ["dodecahedron", "cube"])
def test_outer_bond_is_radial_for_regular_polyhedra(name):
    graph = builtin_graph(name)
    _, outer = bond_directions(graph)
    radial = graph.coords / np.linalg.norm(graph.coords, axis=1, keepdims=True)
    assert np.allclose(outer, radial)


def test_cube_linear_form_matches_direct():
    graph = builtin_graph("cube")
    configs = enumerate_ice_configs(graph)
    p = polarization(configs, graph)
    want = np.array([polarization_direct(int(x), graph) for x in configs])
    assert np.allclose(p, want)


def test_dodecahedron_linear_form_matches_direct(dodeca):
    graph, configs, _, _ = dodeca
    rng = np.random.default_rng(0)
    idx = rng.choice(len(configs), 200, replace=False)
    p = polarization(configs[idx], graph)
    want = np.array([polarization_direct(int(configs[i]), graph) for i in idx])
    assert np.allclose(p, want)


def test_magnitude_is_constant_within_a_class(dodeca):
    _, _, res, mag = dodeca
    lo = np.full(res.n_classes, np.inf)
    hi = np.zeros(res.n_classes)
    np.minimum.at(lo, res.class_id, mag)
    np.maximum.at(hi, res.class_id, mag)
    assert np.max(hi - lo) < 1e-10


def test_inversion_symmetric_classes_are_unpolarized(dodeca):
    _, _, res, mag = dodeca
    class_mag = np.zeros(res.n_classes)
    class_mag[res.class_id] = mag
    assert res.stabilizer_improper.sum() == 36
    assert np.all(class_mag[res.stabilizer_improper] < ZERO_TOL)


def test_dodecahedron_distribution(dodeca):
    _, configs, res, mag = dodeca
    dist = magnitude_distribution(mag, res.class_id)
    assert dist.n_configs.sum() == len(configs) == 3_600_000
    assert dist.n_classes.sum() == res.n_classes
    assert dist.values[0] == 0
    assert dist.n_configs[0] == 18_576
    assert dist.n_classes[0] == 176
    assert dist.values[1] > 1.0  # zero is well separated from the smallest nonzero |P|
