import itertools

import numpy as np
import pytest

from iceorbit import automorphism_group, builtin_graph, graph_from_xyz, classify, enumerate_ice_configs
from iceorbit.canon import act


def brute_force_configs(graph):
    inc = graph.incident_edges()
    out = []
    for x in range(1 << graph.n_edges):
        ok = True
        for v, es in enumerate(inc):
            n_in = sum(((x >> f) & 1) ^ (1 if graph.edges[f, 1] == v else 0) for f in es)
            if n_in not in (1, 2):
                ok = False
                break
        if ok:
            out.append(x)
    return out


def act_vertexwise(graph, sigma, x):
    """Reference action: move each directed bond (donor, acceptor) by sigma."""
    index = {(int(u), int(v)): e for e, (u, v) in enumerate(graph.edges)}
    y = 0
    for e, (u, v) in enumerate(graph.edges):
        d, a = (u, v) if not (x >> e) & 1 else (v, u)
        d, a = sigma[d], sigma[a]
        e2 = index[(min(d, a), max(d, a))]
        if d > a:
            y |= 1 << e2
    return y


@pytest.fixture(scope="module")
def dodeca():
    graph = builtin_graph("dodecahedron")
    group = automorphism_group(graph)
    configs = enumerate_ice_configs(graph)
    return graph, group, configs


def test_cube_matches_brute_force():
    graph = builtin_graph("cube")
    group = automorphism_group(graph)
    assert group.order == 48 and group.proper.sum() == 24

    ref = brute_force_configs(graph)
    configs = enumerate_ice_configs(graph)
    assert configs.tolist() == ref

    for grp in (group, group.rotations()):
        orbits = {frozenset(act_vertexwise(graph, s, x) for s in grp.vertex_perm) for x in ref}
        res = classify(configs, grp)
        assert res.n_classes == len(orbits)
        assert sorted(res.degeneracy.tolist()) == sorted(len(o) for o in orbits)
        assert set(res.representatives.tolist()) == {min(o) for o in orbits}


def test_dodecahedron_group(dodeca):
    graph, group, _ = dodeca
    assert graph.n_vertices == 20 and graph.n_edges == 30
    assert group.order == 120
    assert group.proper.sum() == 60
    # vertex permutations are distinct and form a group under composition
    perms = {tuple(p) for p in group.vertex_perm.tolist()}
    assert len(perms) == 120
    for p, q in itertools.islice(itertools.product(perms, repeat=2), 0, 120 * 120, 97):
        assert tuple(p[i] for i in q) in perms


def test_bit_action_matches_vertex_action(dodeca):
    graph, group, configs = dodeca
    rng = np.random.default_rng(0)
    sample = configs[rng.choice(len(configs), 50, replace=False)]
    for g in range(0, group.order, 7):
        got = act(group, g, sample).tolist()
        want = [act_vertexwise(graph, group.vertex_perm[g], int(x)) for x in sample]
        assert got == want


def test_dodecahedron_counts(dodeca):
    _, group, configs = dodeca
    assert len(configs) == 3_600_000
    assert len(np.unique(configs)) == len(configs)

    res = classify(configs, group)
    assert res.n_classes == 30_026
    assert res.burnside_classes() == res.n_classes
    assert res.degeneracy.sum() == len(configs)
    assert np.all(res.degeneracy * res.stabilizer_order == group.order)

    rot = classify(configs, group.rotations())
    assert rot.n_classes == 60_016
    assert rot.burnside_classes() == rot.n_classes
    # each achiral Ih class stays one I class; each chiral one splits into an enantiomer pair
    n_achiral = int(res.stabilizer_improper.sum())
    assert rot.n_classes == 2 * res.n_classes - n_achiral


def test_xyz_input_with_distortion(tmp_path):
    ref = builtin_graph("dodecahedron")
    rng = np.random.default_rng(1)
    coords = ref.coords * (2.76 / 1.2360679775) + rng.normal(scale=0.05, size=ref.coords.shape)
    lines = [str(2 * len(coords)), "distorted dodecahedron"]
    for x, y, z in coords:
        lines.append(f"O {x:.6f} {y:.6f} {z:.6f}")
        lines.append(f"H {x + 0.5:.6f} {y:.6f} {z:.6f}")
    path = tmp_path / "graph.xyz"
    path.write_text("\n".join(lines) + "\n")

    graph = graph_from_xyz(path)
    assert np.array_equal(graph.edges, ref.edges)
    group = automorphism_group(graph)
    assert group.order == 120 and group.proper.sum() == 60
