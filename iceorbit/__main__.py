"""Enumerate ice-rule hydrogen-bond orientations on a 3-regular water network and classify them by symmetry.

Examples:
    python -m iceorbit                         # built-in 5^12 dodecahedron, full group (Ih)
    python -m iceorbit --group I               # rotations only (mirror images distinguished)
    python -m iceorbit --xyz water.xyz --out results/water
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np

from iceorbit import automorphism_group, builtin_graph, graph_from_xyz, classify, enumerate_ice_configs
from iceorbit.geometry import BUILTIN
from iceorbit.plot import plot_magnitude_distribution
from iceorbit.polarization import ZERO_TOL, magnitude_distribution, polarization


def bitstring(x: int, n_edges: int) -> str:
    """Edge e is character e (left to right): '0' means u->v, '1' means v->u for edge (u, v), u < v."""
    return "".join("1" if (x >> e) & 1 else "0" for e in range(n_edges))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--builtin", choices=sorted(BUILTIN), default="dodecahedron", help="built-in polyhedron")
    src.add_argument("--xyz", type=Path, help="XYZ file with oxygen positions (H atoms are ignored)")
    p.add_argument("--cutoff", type=float, default=None, help="O-O bond cutoff (default: 1.25 x shortest O-O)")
    p.add_argument(
        "--group",
        choices=["Ih", "I"],
        default="Ih",
        help="Ih: all graph automorphisms incl. reflections; I: rotations only",
    )
    p.add_argument("--out", type=Path, default=None, help="output prefix for <out>_classes.csv and <out>.npz")
    args = p.parse_args()

    t0 = time.perf_counter()
    graph = graph_from_xyz(args.xyz, args.cutoff) if args.xyz else builtin_graph(args.builtin)
    full = automorphism_group(graph)
    group = full if args.group == "Ih" else full.rotations()
    t1 = time.perf_counter()
    configs = enumerate_ice_configs(graph)
    t2 = time.perf_counter()
    res = classify(configs, group)
    t3 = time.perf_counter()
    mag = np.linalg.norm(polarization(configs, graph), axis=1)
    rep_mag = np.linalg.norm(polarization(res.representatives, graph), axis=1)
    dist = magnitude_distribution(mag, res.class_id)
    t4 = time.perf_counter()

    print(f"graph: V={graph.n_vertices} E={graph.n_edges}")
    print(f"automorphisms: {full.order} (proper {int(full.proper.sum())}); using {args.group}, order {group.order}")
    print(f"ice-rule configurations: {len(configs):,}")
    print(f"symmetry-distinct classes: {res.n_classes:,} (Burnside: {res.burnside_classes():g})")
    print("stabilizer order -> number of classes (degeneracy = |G|/|Stab|):")
    for s, k in sorted(Counter(res.stabilizer_order.tolist()).items()):
        print(f"  |Stab|={s:3d}  degeneracy={group.order // s:4d}  classes={k:,}")

    zero = mag < ZERO_TOL
    zero_class = rep_mag < ZERO_TOL
    nonzero = dist.values[dist.values > 0]
    print("polarization |P| (unit bond dipoles, outer bonds included):")
    print(f"  distinct values: {len(dist.values)}, min nonzero {nonzero.min():.6f}, max {dist.values.max():.6f}")
    print(f"  mean |P| = {mag.mean():.6f}, <|P|^2> = {(mag**2).mean():.6f}")
    print(f"  P = 0: {int(zero.sum()):,} configurations (probability {zero.mean():.6%}), {int(zero_class.sum()):,} classes")
    for (s, imp), k in sorted(Counter(zip(res.stabilizer_order[zero_class].tolist(), res.stabilizer_improper[zero_class].tolist())).items()):
        print(f"    |Stab|={s:3d} improper={int(imp)}  classes={k:,}  configurations={k * (group.order // s):,}")
    spread = np.zeros(res.n_classes)
    np.maximum.at(spread, res.class_id, np.abs(mag - rep_mag[res.class_id]))
    if spread.max() > ZERO_TOL:
        print(f"  note: |P| varies within a class by up to {spread.max():.3g} (geometry is not exactly symmetric)")
    print(
        f"time: symmetry {t1 - t0:.2f}s, enumeration {t2 - t1:.2f}s, "
        f"classification {t3 - t2:.2f}s, polarization {t4 - t3:.2f}s"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        csv_path = args.out.with_name(args.out.name + "_classes.csv")
        order = np.lexsort((res.representatives, -res.degeneracy))
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["class", "representative", "bits", "degeneracy", "stabilizer_order", "stabilizer_improper", "polarization"]
            )
            for k in order:
                rep = int(res.representatives[k])
                w.writerow(
                    [
                        int(k),
                        rep,
                        bitstring(rep, graph.n_edges),
                        int(res.degeneracy[k]),
                        int(res.stabilizer_order[k]),
                        int(res.stabilizer_improper[k]),
                        f"{rep_mag[k]:.9f}",
                    ]
                )
        pol_path = args.out.with_name(args.out.name + "_polarization.csv")
        with pol_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["polarization", "polarization_squared", "configurations", "probability", "classes"])
            for v, n, p, k in zip(dist.values, dist.n_configs, dist.probability, dist.n_classes):
                w.writerow([f"{v:.9f}", f"{v * v:.9f}", int(n), f"{p:.9g}", int(k)])
        png_path = args.out.with_name(args.out.name + "_polarization.png")
        try:
            plot_magnitude_distribution(dist, png_path, title=f"V={graph.n_vertices} graph, {len(configs):,} configurations")
        except ImportError:
            png_path = None
            print("matplotlib is not installed; skipping the histogram")
        npz_path = args.out.with_name(args.out.name + ".npz")
        np.savez_compressed(
            npz_path,
            coords=graph.coords,
            edges=graph.edges,
            configs=res.configs,
            class_id=res.class_id,
            representatives=res.representatives,
            degeneracy=res.degeneracy,
            stabilizer_order=res.stabilizer_order,
            stabilizer_improper=res.stabilizer_improper,
            vertex_perm=group.vertex_perm,
            proper=group.proper,
            polarization=mag,
            class_polarization=rep_mag,
        )
        for path in (csv_path, pol_path, png_path, npz_path):
            if path is not None:
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
