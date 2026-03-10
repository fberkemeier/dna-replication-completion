"""Command-line interface for DNAscape."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from .fitting import rfit
from .plotting import plotf
from .simulation import rsim


def _load_array(spec: str) -> np.ndarray:
    """Load a 1D/2D numeric array from file path, JSON/Python literal, or comma list."""
    p = Path(spec)
    if p.exists():
        if p.suffix.lower() == ".npy":
            return np.asarray(np.load(p), dtype=float)
        if p.suffix.lower() in {".txt", ".csv"}:
            return np.asarray(np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None), dtype=float)
        raise ValueError(f"Unsupported file extension for array input: {p.suffix}")

    # Try JSON / Python literal first
    try:
        val = json.loads(spec)
        return np.asarray(val, dtype=float)
    except Exception:
        pass
    try:
        val = ast.literal_eval(spec)
        return np.asarray(val, dtype=float)
    except Exception:
        pass

    # Fallback: comma-separated values
    try:
        return np.asarray([float(x.strip()) for x in spec.split(",") if x.strip()], dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not parse array input: {spec}") from exc


def _load_optional_array(spec: str | None):
    return None if spec is None else _load_array(spec)


def _save_rsim_npz(out_path: Path, result: dict):
    payload = {}
    for k, v in result.items():
        if v is None:
            continue
        if isinstance(v, list):
            payload[k] = np.asarray(v)
        else:
            payload[k] = np.asarray(v)
    np.savez_compressed(out_path, **payload)


def cmd_plotf(args: argparse.Namespace) -> int:
    # Headless backend for CLI plotting
    import matplotlib

    matplotlib.use("Agg")

    y_arrays = [_load_array(s) for s in args.arrays]

    if args.x_array is None:
        x_arg = None
    elif len(args.x_array) == 1:
        x_arg = _load_array(args.x_array[0])
    else:
        x_arg = [_load_array(s) for s in args.x_array]

    save_q = args.output is not None
    if save_q:
        out = Path(args.output)
        sname = out.stem
        ext = out.suffix.lstrip(".") or "png"
    else:
        sname = args.sname
        ext = args.ext

    plotf(
        *y_arrays,
        labels=args.labels,
        x_array=x_arg,
        dual_axis=args.dual_axis,
        xtitle=args.xtitle,
        ytitle=args.ytitle,
        title=args.title,
        resolution=args.resolution,
        saveQ=save_q,
        sname=sname,
        ext=ext,
        showQ=not save_q,
    )
    return 0


def cmd_rsim(args: argparse.Namespace) -> int:
    ori_rate = _load_array(args.ori_rate)
    fork_speed = _load_array(args.fork_speed) if args.fork_speed is not None else 1.4

    result = rsim(
        ori_rate=ori_rate,
        fork_speed=fork_speed,
        sim_number=args.sim_number,
        resolution_space=args.resolution_space,
        resolution_time=args.resolution_time,
        perQ=args.perQ,
        stall_rate=args.stall_rate,
        tau=args.tau,
        time_statsQ=args.time_statsQ,
        time_stats_xtQ=args.time_stats_xtQ,
        verbose=not args.quiet,
    )
    out = Path(args.output)
    _save_rsim_npz(out, result)
    print(f"Saved rsim output to {out}")
    return 0


def cmd_rfit(args: argparse.Namespace) -> int:
    source = _load_optional_array(args.source)
    result = rfit(
        d1=args.d1,
        d2=args.d2,
        cell_line=args.cell_line,
        chr_number=args.chr_number,
        fork_speed=args.fork_speed,
        resolution=args.resolution,
        source=[] if source is None else source,
        saveQ=args.saveQ,
    )
    if args.output:
        out = Path(args.output)
        np.savetxt(out, np.asarray(result, dtype=float))
        print(f"Saved rfit output to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dnascape", description="DNAscape command-line interface")
    sub = p.add_subparsers(dest="command", required=True)

    p_plot = sub.add_parser("plotf", help="Plot one or more arrays")
    p_plot.add_argument("--arrays", nargs="+", required=True, help="Y arrays (files, JSON, or comma list)")
    p_plot.add_argument(
        "--x-array",
        nargs="+",
        help="Either one shared X array or one X array per Y array (files, JSON, or comma list)",
    )
    p_plot.add_argument("--labels", nargs="*", default=None, help="Legend labels")
    p_plot.add_argument("--dual-axis", action="store_true", help="Use dual y-axis (requires exactly two arrays)")
    p_plot.add_argument("--xtitle", default="Index")
    p_plot.add_argument("--ytitle", default="Value")
    p_plot.add_argument("--title", default="")
    p_plot.add_argument("--resolution", type=float, default=1.0)
    p_plot.add_argument("--output", default=None, help="Output figure path (e.g., out.png). If omitted, shows plot.")
    p_plot.add_argument("--sname", default="test", help="Fallback figure name when --output is not used")
    p_plot.add_argument("--ext", default="png", help="Fallback figure extension when --output is not used")
    p_plot.set_defaults(func=cmd_plotf)

    p_sim = sub.add_parser("rsim", help="Run replication simulation")
    p_sim.add_argument("--ori-rate", required=True, help="Origin-rate array input (file/JSON/comma list)")
    p_sim.add_argument("--fork-speed", default=None, help="Fork-speed scalar/array input (file/JSON/comma list)")
    p_sim.add_argument("--sim-number", type=int, default=50)
    p_sim.add_argument("--resolution-space", type=float, default=1.0)
    p_sim.add_argument("--resolution-time", type=float, default=1.0)
    p_sim.add_argument("--perQ", action="store_true")
    p_sim.add_argument("--stall-rate", type=float, default=0.0)
    p_sim.add_argument("--tau", type=float, default=np.inf)
    p_sim.add_argument("--time-statsQ", action="store_true")
    p_sim.add_argument("--time-stats-xtQ", action="store_true")
    p_sim.add_argument("--quiet", action="store_true")
    p_sim.add_argument("--output", default="rsim_output.npz", help="Path to npz output")
    p_sim.set_defaults(func=cmd_rsim)

    p_fit = sub.add_parser("rfit", help="Map one data type to another")
    p_fit.add_argument("--d1", default="firing_rate")
    p_fit.add_argument("--d2", default="replication_timing")
    p_fit.add_argument("--cell-line", default="H1")
    p_fit.add_argument("--chr-number", type=int, default=1)
    p_fit.add_argument("--fork-speed", type=float, default=1.4)
    p_fit.add_argument("--resolution", type=float, default=1.0)
    p_fit.add_argument("--source", default=None, help="Optional input array (file/JSON/comma list)")
    p_fit.add_argument("--saveQ", action="store_true")
    p_fit.add_argument("--output", default=None, help="Optional txt output path")
    p_fit.set_defaults(func=cmd_rfit)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
