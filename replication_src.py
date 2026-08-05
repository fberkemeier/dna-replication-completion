"""Reusable tools for replication completion-bound analyses."""

import heapq
import math
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybigtools
from scipy.optimize import curve_fit


def rescale(arr, interval):
    """Linearly rescale an array to a new numeric interval."""
    new_min, new_max = interval
    arr = np.asarray(arr, dtype=float)
    old_min = np.nanmin(arr)
    old_max = np.nanmax(arr)

    if old_max == old_min:
        return np.full_like(arr, (new_min + new_max) / 2.0)

    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def fast_shift_add(dst, src, shift, perQ=False):
    """Add a shifted copy of ``src`` into ``dst`` in-place."""
    if shift == 0:
        dst += src
        return

    if perQ:
        dst += np.roll(src, shift)
        return

    if shift > 0:
        dst[shift:] += src[:-shift]
    else:
        dst[:shift] += src[-shift:]


def mean_squared_error(a, b):
    """Return the mean squared error between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))


def hms(seconds):
    """Format seconds as ``HH:MM:SS``."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def etaf(it, maxiter, start_time):
    """Print a compact elapsed/ETA progress line."""
    elapsed = time.time() - start_time
    average = elapsed / max(it, 1)
    eta = (maxiter - it) * average
    print(f"[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta)}", end="\r")


def etaferr(it, maxiter, start_time, err):
    """Print a compact elapsed/ETA progress line with fit error."""
    elapsed = time.time() - start_time
    average = elapsed / max(it, 1)
    eta = (maxiter - it) * average
    print(
        f"[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta)}  MSE: {err:.4e}",
        end="\r",
    )


def map_firing_timing(
    firing_rate,
    fork_speed=1.4,
    resolution=1.0,
    neighbourhood_range=2000,
    perQ=False,
):
    """Map a firing-rate landscape to its predicted replication timing."""
    rates = np.asarray(firing_rate, dtype=float)

    if rates.ndim != 1:
        raise ValueError("firing_rate must be a 1D array.")
    if np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("firing_rate must be finite and non-negative.")
    if fork_speed <= 0 or resolution <= 0:
        raise ValueError("fork_speed and resolution must be positive.")
    if rates.size == 0:
        return rates.copy()

    fork_speed_per_bin = fork_speed / resolution
    requested_radius = max(1, int(neighbourhood_range / resolution))

    if perQ:
        max_radius = max(0, (rates.size - 2) // 2)
        radius = min(requested_radius, max_radius)
    else:
        radius = requested_radius

    mapped = np.zeros_like(rates)
    cumulative_rate = rates.copy()
    last_exponent = np.zeros_like(rates)
    last_weight = np.ones_like(rates)

    for shift in range(radius + 1):
        if shift:
            fast_shift_add(cumulative_rate, rates, shift, perQ=perQ)
            fast_shift_add(cumulative_rate, rates, -shift, perQ=perQ)

        next_exponent = last_exponent + cumulative_rate / fork_speed_per_bin
        next_weight = np.exp(-next_exponent)

        mapped += (last_weight - next_weight) / np.clip(cumulative_rate, 1e-80, None)
        last_exponent = next_exponent
        last_weight = next_weight

    return mapped


def map_timing_firing(
    observed_timing,
    fork_speed=1.4,
    resolution=1.0,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    err_threshold=15.0,
    deviation_threshold=None,
    verbose=True,
    perQ=False,
):
    """Fit a firing-rate landscape that reproduces an observed timing profile."""
    timing = np.asarray(observed_timing, dtype=float)

    if timing.ndim != 1:
        raise ValueError("observed_timing must be a 1D array.")
    if np.any(~np.isfinite(timing)):
        raise ValueError("observed_timing must be finite.")
    if fork_speed <= 0 or resolution <= 0:
        raise ValueError("fork_speed and resolution must be positive.")
    if maxiter < 0:
        raise ValueError("maxiter must be non-negative.")
    if timing.size == 0:
        return timing.copy()

    fork_speed_per_bin = fork_speed / resolution
    floor = 10.0 ** (-err_threshold)

    rates = (math.pi / (4.0 * fork_speed_per_bin)) * np.maximum(timing, 1e-80) ** (-2)
    rates = np.maximum(rates, floor)

    crop_margin = int(1000 / resolution)
    if perQ or 2 * crop_margin >= timing.size:
        crop = slice(None)
    else:
        crop = slice(crop_margin, -crop_margin)

    predicted = map_firing_timing(
        rates,
        fork_speed=fork_speed,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
    )

    start_time = time.time()

    for iteration in range(1, int(maxiter) + 1):
        safe_timing = np.maximum(timing, 1e-80)
        safe_predicted = np.maximum(predicted, 1e-80)

        updated = rates * (safe_predicted / safe_timing) ** fit_step
        updated = np.maximum(updated, floor)

        if deviation_threshold is not None:
            close = np.abs(predicted - timing) <= deviation_threshold
            updated[close] = rates[close]

        rates = updated
        predicted = map_firing_timing(
            rates,
            fork_speed=fork_speed,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ,
        )

        if verbose:
            mse = mean_squared_error(timing[crop], predicted[crop])
            etaferr(iteration, int(maxiter), start_time, mse)

    return rates


def _validate_trimmed_options(
    stall_rate,
    ffiring_nr,
    ffiring_recycle,
    ffiring_forkQ,
    time_stats_xtQ,
):
    if stall_rate != 0.0:
        raise NotImplementedError("Fork stalling was removed from this trimmed repo.")
    if ffiring_nr is not None or ffiring_recycle != 0.0 or ffiring_forkQ:
        raise NotImplementedError(
            "Firing-factor models were removed from this trimmed repo."
        )
    if time_stats_xtQ:
        raise NotImplementedError(
            "2D space-time statistics were removed from this trimmed repo."
        )


def _count_from_intervals(time_grid, starts, ends):
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)

    if starts.size == 0:
        return np.zeros_like(time_grid, dtype=float)

    starts.sort()
    ends.sort()

    started = np.searchsorted(starts, time_grid, side="right")
    ended = np.searchsorted(ends, time_grid, side="left")
    return (started - ended).astype(float)


def _propagate_linear(t_fire, edge_time):
    n = len(t_fire)
    g = np.empty(n, dtype=float)
    src = np.empty(n, dtype=np.int32)
    prev = np.full(n, -1, dtype=np.int32)

    g[0], src[0], prev[0] = t_fire[0], 0, 0

    for idx in range(1, n):
        candidate = g[idx - 1] + edge_time
        if t_fire[idx] <= candidate:
            g[idx], src[idx], prev[idx] = t_fire[idx], idx, idx
        else:
            g[idx], src[idx], prev[idx] = candidate, src[idx - 1], idx - 1

    for idx in range(n - 2, -1, -1):
        candidate = g[idx + 1] + edge_time
        if candidate < g[idx]:
            g[idx], src[idx], prev[idx] = candidate, src[idx + 1], idx + 1

    return g, src, prev


def _propagate_circular(t_fire, edge_time):
    n = len(t_fire)
    g = t_fire.copy()
    src = np.arange(n, dtype=np.int32)
    prev = np.arange(n, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    queue = [(g[idx], idx) for idx in range(n) if np.isfinite(g[idx])]
    heapq.heapify(queue)

    while queue:
        time_u, u = heapq.heappop(queue)
        if visited[u] or time_u > g[u]:
            continue

        visited[u] = True

        for v in ((u + 1) % n, (u - 1) % n):
            candidate = time_u + edge_time
            if candidate < g[v]:
                g[v] = candidate
                src[v] = src[u]
                prev[v] = u
                heapq.heappush(queue, (candidate, v))

    return g, src, prev


def simulate_replication(
    ori_rate,
    fork_speed,
    resolution_space=1.0,
    resolution_time=1.0,
    rng=None,
    perQ=False,
    stall_rate=0.0,
    tau=np.inf,
    ffiring_nr=None,
    ffiring_recycle=0.0,
    ffiring_forkQ=False,
    time_statsQ=False,
    time_stats_xtQ=False,
    time_stats_densQ=None,
    time_grid=None,
    max_rep_time=2000.0,
):
    """Run one stochastic replication simulation on a 1D rate landscape."""
    del tau

    if time_stats_densQ is not None:
        time_stats_xtQ = bool(time_stats_densQ)

    _validate_trimmed_options(
        stall_rate=stall_rate,
        ffiring_nr=ffiring_nr,
        ffiring_recycle=ffiring_recycle,
        ffiring_forkQ=ffiring_forkQ,
        time_stats_xtQ=time_stats_xtQ,
    )

    rates = np.asarray(ori_rate, dtype=float)
    if rates.ndim != 1:
        raise ValueError("ori_rate must be a 1D array in this trimmed repo.")
    if np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("ori_rate must be finite and non-negative.")

    fork_speed = float(fork_speed)
    resolution_space = float(resolution_space)
    resolution_time = float(resolution_time)

    if not np.isfinite(fork_speed) or fork_speed <= 0:
        raise ValueError("fork_speed must be a finite positive number.")
    if not np.isfinite(resolution_space) or resolution_space <= 0:
        raise ValueError("resolution_space must be a finite positive number.")
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_time must be a finite positive number.")

    n = int(rates.size)
    if n == 0:
        empty_int = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=float)
        return (
            empty_int,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            None,
            empty_float,
            empty_float,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )

    rng = np.random.default_rng() if rng is None else rng
    edge_time = resolution_space / fork_speed

    t_fire = np.full(n, np.inf, dtype=float)
    positive = rates > 0
    if np.any(positive):
        t_fire[positive] = rng.exponential(scale=1.0 / rates[positive])

    if perQ:
        replication_time, src, prev = _propagate_circular(t_fire, edge_time)
    else:
        replication_time, src, prev = _propagate_linear(t_fire, edge_time)

    if max_rep_time is not None:
        replication_time = np.minimum(replication_time, float(max_rep_time))

    idx = np.arange(n, dtype=np.int32)
    fired_mask = (src == idx) & np.isfinite(t_fire)
    fired_idx = np.flatnonzero(fired_mask).astype(np.int64, copy=False)
    fired_times = t_fire[fired_idx]
    fired_positions = fired_idx.astype(float) * resolution_space

    if fired_idx.size >= 2:
        fired_sorted = np.sort(fired_idx)
        diffs = np.diff(fired_sorted).astype(float)
        if perQ:
            wrap = float(n - fired_sorted[-1] + fired_sorted[0])
            inter_origin_distances = np.concatenate([diffs, [wrap]]) * resolution_space
        else:
            inter_origin_distances = diffs * resolution_space
    else:
        inter_origin_distances = np.array([], dtype=float)

    fork_directionality = np.zeros(n, dtype=float)
    if perQ:
        fork_directionality[prev == ((idx - 1) % n)] = 1.0
        fork_directionality[prev == ((idx + 1) % n)] = -1.0
    else:
        fork_directionality[prev == (idx - 1)] = 1.0
        fork_directionality[prev == (idx + 1)] = -1.0

    time_stats = None
    if time_statsQ:
        if time_grid is None:
            raise ValueError("time_statsQ=True requires a time_grid array.")

        time_grid = np.asarray(time_grid, dtype=float)
        t_max = float(time_grid[-1])

        parents = prev.astype(np.int64, copy=False)
        good = (
            (parents >= 0)
            & np.isfinite(replication_time)
            & np.isfinite(replication_time[parents])
        )
        cols = np.flatnonzero(good).astype(np.int64, copy=False)
        starts = replication_time[parents[cols]]
        ends = replication_time[cols]

        valid = np.isfinite(starts) & np.isfinite(ends) & (starts < ends) & (starts <= t_max)
        total_forks = _count_from_intervals(
            time_grid,
            starts[valid],
            np.minimum(ends[valid], t_max),
        )

        time_stats = {
            "time_grid": time_grid,
            "total_forks": total_forks,
            "active_forks": total_forks.copy(),
            "stalled_forks": np.zeros_like(total_forks),
            "firing_factors": None,
        }

    return (
        fired_idx,
        fired_positions,
        fired_times,
        replication_time,
        inter_origin_distances,
        time_stats,
        fork_directionality,
        t_fire,
        src,
        prev,
    )


def rsim(
    ori_rate=0.1,
    fork_speed=1.4,
    sim_number=50,
    resolution_space=1.0,
    resolution_time=1.0,
    perQ=False,
    stall_rate=0.0,
    tau=np.inf,
    ffiring_nr=None,
    ffiring_recycle=0.0,
    ffiring_forkQ=False,
    time_statsQ=False,
    time_stats_xtQ=False,
    time_stats_densQ=None,
    time_grid=np.arange(0.0, 1501.0, 1.0),
    max_rep_time=2000.0,
    seed=None,
    verbose=True,
    print_every=1,
    dens_block_n=4096,
    dens_memmap_dir=None,
):
    """Run an ensemble of trimmed replication simulations."""
    del dens_block_n, dens_memmap_dir

    if time_stats_densQ is not None:
        time_stats_xtQ = bool(time_stats_densQ)

    _validate_trimmed_options(
        stall_rate=stall_rate,
        ffiring_nr=ffiring_nr,
        ffiring_recycle=ffiring_recycle,
        ffiring_forkQ=ffiring_forkQ,
        time_stats_xtQ=time_stats_xtQ,
    )

    rates = np.asarray(ori_rate, dtype=float)
    if rates.ndim != 1:
        raise ValueError("ori_rate must be a 1D array in this trimmed repo.")

    sim_number = int(sim_number)
    if sim_number <= 0:
        raise ValueError("sim_number must be a positive integer.")

    n = int(rates.size)
    seeds = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()
    generators = [np.random.default_rng(state) for state in seeds.spawn(sim_number)]

    rep_times_per_sim = np.empty((sim_number, n), dtype=float)
    num_oris = np.empty(sim_number, dtype=np.int64)
    fire_counts = np.zeros(n, dtype=np.int64)
    all_inter_origin_distances = []

    sum_rep_time = np.zeros(n, dtype=float)
    count_rep_time = np.zeros(n, dtype=np.int64)
    sum_fork_directionality = np.zeros(n, dtype=float)
    count_fork_directionality = np.zeros(n, dtype=np.int64)

    if time_grid is not None:
        time_grid = np.asarray(time_grid, dtype=float)
        sum_replicated_fraction = np.zeros(time_grid.size, dtype=float)
        dt_grid = float(time_grid[1] - time_grid[0]) if time_grid.size >= 2 else None
    else:
        sum_replicated_fraction = None
        dt_grid = None

    if time_statsQ:
        if time_grid is None:
            raise ValueError("time_statsQ=True requires a time_grid array.")
        sum_total_forks = np.zeros(time_grid.size, dtype=float)
        sum_active_forks = np.zeros(time_grid.size, dtype=float)
        sum_stalled_forks = np.zeros(time_grid.size, dtype=float)
    else:
        sum_total_forks = None
        sum_active_forks = None
        sum_stalled_forks = None

    start_time = time.time()

    for iteration, rng in enumerate(generators, start=1):
        fired_idx, _, _, rep_time, iod, time_stats, fdir, _, _, _ = simulate_replication(
            ori_rate=rates,
            fork_speed=fork_speed,
            resolution_space=resolution_space,
            resolution_time=resolution_time,
            rng=rng,
            perQ=perQ,
            stall_rate=stall_rate,
            tau=tau,
            ffiring_nr=ffiring_nr,
            ffiring_recycle=ffiring_recycle,
            ffiring_forkQ=ffiring_forkQ,
            time_statsQ=time_statsQ,
            time_stats_xtQ=time_stats_xtQ,
            time_stats_densQ=time_stats_densQ,
            time_grid=time_grid,
            max_rep_time=max_rep_time,
        )

        rep_times_per_sim[iteration - 1, :] = rep_time
        num_oris[iteration - 1] = int(fired_idx.size)

        finite = np.isfinite(rep_time)
        sum_rep_time[finite] += rep_time[finite]
        count_rep_time[finite] += 1
        sum_fork_directionality[finite] += fdir[finite]
        count_fork_directionality[finite] += 1

        if fired_idx.size:
            fire_counts[fired_idx] += 1
        if iod.size:
            all_inter_origin_distances.append(iod)

        if time_grid is not None:
            sorted_times = np.sort(rep_time[finite])
            if sorted_times.size:
                sum_replicated_fraction += (
                    np.searchsorted(sorted_times, time_grid, side="right") / float(n)
                )

        if time_statsQ and time_stats is not None:
            sum_total_forks += time_stats["total_forks"]
            sum_active_forks += time_stats["active_forks"]
            sum_stalled_forks += time_stats["stalled_forks"]

        if verbose and (iteration % int(print_every) == 0 or iteration == sim_number):
            etaf(iteration, sim_number, start_time)

    avg_replication_timing = np.full(n, np.inf, dtype=float)
    observed = count_rep_time > 0
    avg_replication_timing[observed] = (
        sum_rep_time[observed] / count_rep_time[observed].astype(float)
    )

    avg_fork_directionality = np.full(n, np.nan, dtype=float)
    fdir_observed = count_fork_directionality > 0
    avg_fork_directionality[fdir_observed] = (
        sum_fork_directionality[fdir_observed]
        / count_fork_directionality[fdir_observed].astype(float)
    )

    replicated_fraction = (
        sum_replicated_fraction / float(sim_number)
        if sum_replicated_fraction is not None
        else None
    )

    if time_statsQ:
        avg_total_forks = sum_total_forks / float(sim_number)
        avg_active_forks = sum_active_forks / float(sim_number)
        avg_stalled_forks = sum_stalled_forks / float(sim_number)
    else:
        avg_total_forks = None
        avg_active_forks = None
        avg_stalled_forks = None

    return {
        "replication_timing": avg_replication_timing,
        "rep_times_per_sim": rep_times_per_sim,
        "s_phase_duration": rep_times_per_sim.max(axis=1),
        "num_oris": num_oris.tolist(),
        "inter_origin_distances": (
            np.concatenate(all_inter_origin_distances)
            if all_inter_origin_distances
            else np.array([], dtype=float)
        ),
        "efficiency": fire_counts.astype(float) / float(sim_number),
        "fork_directionality": avg_fork_directionality,
        "time_grid": time_grid,
        "dt_grid": dt_grid,
        "replicated_fraction": replicated_fraction,
        "replicated_fraction_xt": None,
        "initiation_events_xt": None,
        "coalescence_events_xt": None,
        "right_moving_fork_density_xt": None,
        "left_moving_fork_density_xt": None,
        "total_forks": avg_total_forks,
        "active_forks": avg_active_forks,
        "stalled_forks": avg_stalled_forks,
        "firing_factors": None,
    }


def rfit(
    d1="firing_rate",
    d2="replication_timing",
    cell_line="H1",
    chr_number=1,
    fork_speed=1.4,
    resolution=1.0,
    source=None,
    saveQ=False,
    **kwargs,
):
    """Minimal mapper wrapper kept for notebook compatibility."""
    del cell_line, chr_number

    if source is None:
        raise ValueError(
            "This trimmed repository expects 'source' to be passed explicitly."
        )
    if saveQ:
        raise NotImplementedError("saveQ was removed from this trimmed repo.")

    data = np.asarray(source, dtype=float)

    if d1 == "replication_timing" and d2 == "firing_rate":
        return map_timing_firing(
            data,
            fork_speed=fork_speed,
            resolution=resolution,
            **kwargs,
        )
    if d1 == "firing_rate" and d2 == "replication_timing":
        return map_firing_timing(
            data,
            fork_speed=fork_speed,
            resolution=resolution,
            **kwargs,
        )

    raise ValueError(
        "Only replication_timing <-> firing_rate mappings are kept in this repo."
    )


def plotf(
    *arrays,
    labels=None,
    x_array=None,
    resolution=1.0,
    invyQ=False,
    figsize=(10, 4),
    xlims=(None, None),
    ylims=(None, None),
    title="",
    xtitle="Index",
    ytitle="Value",
    logyQ=False,
    logxQ=False,
    saveQ=False,
    sname="test",
    ext="pdf",
    fig=None,
    ax=None,
    showQ=True,
    return_handles=False,
):
    """Plot one or more arrays against a shared or per-array x-axis."""
    if not arrays:
        raise ValueError("plotf requires at least one array.")

    curves = [np.asarray(arr, dtype=float) for arr in arrays]

    if labels is not None and len(labels) != len(curves):
        raise ValueError("labels length must match the number of plotted arrays.")

    if x_array is None:
        xs = [np.arange(len(arr), dtype=float) * resolution for arr in curves]
    elif (
        isinstance(x_array, (list, tuple))
        and len(x_array) == len(curves)
        and all(np.ndim(values) > 0 for values in x_array)
    ):
        xs = [np.asarray(values, dtype=float) for values in x_array]
    else:
        shared = np.asarray(x_array, dtype=float)
        if shared.ndim != 1:
            raise ValueError(
                "x_array must be a 1D array or a list of 1D arrays matching the data."
            )
        xs = [shared] * len(curves)

    for idx, (x_vals, y_vals) in enumerate(zip(xs, curves)):
        if x_vals.ndim != 1:
            raise ValueError(f"x_array[{idx}] must be 1D.")
        if len(x_vals) != len(y_vals):
            raise ValueError(f"x_array[{idx}] must match arrays[{idx}] in length.")

    created_axes = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) if fig is None else (fig, fig.add_subplot(111))
    elif fig is None:
        fig = ax.figure

    for idx, (x_vals, y_vals) in enumerate(zip(xs, curves)):
        label = labels[idx] if labels is not None else None
        ax.plot(x_vals, y_vals, label=label)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_title(title)
    if logyQ:
        ax.set_yscale("log")
    if logxQ:
        ax.set_xscale("log")

    if xlims is not None and any(value is not None for value in xlims):
        ax.set_xlim(xlims)
    if ylims is not None and any(value is not None for value in ylims):
        ax.set_ylim(ylims)

    if invyQ:
        ax.invert_yaxis()

    if labels is not None:
        ax.legend()

    fig.tight_layout()

    if saveQ:
        out_path = Path("figures") / f"plot_{sname}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")

    if created_axes and showQ:
        plt.show()

    if return_handles:
        return fig, ax
    return None

plt.rcParams.update({
    "figure.figsize": (7.2, 4.6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = Path("data")
TIMING_DIR = Path("timing")
FIGURE_DIR = Path("figures")
TIMING_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

S_PHASE_BINS = ["S1", "S2", "S3", "S4", "S5"]
STANDARD_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]




def safe_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("$", "")
        .replace("\\", "")
        .replace("+", "plus")
    )


def available_cell_lines_from_data(data_dir=DATA_DIR, resolution=10_000,
                                   s_phase_bins=S_PHASE_BINS):
    first_bin = s_phase_bins[0]
    suffix = f"_{first_bin}_{resolution}.bw"
    cell_lines = []

    for path in sorted(data_dir.glob(f"*_{first_bin}_{resolution}.bw")):
        cell_line = path.name[:-len(suffix)]

        if all((data_dir / f"{cell_line}_{sbin}_{resolution}.bw").exists()
               for sbin in s_phase_bins):
            cell_lines.append(cell_line)

    return cell_lines


def get_bigwig_chrom_sizes(cell_line, resolution=10_000, data_dir=DATA_DIR, sbin="S1"):
    path = data_dir / f"{cell_line}_{sbin}_{resolution}.bw"

    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")

    bw = pybigtools.open(str(path))

    try:
        chroms = dict(bw.chroms())
    finally:
        bw.close()

    return chroms


def resolve_chrom_name(chrom, chrom_sizes):
    candidates = [chrom]

    if chrom.startswith("chr"):
        candidates.append(chrom[3:])
    else:
        candidates.append(f"chr{chrom}")

    if chrom in {"chrM", "M", "MT"}:
        candidates.extend(["chrM", "M", "MT"])

    for candidate in candidates:
        if candidate in chrom_sizes:
            return candidate

    return None


def require_chrom_name(chrom, chrom_sizes, cell_line, resolution):
    resolved = resolve_chrom_name(chrom, chrom_sizes)

    if resolved is None:
        examples = ", ".join(list(chrom_sizes)[:8])
        raise ValueError(
            f"{chrom} is not present in {cell_line}_S1_{resolution}.bw. "
            f"Available chromosome examples: {examples}"
        )

    return resolved


def build_chromosome_configs(cell_line="DM", chroms=("chr20",), resolution=10_000,
                             data_dir=DATA_DIR, analysis_tag="interval"):
    chrom_sizes = get_bigwig_chrom_sizes(
        cell_line=cell_line,
        resolution=resolution,
        data_dir=data_dir,
    )

    configs = {}

    for requested_chrom in chroms:
        chrom = require_chrom_name(
            requested_chrom,
            chrom_sizes=chrom_sizes,
            cell_line=cell_line,
            resolution=resolution,
        )

        chrom_length = int(chrom_sizes[chrom])
        end = chrom_length - (chrom_length % resolution)

        key = f"{cell_line}_{analysis_tag}_{safe_filename(requested_chrom)}"

        configs[key] = {
            "key": key,
            "label": f"{cell_line}, {requested_chrom} finite interval",
            "short_label": f"{cell_line} {requested_chrom} interval",
            "point_label": f"{cell_line} {requested_chrom}",
            "cell_line": cell_line,
            "chrom": chrom,
            "requested_chrom": requested_chrom,
            "start": 0,
            "end": end,
            "resolution": resolution,
            "fit_periodic": False,
            "sim_periodic": False,
            "bound_geometries": ["interval"],
            "boundary_conditions": "zero_inflow",
            "analysis_type": "interval_profile",
        }

    return configs


def build_chromosome_configs_for_cell_lines(cell_lines, chroms=("chr20",),
                                           resolution=10_000, data_dir=DATA_DIR,
                                           analysis_tag="interval"):
    configs = {}

    for cell_line in cell_lines:
        configs.update(build_chromosome_configs(
            cell_line=cell_line,
            chroms=chroms,
            resolution=resolution,
            data_dir=data_dir,
            analysis_tag=analysis_tag,
        ))

    return configs


def build_periodic_interval_configs(cell_lines, regions, resolution=10_000,
                                    data_dir=DATA_DIR):
    if isinstance(cell_lines, str):
        cell_lines = [cell_lines]

    configs = {}

    for cell_line in cell_lines:
        chrom_sizes = get_bigwig_chrom_sizes(
            cell_line=cell_line,
            resolution=resolution,
            data_dir=data_dir,
        )

        for region in regions:
            requested_chrom = region["chrom"]
            chrom = require_chrom_name(
                requested_chrom,
                chrom_sizes=chrom_sizes,
                cell_line=cell_line,
                resolution=resolution,
            )

            region_id = region.get(
                "region_id",
                safe_filename(f"{requested_chrom}_{region['start']}_{region['end']}").upper(),
            )
            label = region.get("label", f"{requested_chrom}:{region['start']}-{region['end']}")
            short_label = region.get("short_label", label)
            key = f"{cell_line}_{region_id}"

            configs[key] = {
                "key": key,
                "label": f"{cell_line}, {label}",
                "short_label": f"{cell_line} {short_label}",
                "point_label": f"{cell_line} {short_label}",
                "cell_line": cell_line,
                "chrom": chrom,
                "requested_chrom": requested_chrom,
                "start": region["start"],
                "end": region["end"],
                "resolution": resolution,
                "fit_periodic": True,
                "sim_periodic": True,
                "bound_geometries": ["torus"],
                "analysis_type": "periodic_interval",
            }

    return configs


def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def read_bigwig_bin_mean(bw, chrom, start, end):
    vals = bw.values(chrom, int(start), int(end))

    if vals is None:
        return np.nan

    vals = np.asarray(
        [np.nan if v is None else float(v) for v in vals],
        dtype=float,
    )

    if vals.size == 0:
        return np.nan

    return np.nanmean(vals)


def compute_timing_curve(chrom, start, end, cell_line="DM", resolution=10_000,
                         data_dir=DATA_DIR, s_phase_bins=S_PHASE_BINS):
    x_bins = np.arange(1, len(s_phase_bins) + 1, dtype=float)

    bw_files = [
        pybigtools.open(str(data_dir / f"{cell_line}_{sbin}_{resolution}.bw"))
        for sbin in s_phase_bins
    ]

    positions = np.arange(start, end, resolution, dtype=int)
    rt_values = []

    try:
        for pos in positions:
            signals = np.array([
                read_bigwig_bin_mean(bw, chrom, pos, pos + resolution)
                for bw in bw_files
            ], dtype=float)

            if np.any(~np.isfinite(signals)) or np.nansum(signals) <= 0:
                rt_values.append(np.nan)
                continue

            norm_signals = signals / np.sum(signals)
            cumulative = np.cumsum(norm_signals)

            try:
                popt, _ = curve_fit(
                    sigmoid,
                    x_bins,
                    cumulative,
                    p0=(2.0, 3.0),
                    bounds=([0.1, 1.0], [10.0, len(s_phase_bins)]),
                    maxfev=10_000,
                )
                _, x0 = popt
                rt_values.append(x0)
            except RuntimeError:
                rt_values.append(np.nan)

    finally:
        for bw in bw_files:
            bw.close()

    return positions, np.asarray(rt_values, dtype=float)


def compute_timing_curve_from_config(cfg, data_dir=DATA_DIR):
    return compute_timing_curve(
        chrom=cfg["chrom"],
        start=cfg["start"],
        end=cfg["end"],
        cell_line=cfg["cell_line"],
        resolution=cfg["resolution"],
        data_dir=data_dir,
    )


def timing_cache_stem(cfg):
    chrom = cfg.get("requested_chrom", cfg["chrom"])
    analysis_type = cfg.get("analysis_type", "domain")
    return safe_filename(
        f"{cfg['cell_line']}_{analysis_type}_{chrom}_{cfg['start']}_{cfg['end']}_res{cfg['resolution']}"
    )


def timing_cache_path(cfg, timing_dir=TIMING_DIR):
    return Path(timing_dir) / f"{timing_cache_stem(cfg)}_raw_timing.csv"


def timing_curve_table(cfg, positions, timing_raw):
    positions = np.asarray(positions, dtype=np.int64)
    timing_raw = np.asarray(timing_raw, dtype=float)
    resolution = int(cfg["resolution"])

    return pd.DataFrame({
        "cell_line": cfg["cell_line"],
        "analysis_type": cfg.get("analysis_type", ""),
        "chrom": cfg["chrom"],
        "requested_chrom": cfg.get("requested_chrom", cfg["chrom"]),
        "region_start_bp": int(cfg["start"]),
        "region_end_bp": int(cfg["end"]),
        "resolution_bp": resolution,
        "bin_start_bp": positions,
        "bin_end_bp": positions + resolution,
        "timing_s_phase_midpoint": timing_raw,
    })


def _arrays_from_timing_table(table):
    return (
        table["bin_start_bp"].to_numpy(dtype=int),
        table["timing_s_phase_midpoint"].to_numpy(dtype=float),
    )


def load_timing_curve_csv(path):
    table = pd.read_csv(path)
    required = {"bin_start_bp", "timing_s_phase_midpoint"}
    missing = required.difference(table.columns)

    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return _arrays_from_timing_table(table)


def save_timing_curve_csv(cfg, data_dir=DATA_DIR, timing_dir=TIMING_DIR,
                          overwrite=False):
    path = timing_cache_path(cfg, timing_dir=timing_dir)

    if path.exists() and not overwrite:
        print(f"Timing CSV already exists: {path}")
        return path, pd.read_csv(path)

    positions, timing_raw = compute_timing_curve_from_config(
        cfg,
        data_dir=data_dir,
    )
    table = timing_curve_table(cfg, positions, timing_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Wrote raw timing CSV: {path} ({len(table):,} bins)")

    return path, table


def get_timing_curve_from_config(cfg, data_dir=DATA_DIR, timing_dir=TIMING_DIR,
                                 cache_mode="auto", overwrite=False):
    modes = {"auto", "load", "build", "ignore"}

    if cache_mode not in modes:
        raise ValueError(f"cache_mode must be one of {sorted(modes)}")

    if cache_mode == "ignore":
        return compute_timing_curve_from_config(cfg, data_dir=data_dir)

    path = timing_cache_path(cfg, timing_dir=timing_dir)

    if path.exists() and cache_mode in {"auto", "load"} and not overwrite:
        print(f"Loading raw timing CSV: {path}")
        return load_timing_curve_csv(path)

    if cache_mode == "load":
        raise FileNotFoundError(
            f"Raw timing CSV not found: {path}. Run preprocessing first, "
            "or use cache_mode='auto' to build it when missing."
        )

    _, table = save_timing_curve_csv(
        cfg,
        data_dir=data_dir,
        timing_dir=timing_dir,
        overwrite=overwrite or cache_mode == "build",
    )
    return _arrays_from_timing_table(table)


def timing_cache_table(configs, selected_keys=None, timing_dir=TIMING_DIR):
    if selected_keys is None:
        selected_keys = list(configs)

    rows = []

    for key in selected_keys:
        cfg = configs[key]
        path = timing_cache_path(cfg, timing_dir=timing_dir)
        rows.append({
            "dataset_key": key,
            "dataset_label": cfg["label"],
            "raw_timing_csv": str(path),
            "exists": path.exists(),
        })

    return pd.DataFrame(rows)


def preprocess_timing_collection(configs, selected_keys=None, data_dir=DATA_DIR,
                                  timing_dir=TIMING_DIR, overwrite=False):
    if selected_keys is None:
        selected_keys = list(configs)

    rows = []

    for key in selected_keys:
        cfg = configs[key]
        path = timing_cache_path(cfg, timing_dir=timing_dir)

        if path.exists() and not overwrite:
            status = "exists"
        else:
            path, _ = save_timing_curve_csv(
                cfg,
                data_dir=data_dir,
                timing_dir=timing_dir,
                overwrite=overwrite,
            )
            status = "written"

        rows.append({
            "dataset_key": key,
            "dataset_label": cfg["label"],
            "raw_timing_csv": str(path),
            "status": status,
        })

    table = pd.DataFrame(rows)
    return table


def refinef_local(x, resolution_factor=1, mode="smooth"):
    x = np.asarray(x, dtype=float)

    if len(x) == 0:
        return x

    if resolution_factor <= 0:
        raise ValueError("resolution_factor must be positive")

    m = max(1, int(round(len(x) * resolution_factor)))
    u = np.clip(np.arange(m) / resolution_factor, 0, len(x) - 1)

    if mode == "smooth":
        return np.interp(u, np.arange(len(x)), x)

    if mode == "const":
        return x[np.clip(np.floor(u + 0.5).astype(int), 0, len(x) - 1)]

    raise ValueError("mode must be 'smooth' or 'const'")


def smoothf_local(data, window=50, periodic=False):
    data = np.asarray(data, dtype=float)
    n = len(data)

    if n == 0:
        return data

    window = int(max(1, min(window, n)))
    kernel = np.ones(window, dtype=float)

    if periodic:
        left_width = window // 2
        right_width = window - 1 - left_width
        padded = np.pad(data, (left_width, right_width), mode="wrap")
        return np.convolve(padded, kernel / kernel.sum(), mode="valid")

    left = data[1:window + 1][::-1] if n > 1 else data
    right = data[-window - 1:-1][::-1] if n > 1 else data
    padded = np.concatenate([left, data, right])

    smoothed = np.convolve(padded, kernel / kernel.sum(), mode="same")
    return smoothed[len(left):len(left) + n]


def fill_nan_linear(y):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    ok = np.isfinite(y)

    if ok.sum() < 2:
        raise ValueError("Not enough finite timing values to interpolate.")

    return np.interp(x, x[ok], y[ok])


def prepare_timing_curve(positions_raw, timing_raw, resolution=10_000,
                         refine_factor=10, smooth_window=50,
                         timing_range=(60, 10), slice_stop=None,
                         centromeres_bp=None, chrom=None,
                         requested_chrom=None,
                         centromere_fill_value=None,
                         periodic=False):
    timing_filled = fill_nan_linear(timing_raw)

    timing = refinef_local(timing_filled, resolution_factor=refine_factor)
    timing = smoothf_local(timing, window=smooth_window, periodic=periodic)

    positions = refinef_local(positions_raw, resolution_factor=refine_factor)

    if slice_stop is not None:
        timing = timing[:slice_stop]
        positions = positions[:slice_stop]

    dx_kb = resolution / refine_factor / 1000.0
    timing_pre_rescale_before_centromere_override = np.asarray(timing, dtype=float).copy()

    if centromere_fill_value is None:
        if float(timing_range[0]) <= float(timing_range[1]):
            fill_value = float(np.nanmin(timing))
            fill_strategy = "pre_rescale_min_for_timing_min"
        else:
            fill_value = float(np.nanmax(timing))
            fill_strategy = "pre_rescale_max_for_timing_min"
    else:
        fill_value = centromere_fill_value
        fill_strategy = "explicit_pre_rescale"

    timing_pre_rescale, centromere_mask, centromere_overrides = (
        _apply_centromere_timing_override_array(
            timing,
            positions,
            dx_kb,
            centromeres_bp=centromeres_bp,
            chrom=chrom,
            requested_chrom=requested_chrom,
            fill_value=fill_value,
            fill_strategy=fill_strategy,
            stage="pre_rescale",
        )
    )

    timing_without_centromere_override = np.asarray(
        rescale(timing_pre_rescale_before_centromere_override, timing_range),
        dtype=float,
    )
    timing = np.asarray(rescale(timing_pre_rescale, timing_range), dtype=float)

    if not centromere_overrides.empty:
        rescaled_values = timing[centromere_mask]
        fill_value_after_rescale = (
            float(np.nanmin(rescaled_values)) if rescaled_values.size else np.nan
        )
        centromere_overrides = centromere_overrides.assign(
            fill_value_after_rescale=fill_value_after_rescale,
        )

    return {
        "positions_bp": positions,
        "positions_kb": positions / 1000.0,
        "timing_min": timing,
        "timing_min_before_centromere_override": timing_without_centromere_override,
        "timing_pre_rescale": timing_pre_rescale,
        "timing_pre_rescale_before_centromere_override": (
            timing_pre_rescale_before_centromere_override
        ),
        "dx_kb": dx_kb,
        "refine_factor": refine_factor,
        "smooth_window": smooth_window,
        "timing_range": timing_range,
        "centromere_override_mask": centromere_mask,
        "centromere_override_value": (
            float(centromere_overrides["fill_value_after_rescale"].iloc[0])
            if not centromere_overrides.empty else np.nan
        ),
        "centromere_override_value_pre_rescale": (
            float(centromere_overrides["fill_value"].iloc[0])
            if not centromere_overrides.empty else np.nan
        ),
        "centromere_overrides": centromere_overrides,
    }


def _normalise_centromere_intervals(centromeres_bp, chrom=None, requested_chrom=None):
    if centromeres_bp is None:
        return []

    if isinstance(centromeres_bp, dict):
        intervals = None
        for key in (chrom, requested_chrom):
            if key in centromeres_bp:
                intervals = centromeres_bp[key]
                break
        if intervals is None:
            return []
    else:
        intervals = centromeres_bp

    if len(intervals) == 0:
        return []

    if len(intervals) == 2 and all(np.isscalar(x) for x in intervals):
        intervals = [intervals]

    normalised = []
    for start_bp, end_bp in intervals:
        start_bp = int(start_bp)
        end_bp = int(end_bp)
        if end_bp <= start_bp:
            raise ValueError(f"Invalid centromere interval: {start_bp}-{end_bp}")
        normalised.append((start_bp, end_bp))

    return normalised


def _apply_centromere_timing_override_array(timing, positions, dx_kb,
                                            centromeres_bp, chrom=None,
                                            requested_chrom=None,
                                            fill_value=None,
                                            fill_strategy="pre_rescale_min",
                                            stage="pre_rescale"):
    timing = np.asarray(timing, dtype=float).copy()
    positions = np.asarray(positions, dtype=float)
    mask = np.zeros(timing.size, dtype=bool)

    intervals = _normalise_centromere_intervals(
        centromeres_bp,
        chrom=chrom,
        requested_chrom=requested_chrom,
    )
    if not intervals:
        return timing, mask, pd.DataFrame()

    if fill_value is None:
        fill_value = float(np.nanmin(timing))

    dx_bp = float(dx_kb) * 1000.0
    rows = []

    for start_bp, end_bp in intervals:
        interval_mask = (positions < end_bp) & ((positions + dx_bp) > start_bp)
        timing[interval_mask] = fill_value
        mask |= interval_mask
        rows.append({
            "chrom": chrom,
            "requested_chrom": requested_chrom or chrom,
            "start_bp": start_bp,
            "end_bp": end_bp,
            "fill_value": float(fill_value),
            "fill_strategy": fill_strategy,
            "stage": stage,
            "bins": int(interval_mask.sum()),
            "dx_kb": float(dx_kb),
        })

    return timing, mask, pd.DataFrame(rows)


def apply_centromere_timing_override(prepared, centromeres_bp, timing_range,
                                     chrom=None, requested_chrom=None,
                                     fill_value=None):
    if fill_value is None:
        fill_value = 0.5 * (float(timing_range[0]) + float(timing_range[1]))

    timing, mask, centromere_overrides = _apply_centromere_timing_override_array(
        prepared["timing_min"],
        prepared["positions_bp"],
        prepared["dx_kb"],
        centromeres_bp=centromeres_bp,
        chrom=chrom,
        requested_chrom=requested_chrom,
        fill_value=fill_value,
        fill_strategy="timing_range_midpoint",
        stage="post_rescale",
    )
    if not centromere_overrides.empty:
        centromere_overrides = centromere_overrides.assign(
            fill_value_after_rescale=float(fill_value),
        )

    prepared["timing_min_before_centromere_override"] = prepared["timing_min"]
    prepared["timing_min"] = timing
    prepared["centromere_override_mask"] = mask
    prepared["centromere_override_value"] = float(fill_value)
    prepared["centromere_override_value_pre_rescale"] = np.nan
    prepared["centromere_overrides"] = centromere_overrides

    return prepared["centromere_overrides"]


GEOMETRY_LABELS = {
    "torus": r"Torus $\mathbb{T}_L$",
    "interval": r"Interval $D=[0,L]$",
    "line": r"Full line $\mathbb{R}$",
    "halfline": r"Half-line $\mathbb{R}_+$",
}

MIN_EMPIRICAL_TAIL_COUNT = 10


def _higher_quantiles(x, q):
    x = np.sort(np.asarray(x, dtype=float), axis=0)
    n = x.shape[0]
    q = np.asarray(q, dtype=float)
    k = np.clip(np.ceil(q * n).astype(int) - 1, 0, n - 1)
    return x[k]


def _normalise_ignore_mask(ignore_mask, n, label="ignore_mask"):
    if ignore_mask is None:
        return np.zeros(n, dtype=bool)

    ignore_mask = np.asarray(ignore_mask, dtype=bool)

    if ignore_mask.ndim != 1 or ignore_mask.size != n:
        raise ValueError(f"{label} must be a 1D boolean array with length {n}.")

    if ignore_mask.all():
        raise ValueError(f"{label} excludes every position.")

    return ignore_mask


def _boundary_buffer_ignore_mask(positions_bp, dx_bp, buffer_bp):
    positions_bp = np.asarray(positions_bp, dtype=float)

    if positions_bp.ndim != 1 or positions_bp.size == 0:
        raise ValueError("positions_bp must be a non-empty 1D array.")

    buffer_bp = float(buffer_bp)
    dx_bp = float(dx_bp)

    if buffer_bp < 0 or dx_bp <= 0:
        raise ValueError("buffer_bp must be nonnegative and dx_bp must be positive.")

    domain_start_bp = float(positions_bp[0])
    domain_end_bp = float(positions_bp[-1] + dx_bp)
    domain_length_bp = domain_end_bp - domain_start_bp

    if 2.0 * buffer_bp >= domain_length_bp:
        raise ValueError(
            "The empirical interior buffer excludes the complete spatial domain."
        )

    target_start_bp = domain_start_bp + buffer_bp
    target_end_bp = domain_end_bp - buffer_bp
    mask = (
        (positions_bp < target_start_bp)
        | ((positions_bp + dx_bp) > target_end_bp)
    )

    return mask, target_start_bp, target_end_bp


def _width(sigma, L, vmin, geometry="torus"):
    sigma = np.asarray(sigma, dtype=float)

    if geometry == "torus":
        return np.minimum(2.0 * vmin * sigma, L)

    if geometry == "interval":
        return np.minimum(vmin * sigma, L)

    if geometry == "line":
        return 2.0 * vmin * sigma

    if geometry == "halfline":
        return vmin * sigma

    raise ValueError("geometry must be 'torus', 'interval', 'line', or 'halfline'")


def _torus_local_mass(I, lengths, dx=1.0):
    """
    Local initiation mass on a torus.

    I is assumed to be an initiation rate per grid site per minute.
    The default dx=1 therefore computes mass by summing fitted bin rates.
    Use dx only if I has first been converted to a density per physical unit.
    """
    I = np.asarray(I, dtype=float)
    lengths = np.atleast_1d(np.asarray(lengths, dtype=float))

    n = len(I)
    L = n * dx
    idx = np.arange(n)

    mass = np.concatenate([I, I])
    prefix = np.concatenate([[0.0], np.cumsum(mass)])

    out = np.empty_like(lengths, dtype=float)

    for k, r in enumerate(lengths):
        if r <= 0:
            out[k] = 0.0
            continue

        if r >= L:
            out[k] = dx * np.sum(I)
            continue

        u = r / dx
        q = int(np.floor(u))
        frac = u - q

        totals = dx * (prefix[idx + q] - prefix[idx])

        if frac > 0:
            left_aligned = totals + frac * dx * I[(idx + q) % n]
            right_aligned = totals + frac * dx * I[(idx - 1) % n]
            out[k] = min(np.min(left_aligned), np.min(right_aligned))
        else:
            out[k] = np.min(totals)

    return out


def _interval_local_mass(I, lengths, dx=1.0):
    """
    Local initiation mass on the finite interval D=[0,L].

    With the default dx=1, I is treated as a fitted rate per grid site per
    minute, and m_I,D(r) is obtained by summing rates over non-wrapping
    subintervals of D.
    """
    I = np.asarray(I, dtype=float)
    lengths = np.atleast_1d(np.asarray(lengths, dtype=float))

    n = len(I)
    L = n * dx
    prefix = np.concatenate([[0.0], np.cumsum(I)])
    total_mass = dx * np.sum(I)

    out = np.empty_like(lengths, dtype=float)

    for k, r in enumerate(lengths):
        if r <= 0:
            out[k] = 0.0
            continue

        if r >= L:
            out[k] = total_mass
            continue

        u = r / dx
        q = int(np.floor(u))
        frac = u - q

        if frac > 0:
            left_idx = np.arange(n - q)
            left_totals = dx * (
                prefix[left_idx + q] - prefix[left_idx]
            )
            left_totals = left_totals + frac * dx * I[left_idx + q]

            right_idx = np.arange(1, n - q + 1)
            right_totals = dx * (
                prefix[right_idx + q] - prefix[right_idx]
            )
            right_totals = right_totals + frac * dx * I[right_idx - 1]

            out[k] = min(np.min(left_totals), np.min(right_totals))
        else:
            idx = np.arange(n - q + 1)
            totals = dx * (prefix[idx + q] - prefix[idx])
            out[k] = np.min(totals)

    return out


def _periodic_extension_local_mass(I, lengths, dx=1.0):
    """
    Optional local mass on R using a periodic extension of the fitted landscape.

    This is useful for topology-only comparisons.
    """
    I = np.asarray(I, dtype=float)
    lengths = np.atleast_1d(np.asarray(lengths, dtype=float))

    n = len(I)
    L = n * dx
    total_mass = dx * np.sum(I)

    out = np.empty_like(lengths, dtype=float)

    for k, r in enumerate(lengths):
        if r <= 0:
            out[k] = 0.0
            continue

        n_periods = int(np.floor(r / L))
        remainder = r - n_periods * L

        out[k] = n_periods * total_mass

        if remainder > 1e-12:
            out[k] += _torus_local_mass(I, [remainder], dx=dx)[0]

    return out


def _local_mass(I, lengths, dx=1.0, geometry="torus", line_extension=None):
    if geometry == "torus":
        return _torus_local_mass(I, lengths, dx=dx)

    if geometry == "interval":
        return _interval_local_mass(I, lengths, dx=dx)

    if geometry == "line" and line_extension == "periodic":
        return _periodic_extension_local_mass(I, lengths, dx=dx)

    if geometry in {"line", "halfline"}:
        raise ValueError(
            "A finite initiation-rate array does not define a full-line or "
            "half-line local mass. Use geometry='interval', geometry='torus', "
            "or provide the explicit periodic full-line extension."
        )

    raise ValueError(f"Unsupported geometry: {geometry!r}.")


def completion_survival_exponent_curve(frates, vmin_grid=1.4, dx_grid=1.0, dx_kb=None,
                                       geometry="torus", rhs_max=None, num_t=4000,
                                       line_extension=None):
    """
    Build a certified lower approximation of the survival exponent.

    Parameters
    ----------
    frates : array
        Fitted initiation rates, interpreted as rates per grid site
        per minute.
    vmin_grid : float
        Fork speed in grid sites per minute.
    dx_grid : float
        Grid spacing in the same units used to define frates. In this notebook
        this should remain 1.0.
    dx_kb : float or None
        Physical size of one grid site in kb. This is stored only to
        label local-mass plots in kb.
    """
    I = np.asarray(frates, dtype=float)

    if I.ndim != 1:
        raise ValueError("frates must be a 1D array")

    if np.any(I < 0):
        raise ValueError("frates must be nonnegative")

    if vmin_grid <= 0 or dx_grid <= 0:
        raise ValueError("vmin_grid and dx_grid must be positive")

    L_grid = len(I) * dx_grid
    total_mass = dx_grid * np.sum(I)

    if total_mass <= 0:
        raise ValueError("Total initiation mass must be positive.")

    if rhs_max is None:
        rhs_max = np.log(1.0 / 1e-4)

    t_reference = L_grid / (2.0 * vmin_grid)
    t_max = t_reference + rhs_max / max(total_mass, 1e-12) + dx_grid / vmin_grid

    while True:
        t = np.linspace(0.0, t_max, num_t)
        r_grid = _width(t, L=L_grid, vmin=vmin_grid, geometry=geometry)

        r_unique, inv = np.unique(r_grid, return_inverse=True)
        m_unique = _local_mass(
            I,
            r_unique,
            dx=dx_grid,
            geometry=geometry,
            line_extension=line_extension,
        )
        mI = m_unique[inv]

        dt = np.diff(t)
        F = np.empty_like(t)
        F[0] = 0.0
        F[1:] = np.cumsum(mI[:-1] * dt)

        if F[-1] >= rhs_max:
            break

        t_max *= 2.0

    out = {
        "I": I,
        "L_grid": L_grid,
        "dx_grid": dx_grid,
        "dx_kb": dx_kb,
        "vmin_grid": float(vmin_grid),
        "geometry": geometry,
        "geometry_label": GEOMETRY_LABELS[geometry],
        "t": t,
        "r": r_grid,
        "mI": mI,
        "F": F,
    }

    if dx_kb is not None:
        out["L_kb"] = L_grid * dx_kb / dx_grid
        out["r_kb"] = r_grid * dx_kb / dx_grid
        out["vmin_kb_min"] = vmin_grid * dx_kb / dx_grid

    if geometry == "line":
        out["line_extension"] = line_extension

    return out


def completion_time_bound(eps, frates, vmin_grid=1.4, dx_grid=1.0,
                          dx_kb=None, geometry="torus", num_t=4000,
                          line_extension=None):
    eps = np.asarray(eps, dtype=float)

    if np.any((eps <= 0) | (eps >= 1)):
        raise ValueError("eps must lie strictly in (0, 1).")

    rhs = np.log(1.0 / eps)

    aux = completion_survival_exponent_curve(
        frates=frates,
        vmin_grid=vmin_grid,
        dx_grid=dx_grid,
        dx_kb=dx_kb,
        geometry=geometry,
        rhs_max=float(np.max(rhs)),
        num_t=num_t,
        line_extension=line_extension,
    )

    threshold_indices = np.searchsorted(aux["F"], rhs, side="left")
    left_indices = np.maximum(threshold_indices - 1, 0)
    left_times = aux["t"][left_indices]
    left_exponents = aux["F"][left_indices]
    left_masses = aux["mI"][left_indices]
    right_times = aux["t"][threshold_indices]

    positive_masses = left_masses > 0
    safe_masses = np.where(positive_masses, left_masses, 1.0)
    interpolated_times = left_times + (rhs - left_exponents) / safe_masses
    T_bound = np.where(positive_masses, interpolated_times, right_times)
    T_bound = np.minimum(np.maximum(T_bound, left_times), right_times)

    return T_bound, aux


def empirical_completion_curve(rep_times_per_sim, eps_grid, ignore_mask=None):
    """
    Empirical curve to compare with the uniform survival bound:
    max_x quantile_{1-eps} T(x), matching the L-infinity survival logic.
    """
    tau = np.asarray(rep_times_per_sim, dtype=float)

    if tau.ndim != 2:
        raise ValueError("rep_times_per_sim must have shape (n_sims, n_pos).")

    eps = np.asarray(eps_grid, dtype=float)
    tail_counts = tau.shape[0] * eps
    if np.any(tail_counts < MIN_EMPIRICAL_TAIL_COUNT - 1e-12):
        minimum_eps = MIN_EMPIRICAL_TAIL_COUNT / tau.shape[0]
        raise ValueError(
            f"Empirical completion quantiles require N * epsilon >= "
            f"{MIN_EMPIRICAL_TAIL_COUNT}; with N={tau.shape[0]}, use "
            f"epsilon >= {minimum_eps:g}."
        )

    ignore_mask = _normalise_ignore_mask(ignore_mask, tau.shape[1])
    tau = tau[:, ~ignore_mask]

    q = 1.0 - eps

    return _higher_quantiles(tau, q).max(axis=1)


def empirical_expected_time(rep_times_per_sim, ignore_mask=None):
    """
    Empirical max_x E[T(x)], matching the integrated uniform survival bound.
    """
    tau = np.asarray(rep_times_per_sim, dtype=float)

    if tau.ndim != 2:
        raise ValueError("rep_times_per_sim must have shape (n_sims, n_pos).")

    ignore_mask = _normalise_ignore_mask(ignore_mask, tau.shape[1])
    tau = tau[:, ~ignore_mask]

    return float(np.nanmax(np.nanmean(tau, axis=0)))


def expected_time_bound_from_curve(aux):
    """
    Integrate a one-sided survival estimate to bound the expected time.
    """
    t = np.asarray(aux["t"], dtype=float)
    F = np.asarray(aux["F"], dtype=float)
    mI = np.asarray(aux["mI"], dtype=float)

    survival_upper = np.exp(-F)
    dt = np.diff(t)
    body = float(np.sum(survival_upper[:-1] * dt))
    tail_rate = float(mI[-1])
    if not np.isfinite(tail_rate) or tail_rate <= 0:
        raise ValueError("The final local mass must be positive to bound the tail.")
    tail = float(survival_upper[-1] / tail_rate)
    total = body + tail

    return {
        "survival_upper": survival_upper,
        "expected_time_bound": total,
        "expected_time_bound_body": body,
        "expected_time_bound_tail": tail,
        "expected_time_bound_tail_fraction": tail / total if total > 0 else 0.0,
        "expected_time_tail_rate": tail_rate,
    }


def compare_completion_bounds(frates, rep_times_per_sim=None, fork_speed_grid=1.4,
                              dx_grid=1.0, dx_kb=None, eps_grid=None,
                              geometry="torus", num_t=4000,
                              line_extension=None, ignore_mask=None):
    if eps_grid is None:
        minimum_eps = 1e-4
        if rep_times_per_sim is not None:
            simulation_count = int(np.asarray(rep_times_per_sim).shape[0])
            if simulation_count <= 0:
                raise ValueError("rep_times_per_sim must contain at least one simulation.")
            minimum_eps = max(
                minimum_eps,
                MIN_EMPIRICAL_TAIL_COUNT / simulation_count,
            )
            if minimum_eps >= 0.99:
                raise ValueError(
                    "At least 11 simulations are required for an empirical "
                    "completion curve."
                )
        eps_grid = np.geomspace(minimum_eps, 0.99, 100)

    eps_grid = np.asarray(eps_grid, dtype=float)
    frates = np.asarray(frates, dtype=float)
    ignore_mask = _normalise_ignore_mask(ignore_mask, frates.size)

    T_theory, aux = completion_time_bound(
        eps=eps_grid,
        frates=frates,
        vmin_grid=fork_speed_grid,
        dx_grid=dx_grid,
        dx_kb=dx_kb,
        geometry=geometry,
        num_t=num_t,
        line_extension=line_extension,
    )

    out = {
        "eps": eps_grid,
        "T_theory": T_theory,
        "aux": aux,
        "geometry": geometry,
        "geometry_label": GEOMETRY_LABELS[geometry],
        "fork_speed_grid": fork_speed_grid,
        "dx_grid": dx_grid,
        "dx_kb": dx_kb,
        "ignore_mask": ignore_mask,
        "ignored_positions": int(ignore_mask.sum()),
        "used_positions": int((~ignore_mask).sum()),
        "theoretical_positions": int(frates.size),
        "empirical_used_positions": int((~ignore_mask).sum()),
    }

    out.update(expected_time_bound_from_curve(aux))

    if geometry == "line":
        out["line_extension"] = line_extension

    if dx_kb is not None:
        out["fork_speed_kb_min"] = fork_speed_grid * dx_kb / dx_grid

    if rep_times_per_sim is not None:
        simulation_count = int(np.asarray(rep_times_per_sim).shape[0])
        out["T_empirical"] = empirical_completion_curve(
            rep_times_per_sim,
            eps_grid=eps_grid,
            ignore_mask=ignore_mask,
        )
        out["simulation_count"] = simulation_count
        out["empirical_tail_counts"] = simulation_count * eps_grid
        out["minimum_empirical_tail_count"] = float(
            np.min(out["empirical_tail_counts"])
        )
        out["expected_time_empirical_pointwise"] = empirical_expected_time(
            rep_times_per_sim,
            ignore_mask=ignore_mask,
        )

    return out


def plot_replicated_fraction_map(rep_times_per_sim, positions_kb=None,
                                 nt=300, ylims=None, title=None):
    tau = np.asarray(rep_times_per_sim, dtype=float)

    if tau.ndim != 2:
        raise ValueError("rep_times_per_sim must have shape (n_sims, n_pos).")

    n_sims, n_pos = tau.shape

    if positions_kb is None:
        x = np.arange(n_pos)
        xlabel = "Position"
    else:
        x = np.asarray(positions_kb, dtype=float)
        xlabel = "Chromosome position (kb)"

    tmin = max(0.0, np.nanmin(tau))
    tmax = np.nanmax(tau)
    t_grid = np.linspace(tmin, tmax, nt)

    S = (tau[None, :, :] <= t_grid[:, None, None]).mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], t_grid[0], t_grid[-1]],
        vmin=0.0,
        vmax=1.0,
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"Replicated fraction $f(x,t)$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Replication time (min)")
    ax.set_title(title or r"Empirical replicated fraction $f(x,t)$")

    if ylims is not None:
        ax.set_ylim(*ylims)
    else:
        ax.invert_yaxis()

    plt.tight_layout()
    return fig, ax, x, t_grid, S


def plot_geometry_bounds(results_by_geometry, title=None, empiricalQ=True):
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for geometry, res in results_by_geometry.items():
        ax.plot(
            res["eps"],
            res["T_theory"],
            label=f"{GEOMETRY_LABELS[geometry]} bound",
        )

    if empiricalQ:
        first_res = next(iter(results_by_geometry.values()))

        if "T_empirical" in first_res:
            empirical_label = first_res.get("empirical_label", "empirical simulation")
            if "simulation_count" in first_res:
                empirical_label = (
                    f"{empirical_label} "
                    f"(N={first_res['simulation_count']:,}, "
                    f"min N epsilon={first_res['minimum_empirical_tail_count']:g})"
                )
            ax.plot(
                first_res["eps"],
                first_res["T_empirical"],
                linestyle="--",
                label=empirical_label,
            )

    ax.set_xscale("log")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel(r"Completion time $T_\varepsilon$ (min)")
    ax.set_title(title or "Completion-time bound")
    ax.legend()
    plt.tight_layout()

    return fig, ax


def plot_expected_time_bounds(results_by_geometry, title=None):
    items = list(results_by_geometry.items())
    labels = {
        "torus": "Torus",
        "interval": "D=[0,L]",
        "line": "R",
        "halfline": "R+",
    }
    theory = np.array([res["expected_time_bound"] for _, res in items], dtype=float)
    empirical = np.array([
        res.get("expected_time_empirical_pointwise", np.nan)
        for _, res in items
    ], dtype=float)
    valid = np.isfinite(empirical) & np.isfinite(theory)

    if not np.any(valid):
        raise ValueError("Expected-time plot requires empirical pointwise simulation estimates.")

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(empirical[valid], theory[valid], color="black", zorder=3)

    for (geometry, res), x_val, y_val, keep in zip(items, empirical, theory, valid):
        if keep:
            ax.annotate(
                labels.get(geometry, res["geometry_label"]),
                xy=(x_val, y_val),
                xytext=(6, 6),
                textcoords="offset points",
            )

    lo = float(min(np.min(empirical[valid]), np.min(theory[valid])))
    hi = float(max(np.max(empirical[valid]), np.max(theory[valid])))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo = max(0.0, lo - pad)
    hi = hi + pad
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="0.5")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel(r"Empirical $\max_x \mathbb{E}[T(x)]$ (min)")
    ax.set_ylabel("Theoretical expected-time bound (min)")
    ax.set_title(title or "Expected local replication-time bound")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    return fig, ax


def expected_time_pair_table(results):
    labels = {
        "torus": "Torus",
        "interval": "D=[0,L]",
        "line": "R",
        "halfline": "R+",
    }
    rows = []

    for key, result in results.items():
        cfg = result["config"]

        for geometry, bound_result in result["bounds"].items():
            rows.append({
                "dataset_key": key,
                "cell_line": cfg["cell_line"],
                "point_label": cfg.get("point_label", cfg.get("short_label", cfg["label"])),
                "dataset_label": cfg["label"],
                "geometry": geometry,
                "geometry_label": labels.get(geometry, bound_result.get("geometry_label", geometry)),
                "E_empirical_pointwise_min": bound_result.get("expected_time_empirical_pointwise", np.nan),
                "E_theory_min": bound_result["expected_time_bound"],
                "empirical_interior_buffer_bp": bound_result.get(
                    "empirical_interior_buffer_bp",
                    np.nan,
                ),
                "expected_time_comparison_certified": bound_result.get(
                    "expected_time_comparison_certified",
                    True,
                ),
            })

    return pd.DataFrame(rows)


def _resolve_axis_limits(requested, fallback):
    if requested is None:
        return fallback

    if len(requested) != 2:
        raise ValueError("Axis limits must be a (min, max) pair.")

    lo = fallback[0] if requested[0] is None else requested[0]
    hi = fallback[1] if requested[1] is None else requested[1]

    return (lo, hi)


def plot_expected_time_pair_scatter(results, title=None, label_col="point_label",
                                    xlims=None, ylims=None,
                                    equal_aspect=True, xlabel=None, ylabel=None):
    table = expected_time_pair_table(results)
    xcol = "E_empirical_pointwise_min"
    ycol = "E_theory_min"
    valid = np.isfinite(table[xcol]) & np.isfinite(table[ycol])
    plot_table = table.loc[valid].copy()

    if plot_table.empty:
        raise ValueError("Expected-time scatter requires empirical pointwise simulation estimates.")

    if label_col not in plot_table.columns:
        raise ValueError(f"label_col={label_col!r} is not a column in the expected-time table.")

    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    plot_table["_scatter_label"] = plot_table[label_col].astype(str)

    for scatter_label, group in plot_table.groupby("_scatter_label", sort=False):
        ax.scatter(group[xcol], group[ycol], label=scatter_label, zorder=3)

    lo = float(min(plot_table[xcol].min(), plot_table[ycol].min()))
    hi = float(max(plot_table[xcol].max(), plot_table[ycol].max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo = max(0.0, lo - pad)
    hi = hi + pad

    auto_limits = (lo, hi)
    xlims = _resolve_axis_limits(xlims, auto_limits)
    ylims = _resolve_axis_limits(ylims, auto_limits)

    ref_lo = min(xlims[0], ylims[0])
    ref_hi = max(xlims[1], ylims[1])
    ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], linestyle="--", linewidth=1, color="0.5")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel(xlabel or r"Empirical $\max_x \mathbb{E}[T(x)]$ (min)")
    ax.set_ylabel(ylabel or "Theoretical expected-time bound (min)")
    ax.set_title(title or "Expected local replication-time bound")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.legend(title="Dataset")
    plt.tight_layout()

    return fig, ax, table


def plot_bound_tightness(result, title=None):
    if "T_empirical" not in result:
        raise ValueError("This result has no empirical simulation curve.")

    eps = result["eps"]
    theory = result["T_theory"]
    empirical = result["T_empirical"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(eps, theory / empirical)
    ax.axhline(1.0, linestyle="--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel(r"Theoretical / empirical $T_\varepsilon$")
    ax.set_title(title or f"Bound tightness: {result['geometry_label']}")
    plt.tight_layout()

    return fig, ax


def plot_local_mass(result, title=None):
    aux = result["aux"]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = aux["r_kb"] if "r_kb" in aux else aux["r"]
    xlabel = "Arc or interval length r (kb)" if "r_kb" in aux else "Arc or interval length r (grid units)"
    ax.plot(x, aux["mI"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Local initiation mass $m_I(r)$")
    ax.set_title(title or f"Local initiation-mass function: {aux['geometry_label']}")
    plt.tight_layout()

    return fig, ax


def run_single_dataset(cfg, fork_speed_kb_min=1.4, sim_number=1000,
                       refine_factor=10, smooth_window=50,
                       timing_range=(60, 10), eps_grid=None,
                       slice_stop=None, num_t_bound=4000,
                       max_rep_time=2000.0, timing_cache_mode="auto",
                       timing_dir=TIMING_DIR,
                       overwrite_timing_cache=False,
                       centromeres_bp=None,
                       centromere_fill_value=None,
                       empirical_interior_buffer_bp=None):
    if eps_grid is None:
        simulation_count = int(sim_number)
        if simulation_count <= 0:
            raise ValueError("sim_number must be a positive integer.")
        minimum_eps = max(1e-4, MIN_EMPIRICAL_TAIL_COUNT / simulation_count)
        if minimum_eps >= 0.99:
            raise ValueError(
                "At least 11 simulations are required for an empirical "
                "completion curve."
            )
        eps_grid = np.geomspace(minimum_eps, 0.99, 100)

    bound_geometries = tuple(cfg["bound_geometries"])

    if "interval" in bound_geometries:
        if cfg["fit_periodic"] or cfg["sim_periodic"]:
            raise ValueError(
                "Interval bounds require non-periodic fitting and simulation."
            )
        if cfg.get("boundary_conditions") != "zero_inflow":
            raise ValueError(
                "Interval bounds require boundary_conditions='zero_inflow'."
            )

    if "torus" in bound_geometries:
        if not cfg["fit_periodic"] or not cfg["sim_periodic"]:
            raise ValueError(
                "Torus bounds require periodic fitting and simulation."
            )

    if "line" in bound_geometries:
        if cfg.get("line_extension") != "periodic":
            raise ValueError(
                "Full-line bounds from a finite profile require "
                "line_extension='periodic'."
            )
        if not cfg["sim_periodic"] and empirical_interior_buffer_bp is None:
            raise ValueError(
                "A non-periodic simulation compared with the full-line bound "
                "requires empirical_interior_buffer_bp."
            )

    print(f"Processing: {cfg['label']}")
    print(f"Region: {cfg['chrom']}:{cfg['start']}-{cfg['end']}")
    print(f"Model periodicity: fit={cfg['fit_periodic']}, simulation={cfg['sim_periodic']}")
    if "interval" in bound_geometries:
        print("Boundary conditions: zero incoming forks at x=0 and x=L")
    print(f"Bound geometries: {list(bound_geometries)}")

    positions_raw, timing_raw = get_timing_curve_from_config(
        cfg,
        timing_dir=timing_dir,
        cache_mode=timing_cache_mode,
        overwrite=overwrite_timing_cache,
    )

    prepared = prepare_timing_curve(
        positions_raw=positions_raw,
        timing_raw=timing_raw,
        resolution=cfg["resolution"],
        refine_factor=refine_factor,
        smooth_window=smooth_window,
        timing_range=timing_range,
        slice_stop=slice_stop,
        centromeres_bp=centromeres_bp,
        chrom=cfg["chrom"],
        requested_chrom=cfg.get("requested_chrom", cfg["chrom"]),
        centromere_fill_value=centromere_fill_value,
        periodic=cfg["fit_periodic"],
    )

    centromere_overrides = prepared["centromere_overrides"]
    if not centromere_overrides.empty:
        overridden_bins = int(centromere_overrides["bins"].sum())
        fill_value = float(centromere_overrides["fill_value"].iloc[0])
        fill_value_after_rescale = float(
            centromere_overrides["fill_value_after_rescale"].iloc[0]
        )
        strategy = str(centromere_overrides["fill_strategy"].iloc[0])
        if strategy == "pre_rescale_min_for_timing_min":
            value_label = "pre-rescale minimum for timing minimum"
        elif strategy == "pre_rescale_max_for_timing_min":
            value_label = "pre-rescale maximum for timing minimum"
        else:
            value_label = "pre-rescale value"
        print(
            f"Centromere override: set {overridden_bins:,} bins "
            f"to {value_label} {fill_value:g} "
            f"({fill_value_after_rescale:g} min after rescaling)"
        )

    timedata = prepared["timing_min"]
    dx_kb = prepared["dx_kb"]
    centromere_ignore_mask = prepared.get("centromere_override_mask")
    if centromere_ignore_mask is None:
        centromere_ignore_mask = np.zeros(timedata.size, dtype=bool)
    centromere_ignore_mask = np.asarray(centromere_ignore_mask, dtype=bool)

    boundary_ignore_mask = np.zeros(timedata.size, dtype=bool)
    target_start_bp = float(prepared["positions_bp"][0])
    target_end_bp = float(prepared["positions_bp"][-1] + dx_kb * 1000.0)
    if empirical_interior_buffer_bp is not None:
        boundary_ignore_mask, target_start_bp, target_end_bp = (
            _boundary_buffer_ignore_mask(
                prepared["positions_bp"],
                dx_bp=dx_kb * 1000.0,
                buffer_bp=empirical_interior_buffer_bp,
            )
        )

    completion_ignore_mask = centromere_ignore_mask | boundary_ignore_mask

    fork_speed_grid = fork_speed_kb_min / dx_kb

    print(f"Grid spacing: {dx_kb:g} kb per simulation site")
    print(f"Fork speed: {fork_speed_kb_min:g} kb/min = {fork_speed_grid:g} grid sites/min")

    frates = rfit(
        "replication_timing",
        "firing_rate",
        source=timedata,
        fork_speed=fork_speed_grid,
        maxiter=10,
        fit_step=2,
        perQ=cfg["fit_periodic"],
    )

    simres = rsim(
        ori_rate=frates,
        fork_speed=fork_speed_grid,
        sim_number=sim_number,
        perQ=cfg["sim_periodic"],
        time_statsQ=True,
        time_stats_xtQ=False,
        time_stats_densQ=None,
        max_rep_time=max_rep_time,
    )

    rep_times_per_sim = simres["rep_times_per_sim"]

    if centromere_ignore_mask.any():
        print(
            f"Empirical target: ignoring {int(centromere_ignore_mask.sum()):,} "
            "centromere bins in empirical target summaries; the theoretical "
            "bound uses the complete fitted domain"
        )

    if boundary_ignore_mask.any():
        print(
            f"Empirical target: chromosome interior "
            f"{target_start_bp / 1e6:g}-{target_end_bp / 1e6:g} Mb "
            f"with a {float(empirical_interior_buffer_bp) / 1e6:g} Mb "
            "buffer at each end"
        )

    bounds = {}

    for geometry in bound_geometries:
        bound_result = compare_completion_bounds(
            frates=frates,
            rep_times_per_sim=rep_times_per_sim,
            fork_speed_grid=fork_speed_grid,
            dx_grid=1.0,
            dx_kb=dx_kb,
            eps_grid=eps_grid,
            geometry=geometry,
            num_t=num_t_bound,
            line_extension=cfg.get("line_extension"),
            ignore_mask=completion_ignore_mask,
        )
        bound_result["empirical_interior_buffer_bp"] = (
            None if empirical_interior_buffer_bp is None
            else float(empirical_interior_buffer_bp)
        )
        bound_result["empirical_target_start_bp"] = target_start_bp
        bound_result["empirical_target_end_bp"] = target_end_bp
        bound_result["boundary_ignored_positions"] = int(boundary_ignore_mask.sum())
        bound_result["centromere_ignored_positions"] = int(centromere_ignore_mask.sum())

        if geometry == "line" and empirical_interior_buffer_bp is not None:
            validity_horizon_min = (
                float(empirical_interior_buffer_bp) / 1000.0 / fork_speed_kb_min
            )
            comparison_max_min = float(np.nanmax(bound_result["T_theory"]))
            if "T_empirical" in bound_result:
                comparison_max_min = max(
                    comparison_max_min,
                    float(np.nanmax(bound_result["T_empirical"])),
                )
            bound_result["interior_validity_horizon_min"] = validity_horizon_min
            bound_result["interior_comparison_max_min"] = comparison_max_min
            bound_result["interior_horizon_valid"] = (
                comparison_max_min <= validity_horizon_min
            )
            bound_result["empirical_label"] = "empirical chromosome interior"
            bound_result["expected_time_comparison_certified"] = False

            if not bound_result["interior_horizon_valid"]:
                raise ValueError(
                    f"The full-line/interior comparison reaches "
                    f"{comparison_max_min:g} min, beyond the buffer-validity "
                    f"horizon {validity_horizon_min:g} min. Increase "
                    "empirical_interior_buffer_bp or reduce the reported time range."
                )

            print(
                f"Interior comparison horizon: {validity_horizon_min:g} min; "
                f"largest compared T_epsilon: {comparison_max_min:g} min"
            )

        bounds[geometry] = bound_result

    return {
        "config": cfg,
        "positions_raw": positions_raw,
        "timing_raw": timing_raw,
        "prepared": prepared,
        "timedata": timedata,
        "frates": frates,
        "simres": simres,
        "rep_times_per_sim": rep_times_per_sim,
        "bounds": bounds,
        "centromere_overrides": centromere_overrides,
        "centromere_ignore_mask": centromere_ignore_mask,
        "boundary_ignore_mask": boundary_ignore_mask,
        "completion_ignore_mask": completion_ignore_mask,
        "empirical_target_mask": ~completion_ignore_mask,
        "empirical_interior_buffer_bp": empirical_interior_buffer_bp,
        "empirical_target_start_bp": target_start_bp,
        "empirical_target_end_bp": target_end_bp,
        "fork_speed_kb_min": fork_speed_kb_min,
        "fork_speed_grid": fork_speed_grid,
        "sim_number": sim_number,
        "max_rep_time": max_rep_time,
        "timing_cache_mode": timing_cache_mode,
        "timing_cache_path": str(timing_cache_path(cfg, timing_dir=timing_dir)),
        "eps_grid": eps_grid,
    }


def _positive_percentile_limits(values, percentiles=(1, 99.5), log_pad_fraction=0.12):
    if percentiles is None:
        return (None, None)

    finite_positive = np.asarray(values, dtype=float)
    finite_positive = finite_positive[np.isfinite(finite_positive) & (finite_positive > 0)]

    if finite_positive.size == 0:
        return (None, None)

    lo, hi = np.percentile(finite_positive, percentiles)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0 or hi <= 0:
        return (None, None)

    if lo == hi:
        return (lo / 10.0, hi * 10.0)

    log_lo, log_hi = np.log10([lo, hi])
    pad = log_pad_fraction * max(log_hi - log_lo, 1e-12)

    return (10.0 ** (log_lo - pad), 10.0 ** (log_hi + pad))


def make_standard_plots(result, save_figures=True,
                        initiation_xlims=(None, None),
                        initiation_ylims=None,
                        initiation_ylim_percentiles=(1, 99.5),
                        initiation_invert_y=False):
    cfg = result["config"]
    prefix = safe_filename(cfg["short_label"])

    positions_kb = result["prepared"]["positions_kb"]
    timedata = result["timedata"]
    frates = result["frates"]
    simres = result["simres"]
    bounds = result["bounds"]

    periodic_plot = bool(cfg.get("fit_periodic") and cfg.get("sim_periodic"))
    if periodic_plot and positions_kb.size:
        closed_positions_kb = np.append(
            positions_kb,
            positions_kb[-1] + result["prepared"]["dx_kb"],
        )
        closed_timedata = np.append(timedata, timedata[0])
        closed_frates = np.append(frates, frates[0])
        closed_simulation_timing = np.append(
            simres["replication_timing"],
            simres["replication_timing"][0],
        )
    else:
        closed_positions_kb = positions_kb
        closed_timedata = timedata
        closed_frates = frates
        closed_simulation_timing = simres["replication_timing"]

    if initiation_ylims is None:
        initiation_ylims = _positive_percentile_limits(
            frates,
            percentiles=initiation_ylim_percentiles,
        )

    fig, ax = plotf(
        closed_frates,
        logyQ=True,
        x_array=closed_positions_kb,
        invyQ=initiation_invert_y,
        xlims=initiation_xlims,
        ylims=initiation_ylims,
        xtitle="Chromosome position (kb)",
        ytitle="Initiation rate",
        labels=[cfg["label"]],
        saveQ=False,
        showQ=False,
        return_handles=True,
    )
    if save_figures:
        fig.savefig(FIGURE_DIR / f"{prefix}_initiation_rate.pdf", bbox_inches="tight")
    plt.show()


    fig, ax = plotf(
        closed_timedata,
        closed_simulation_timing,
        x_array=closed_positions_kb,
        invyQ=False,
        xtitle="Chromosome position (kb)",
        ytitle="Replication timing (min)",
        labels=["Repli-seq", "Simulation"],
        saveQ=False,
        showQ=False,
        return_handles=True,
    )
    if save_figures:
        fig.savefig(FIGURE_DIR / f"{prefix}_timing_repliseq_vs_simulation.pdf", bbox_inches="tight")
    plt.show()


    fig, ax = plot_geometry_bounds(
        bounds,
        title=f"{cfg['label']}: theoretical bound versus simulation",
        empiricalQ=True,
    )
    if save_figures:
        fig.savefig(FIGURE_DIR / f"{prefix}_bound_vs_simulation.pdf", bbox_inches="tight")
    plt.show()


def run_dataset_collection(configs, selected_keys, **kwargs):
    """
    Optional batch helper. Use this only when you are ready to loop over several datasets.
    """
    results = {}

    for key in selected_keys:
        results[key] = run_single_dataset(configs[key], **kwargs)

    return results


def completion_summary_table(results, eps_values=(0.1, 0.05, 0.01)):
    rows = []

    for key, result in results.items():
        cfg = result["config"]

        for geometry, bound_result in result["bounds"].items():
            eps_grid = bound_result["eps"]

            for eps in eps_values:
                row = {
                    "dataset_key": key,
                    "dataset_label": cfg["label"],
                    "geometry": geometry,
                    "eps": eps,
                    "dx_kb": result.get("prepared", {}).get("dx_kb", np.nan),
                    "fork_speed_kb_min": result.get("fork_speed_kb_min", np.nan),
                    "fork_speed_grid_per_min": result.get("fork_speed_grid", np.nan),
                    "T_theory_min": np.interp(eps, eps_grid, bound_result["T_theory"]),
                    "empirical_interior_buffer_bp": bound_result.get(
                        "empirical_interior_buffer_bp",
                        np.nan,
                    ),
                    "interior_validity_horizon_min": bound_result.get(
                        "interior_validity_horizon_min",
                        np.nan,
                    ),
                    "interior_horizon_valid": bound_result.get(
                        "interior_horizon_valid",
                        True,
                    ),
                }

                if "T_empirical" in bound_result:
                    simulation_count = int(bound_result["simulation_count"])
                    row["N_simulations"] = simulation_count
                    row["N_epsilon"] = simulation_count * eps
                    row["T_empirical_min"] = np.interp(
                        eps,
                        eps_grid,
                        bound_result["T_empirical"],
                    )
                    row["theory_over_empirical"] = row["T_theory_min"] / row["T_empirical_min"]

                rows.append(row)

    return pd.DataFrame(rows)


def expected_time_summary_table(results):
    rows = []

    for key, result in results.items():
        cfg = result["config"]

        for geometry, bound_result in result["bounds"].items():
            row = {
                "dataset_key": key,
                "dataset_label": cfg["label"],
                "geometry": geometry,
                "dx_kb": result.get("prepared", {}).get("dx_kb", np.nan),
                "fork_speed_kb_min": result.get("fork_speed_kb_min", np.nan),
                "fork_speed_grid_per_min": result.get("fork_speed_grid", np.nan),
                "E_theory_min": bound_result["expected_time_bound"],
                "tail_fraction": bound_result["expected_time_bound_tail_fraction"],
                "empirical_interior_buffer_bp": bound_result.get(
                    "empirical_interior_buffer_bp",
                    np.nan,
                ),
                "expected_time_comparison_certified": bound_result.get(
                    "expected_time_comparison_certified",
                    True,
                ),
            }

            if "expected_time_empirical_pointwise" in bound_result:
                row["E_empirical_pointwise_min"] = bound_result["expected_time_empirical_pointwise"]
                row["theory_over_empirical_pointwise"] = (
                    row["E_theory_min"] / row["E_empirical_pointwise_min"]
                )

            rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "DATA_DIR",
    "FIGURE_DIR",
    "GEOMETRY_LABELS",
    "STANDARD_CHROMS",
    "S_PHASE_BINS",
    "TIMING_DIR",
    "available_cell_lines_from_data",
    "build_chromosome_configs",
    "build_chromosome_configs_for_cell_lines",
    "build_periodic_interval_configs",
    "compare_completion_bounds",
    "completion_summary_table",
    "completion_survival_exponent_curve",
    "completion_time_bound",
    "compute_timing_curve",
    "compute_timing_curve_from_config",
    "empirical_completion_curve",
    "empirical_expected_time",
    "etaf",
    "etaferr",
    "expected_time_bound_from_curve",
    "expected_time_pair_table",
    "expected_time_summary_table",
    "fast_shift_add",
    "fill_nan_linear",
    "get_bigwig_chrom_sizes",
    "get_timing_curve_from_config",
    "hms",
    "load_timing_curve_csv",
    "make_standard_plots",
    "map_firing_timing",
    "map_timing_firing",
    "mean_squared_error",
    "plot_bound_tightness",
    "plot_expected_time_bounds",
    "plot_expected_time_pair_scatter",
    "plot_geometry_bounds",
    "plot_local_mass",
    "plot_replicated_fraction_map",
    "plotf",
    "prepare_timing_curve",
    "preprocess_timing_collection",
    "read_bigwig_bin_mean",
    "refinef_local",
    "require_chrom_name",
    "rescale",
    "resolve_chrom_name",
    "rfit",
    "rsim",
    "run_dataset_collection",
    "run_single_dataset",
    "safe_filename",
    "save_timing_curve_csv",
    "sigmoid",
    "simulate_replication",
    "smoothf_local",
    "timing_cache_path",
    "timing_cache_stem",
    "timing_cache_table",
    "timing_curve_table",
]
