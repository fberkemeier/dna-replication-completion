"""Core helpers for the replication completion-bounds notebook."""

import heapq
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


__all__ = [
    "plotf",
    "rescale",
    "rfit",
    "rsim",
    "simulate_replication",
]
