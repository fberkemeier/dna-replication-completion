"""High-level fitting wrappers and mapper registry."""

import os
import math
import time

import numpy as np

from .constants import mapt
from .io import _resolve_data_path, savecsv
from .mapping import map_directionality_timing, map_firing_timing, map_timing_directionality
from .simulation import rsim
from .utils import etaferr, mean_squared_error

def map_timing_firing(
    observed_timing,
    fork_speed=1.4,
    fork_speeds=None,
    stall_rate=0.,
    tau=np.inf,
    resolution=1.0,
    neighbourhood_range=2000,
    fit_step=0.6,
    maxiter=10,
    sim_number=1000,
    err_threshold=15.0,
    deviation_threshold=None,
    verbose=True,
    perQ=False,
    chr_number=1,
    save_errQ=False
):

    mse_list = []

    T = np.asarray(observed_timing, dtype=float)
    n = len(T)

    fork_speed_per_bin = fork_speed / resolution

    # initial guess for firing rate
    x0 = (math.pi / (4.0 * fork_speed_per_bin)) * np.maximum(T, 1e-80) ** (-2)
    x = x0

    lm = int(1000 / resolution)
    if perQ:
        crop = slice(None)
    else:
        crop = slice(lm, -lm) if 2 * lm < n else slice(None)
    floor = 10.0 ** (-err_threshold)

    def update_firing_rate(observed, predicted, current):
        t_safe = np.maximum(observed, 1e-80)
        p_safe = np.maximum(predicted, 1e-80)
        ratio  = p_safe / t_safe

        updated = current * (ratio ** fit_step)
        updated = np.maximum(updated, floor)

        if deviation_threshold is not None:
            mse = np.abs(predicted - observed)
            close = mse <= deviation_threshold
            updated[close] = current[close]

        return updated

    # initial prediction
    y = map_firing_timing(
        x,
        fork_speed=fork_speed,
        resolution=resolution,
        neighbourhood_range=neighbourhood_range,
        perQ=perQ,
    )
    mse = mean_squared_error(T[crop], y[crop])

    if stall_rate > 0. or fork_speeds is not None:
        maxiter0 = 0
    else:
        maxiter0 = maxiter

    start_time = time.time()

    for it in range(1, maxiter0 + 1):
        x = update_firing_rate(T, y, x)
        y = map_firing_timing(
            x,
            fork_speed=fork_speed,
            resolution=resolution,
            neighbourhood_range=neighbourhood_range,
            perQ=perQ
        )
        mse = mean_squared_error(T[crop], y[crop])

        if verbose:
            etaferr(it=it, maxiter=maxiter0, start_time=start_time, err=mse)
            mse_list.append(mse)
            

    # Stalling models
    if stall_rate > 0. or fork_speeds is not None:

        alpha = fit_step
        iterations = maxiter
    
        rate_min = 1e-9
        rate_max = None
        T_target = np.asarray(observed_timing, float)
        
        rates = np.clip(np.asarray(x, float), rate_min, None)
        #eff_f = 0.17
        #rates = (math.pi / (4.0 * 10 * fork_speed_per_bin)) * np.maximum(T, 1e-80) ** (-2)
        
        if rate_max is None:
            pos = rates[rates > 0]
            rate_max = (10.0 * np.median(pos)) if pos.size else 1.0
    
        hist_mse = []
        T_pred = None
        start_time = time.time()
    
        for k in range(iterations):
            simres = rsim(
                rates,
                sim_number=sim_number,
                fork_speed=fork_speed,
                fork_speeds=fork_speeds,
                stall_rate=stall_rate,
                tau=tau,
                resolution=resolution,
                verbose=False,
                perQ=perQ
            )
            T_pred = simres['replication_timing']
    
            eps = 1e-12
            ratio = (T_pred + eps) / (T_target + eps)
            rates *= np.power(ratio, alpha)
            np.clip(rates, rate_min, rate_max, out=rates)
    
            mse = T_pred - T_target
            mse = float(np.mean(mse**2))
            hist_mse.append(mse)
    
            if verbose:
                etaferr(it=k+1, maxiter=iterations, start_time=start_time, err=mse)
                mse_list.append(mse)

        x = rates

    #np.savetxt(f"data/fit_error/mse_chr{chr_number}.txt", mse_list, fmt="%.6f")

    if save_errQ:
        savecsv(arr=mse_list, chr_number=chr_number, stitle=f'fit_error', outpath="fit_error")
        
    return x

class DataTypeMapper:

    def __init__(self):
        # map[from_type][to_type] = function
        self._registry = {}

    def register(self, from_type: str, to_type: str, func):

        if from_type not in self._registry:
            self._registry[from_type] = {}

        self._registry[from_type][to_type] = func

    def get(self, from_type: str, to_type: str):

        try:
            return self._registry[from_type][to_type]
        except KeyError:
            raise KeyError(
                f"No mapping registered from '{from_type}' to '{to_type}'. "
                "Use mapper.register(from_type, to_type, func) to add one."
            )

    def map(self, from_type: str, to_type: str, data, **kwargs):

        func = self.get(from_type, to_type)
        return func(data, **kwargs)

    def describe(self):

        print("Available data-type mappings:")
        for src, d in self._registry.items():
            for dst in d:
                print(f"  {src}  â†’  {dst}")

rmap = DataTypeMapper()
rmap.register("firing_rate", "replication_timing", map_firing_timing)
rmap.register("replication_timing", "firing_rate", map_timing_firing)
rmap.register("replication_timing", "fork_directionality", map_timing_directionality)
rmap.register("fork_directionality", "replication_timing", map_directionality_timing)

def rfit(
    d1='firing_rate',
    d2='replication_timing',
    cell_line='H1', chr_number=1,
    fork_speed=1.4, resolution=1.,
    source=[],
    saveQ=False,
    **kwargs
):

    if any(source):
        data1 = source
    else:
        data1 = np.loadtxt(_resolve_data_path(f"{mapt[d1]}/{mapt[d1]}_{cell_line}_chr{chr_number}.txt"), dtype=float)
    data2 = rmap.map(d1, d2, data1, fork_speed=fork_speed, resolution=resolution, **kwargs)

    if saveQ:
        out = _resolve_data_path(f"{mapt[d2]}/{mapt[d2]}_{cell_line}_chr{chr_number}.txt", for_write=True)
        os.makedirs(out.parent, exist_ok=True)
        np.savetxt(out, data2)

    return data2
