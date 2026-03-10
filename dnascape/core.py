# Auto-generated from dnascape/DNAscape_v0.1.3.ipynb

# %% [markdown]
# # DNAscape
# Toolkit for simulating and mapping DNA replication kinetics.
#
# Build 0.1.3:
#
# - FIX PERIODIC FIT ON SMALL DOMAINS!!!
# - Add misfits examples
# - Optimize genes/sequence imports
# - Add circular plot for periodic
# - Option to export as bedgraph, bigwig (and Genome browser examples)
# - Add time-dependent initiation and fork speed
# - Add ETA to map firing to timing
# - Add example with Repli-seq and Ini-seq/SNS-seq
# - Change global style (colours and text)
# - Make it accept both csv and bedgraph
# - Add quicker responsiveness in rfit
# - Make the gene plot a big feature highlight
# - Make stall rate take also arrays
# - Add stochastic and deterministic options for both I and v
# - Add function option for I and v
# - Add possibility of fixed (frozen) regions for I and v (?)
# - Add option for sequence display (colour coded, with gene plots)
# - Add bash option for arguments
# - Add heatmap plotting function
# - Add observed initiation, fork speed, and coalescence for heatmaps
# - Add title option to all plot functions
# - Add log scale options to heatmap plot function
# - Include StreamLit application for uploading, downloading, and zoom in verison
# - Optimize all code (rsim, options for events/frep), standardize, and comment where needed (minimal)
# - Create speed benchmarks

# %% [markdown]
# ## Imports and configuration

# %%
import math
import os
import time
import random
import heapq
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from scipy import stats
from scipy.integrate import cumulative_trapezoid
from scipy.stats import gaussian_kde, mode
from scipy.ndimage import zoom
import gzip
import re
import tempfile
from math import erf, sqrt, exp, pi
from collections import defaultdict
from scipy.fft import rfft, irfft

for folder in (
    Path("figures"),
    Path("data/firing_rate"),
    Path("data/replication_timing"),
    Path("data/fork_directionality"),
    Path("data/efficiency"),
    Path("data/output"),
    Path("data/fit_error"),
    Path("data/region"),
    Path("data/other"),
):
    folder.mkdir(parents=True, exist_ok=True)

# Colours
# Red: a10000ff
# Blue: 2e7aa0ff

# %% [markdown]
# ## Utility functions

# %% [markdown]
# ### Data management

# %%
def load(d='replication_timing', cell_line='H1', chr_number=1, example='hESC'):
    if d != 'example':
        data = np.loadtxt(f"data/{mapt[d]}/{mapt[d]}_{cell_line}_chr{chr_number}.txt", dtype=float)
    else:
        data = np.loadtxt(f"data/{mapt[d]}/{example}_chr{chr_number}.txt", dtype=float)
    return data

# %%
def loadcsv(d='replication_timing', cell_line='H1', chr_number=1, example='hESC', region='region'):
    if d == 'example':
        fname = f"data/{mapt[d]}/{example}.csv"
    elif d == 'region':
        fname = f"data/{mapt[d]}/{region}.csv"
    else:
        fname = f"data/{mapt[d]}/{mapt[d]}_{cell_line}.csv"
    df = pd.read_csv(fname)
    col = f"chr{chr_number}"
    if col not in df.columns:
        raise KeyError(f"Column {col} not found in {fname}")
    data = df[col].dropna().to_numpy(dtype=float)
    return data

# %%
def savetxt(arr, stitle='save_example', outpath="output"):
    np.savetxt(
        f"data/{outpath}/{stitle}.txt",
        np.asarray(arr, float)
    )

# %%
def savecsv(arr, chr_number, stitle='save_example', outpath="output"):
    fname = Path(f"data/{outpath}/{stitle}.csv")
    col = f"chr{chr_number}"
    arr = np.asarray(arr, float)
    df = pd.read_csv(fname) if fname.exists() else pd.DataFrame()
    if len(df) < len(arr):
        df = df.reindex(range(len(arr)))
    df[col] = pd.Series(arr, index=range(len(arr)))
    cols = list(df.columns)
    cols_sorted = sorted(
        cols,
        key=lambda c: int(c[3:]) if c.startswith("chr") else float("inf")
    )
    df = df[cols_sorted]
    df.to_csv(fname, index=False)

# %%
def logistic(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def interp_nans(y):
    y = np.asarray(y, float)
    x = np.arange(y.size)
    m = np.isfinite(y)
    return y if m.sum() < 2 else np.where(m, y, np.interp(x, x[m], y[m]))

def bwmap(chrom, cell_line='Rat', bin_size=10_000,
          bins=["S1","S2","S3","S4"],
          min_total=1e-9, fill_nans=True):

    x = np.arange(1, len(bins) + 1, dtype=float)
    bws = [pybigtools.open(f"data/repli-seq/RepliSeq_{cell_line}_{b}.bw", "r") for b in bins]

    chroms = bws[0].chroms()
    c = str(chrom)
    if c not in chroms:
        c = f"chr{c}" if (not c.startswith("chr") and f"chr{c}" in chroms) else (c[3:] if c.startswith("chr") and c[3:] in chroms else None)
    if c is None:
        raise KeyError(f"Chromosome {chrom!r} not in bigWig.")

    L = int(chroms[c])
    n = (L + bin_size - 1) // bin_size

    frac = np.vstack([
        np.asarray(bw.values(c, 0, L, bins=n, summary="mean", missing=np.nan), float)
        for bw in bws
    ])
    for bw in bws: bw.close()

    rt = np.full(n, np.nan, float)
    for j in range(n):
        s = frac[:, j]
        if np.any(~np.isfinite(s)): 
            continue
        tot = s.sum()
        if tot < min_total:
            continue
        cum = np.cumsum(s / tot)
        try:
            (k, x0), _ = curve_fit(logistic, x, cum,
                                   p0=(1.0, (len(bins)+1)/2),
                                   maxfev=500)
            rt[j] = x0
        except Exception:
            pass

    return interp_nans(rt) if fill_nans else rt

# %%
def _parse_gtf_attributes(attr_str: str) -> dict:
    out = {}
    for m in re.finditer(r'(\S+)\s+"([^"]*)"\s*;', attr_str):
        out[m.group(1)] = m.group(2)
    return out

def load_gene_models_from_gtf_gz(
    gtf_gz_path: str,
    chrom: str,
    start: int,
    end: int,
    keep_gene_types=None,
    max_genes=200
):
    genes = {}
    exons_by_gene = defaultdict(list)

    with gzip.open(gtf_gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue

            seqname, source, feature, s, e, score, strand, frame, attrs = fields
            if seqname != chrom:
                continue

            s = int(s) - 1
            e = int(e)
            if e <= start or s >= end:
                continue

            ad = _parse_gtf_attributes(attrs)
            gene_id = ad.get("gene_id")
            gene_name = ad.get("gene_name", gene_id)

            if keep_gene_types is not None:
                gene_type = ad.get("gene_type") or ad.get("gene_biotype")
                if gene_type not in keep_gene_types:
                    continue

            if feature == "gene":
                if gene_id is None:
                    continue
                genes[gene_id] = {
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "chrom": seqname,
                    "start": s,
                    "end": e,
                    "strand": strand,
                }

            elif feature == "exon":
                if gene_id is None:
                    continue
                exons_by_gene[gene_id].append((s, e))

            if len(genes) > max_genes:
                break

    for gid, exs in exons_by_gene.items():
        if gid not in genes:
            min_s = min(x[0] for x in exs)
            max_e = max(x[1] for x in exs)
            genes[gid] = {
                "gene_id": gid,
                "gene_name": gid,
                "chrom": chrom,
                "start": min_s,
                "end": max_e,
                "strand": ".",
            }
        genes[gid]["exons"] = sorted(exs)

    out = list(genes.values())
    out.sort(key=lambda g: (g["start"], g["end"]))
    return out

# %%
def plot_gene_track(
    genes,
    chrom: str,
    start: int,
    end: int,
    ax=None,
    show_names=True,
    show_direction=True,
    denseQ=False,
    x_show=True,
    label_fontsize=8,
    exon_height=0.2,
    gene_linewidth=1.0,
    max_label_chars=18,
    arrow_frac=0.015,
    arrow_min_bp=200,
    pad_bp=0
):

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    ax.set_xlim(start, end)
    ax.set_yticks([])
    ax.set_ylabel("Genes")

    if x_show:
        ax.set_xlabel(f"{chrom} position (kb)")
    else:
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    span = max(1, end - start)
    arrow_len = max(arrow_min_bp, int(arrow_frac * span))

    # ----- assign rows -----
    genes_sorted = sorted(genes, key=lambda g: (g["start"], g["end"]))

    if denseQ:
        row_of = [i % 2 for i in range(len(genes_sorted))]
        nrows = 2 if genes_sorted else 1
    else:
        row_ends = []
        row_of = []
        for g in genes_sorted:
            gs, ge = max(g["start"], start), min(g["end"], end)
            placed = False
            for r, last_end in enumerate(row_ends):
                if gs >= last_end + pad_bp:
                    row_of.append(r)
                    row_ends[r] = ge
                    placed = True
                    break
            if not placed:
                row_of.append(len(row_ends))
                row_ends.append(ge)
        nrows = max(row_of) + 1 if row_of else 1

    if ax.figure is not None and ax.figure.get_size_inches()[1] < max(2.5, 0.55 * (nrows + 1)):
        ax.figure.set_size_inches(
            ax.figure.get_size_inches()[0],
            max(2.5, 0.55 * (nrows + 1)),
            forward=True
        )

    ax.set_ylim(-1, nrows)

    def y_from_row(r):
        return nrows - 1 - r

    # ----- plot -----
    label_candidates = []
    for g, r in zip(genes_sorted, row_of):
        y = y_from_row(r)
        gs, ge = max(g["start"], start), min(g["end"], end)

        ax.plot([gs, ge], [y, y], linewidth=gene_linewidth, color="black")

        for (es, ee) in g.get("exons", []):
            es2, ee2 = max(es, start), min(ee, end)
            if ee2 <= es2:
                continue
            rect = plt.Rectangle(
                (es2, y - exon_height / 2),
                ee2 - es2,
                exon_height,
                fill=True,
                facecolor="black",
                edgecolor="black",
                linewidth=0.0
            )
            ax.add_patch(rect)

        strand = g.get("strand", ".")
        if show_direction and strand in {"+", "-"}:
            if strand == "+":
                x0, x1 = max(gs, ge - arrow_len), ge
            else:
                x0, x1 = min(ge, gs + arrow_len), gs
            ax.annotate(
                "",
                xy=(x1, y),
                xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", lw=gene_linewidth, color="black"),
            )

        if show_names:
            name = g.get("gene_name", g["gene_id"])
            if len(name) > max_label_chars:
                name = name[: max_label_chars - 1] + "â€¦"
            label = name
            if show_direction and strand in {"+", "-"}:
                label = f"{name} ({strand})"

            xc = 0.5 * (gs + ge)
            t = ax.text(
                xc, y + 0.35, label,
                fontsize=label_fontsize,
                va="bottom",
                ha="center",
                color="black",
                clip_on=True
            )
            t.set_clip_path(ax.patch)
            t.set_visible(False)  # decide visibility after measuring
            label_candidates.append((ge - gs, r, t))

    # show non-overlapping labels, prioritising larger genes
    if show_names and label_candidates:
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()

        occupied = {r: [] for r in set(rr for _, rr, _ in label_candidates)}
        pad = 0.0  # data units; keep 0 unless you want extra spacing

        for _, r, t in sorted(label_candidates, key=lambda x: x[0], reverse=True):
            bb = t.get_window_extent(renderer=renderer).transformed(inv)
            x0, x1 = bb.x0 - pad, bb.x1 + pad

            ok = True
            for a0, a1 in occupied.get(r, []):
                if not (x1 <= a0 or x0 >= a1):
                    ok = False
                    break

            if ok:
                t.set_visible(True)
                occupied.setdefault(r, []).append((x0, x1))

    ax.grid(True, axis="x", linewidth=0.5)
    return ax

# %% [markdown]
# ### Plotting

# %%
def plotf(*arrays, labels=None, x_array=None, dual_axis=False, resolution=1, region_high=None, rname='Region',
          invyQ=False, figsize=(10, 4), xlims=(None, None), ylims=(None, None), title='',
          xtitle='Index', ytitle='Value', x_show=True, y_show=True, logyQ=False, logxQ=False,
          scale_matchQ=False, saveQ=False, sname='test', ext='pdf', layout_rect=(0.12, 0.15, 0.98, 0.95),
          fig=None, ax=None, showQ=True):

    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays)

    show_labels = not (labels is None and n == 1)

    if scale_matchQ and not (dual_axis and n == 2):
        raise ValueError("scale_matchQ=True only makes sense for dual_axis with exactly two arrays.")

    if x_array is not None:
        xs = [np.asarray(x) for x in x_array]
        if len(xs) != n:
            raise ValueError("Length of x_array must match number of arrays")
        for i, (xi, yi) in enumerate(zip(xs, arrays)):
            if len(xi) != len(yi):
                raise ValueError(f"x_array[{i}] and arrays[{i}] must have same length")
    else:
        xs = [np.arange(len(arr)) * resolution for arr in arrays]

    def draw_region(ax_):
        if region_high is None:
            return None
        region_high_arr = np.asarray(region_high)
        if len(region_high_arr) == 0:
            return None

        spans = []
        start = region_high_arr[0]
        prev = start
        for x in region_high_arr[1:]:
            if x == prev + resolution:
                prev = x
            else:
                spans.append((start, prev + resolution))
                start = x
                prev = x
        spans.append((start, prev + resolution))

        h0 = None
        for a, b in spans:
            h = ax_.axvspan(a, b, color="lightgrey", alpha=0.4, zorder=0)
            if h0 is None:
                h0 = h
        return h0

    def _no_offset(ax_):
        ax_.yaxis.get_offset_text().set_visible(False)

    def _apply_layout(fig_):
        fig_.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1],
                             right=layout_rect[2], top=layout_rect[3])

    created_axes = ax is None

    if n == 2 and dual_axis:
        if ax is None:
            if fig is None:
                fig, ax1 = plt.subplots(figsize=figsize)
            else:
                ax1 = fig.add_subplot(111)
        else:
            ax1 = ax
            if fig is None:
                fig = ax1.figure

        ax2 = ax1.twinx()
        region_handle = draw_region(ax1)

        if labels is not None:
            if len(labels) != 2:
                raise ValueError("For dual_axis=True, labels must have length 2 (or be None).")
            lab1, lab2 = labels
        else:
            lab1, lab2 = ("array_1", "array_2") if show_labels else (None, None)

        p1, = ax1.plot(xs[0], arrays[0], color='tab:blue')
        p2, = ax2.plot(xs[1], arrays[1], color='tab:red')

        if isinstance(logyQ, tuple):
            logyQ1, logyQ2 = logyQ
        else:
            logyQ1 = logyQ2 = bool(logyQ)

        if logyQ1:
            ax1.set_yscale('log')
        if logyQ2:
            ax2.set_yscale('log')

        if isinstance(logxQ, tuple):
            logxQ1, logxQ2 = logxQ
        else:
            logxQ1 = logxQ2 = bool(logxQ)

        if logxQ1:
            ax1.set_xscale('log')
        if logxQ2:
            ax2.set_xscale('log')

        if isinstance(invyQ, tuple):
            invyQ1, invyQ2 = invyQ
        else:
            invyQ1 = invyQ2 = bool(invyQ)

        if invyQ1:
            ax1.invert_yaxis()
        if invyQ2:
            ax2.invert_yaxis()

        if scale_matchQ:
            match_yaxis_scales(ax1, ax2, arrays[0], arrays[1])

        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims if not isinstance(ylims, tuple) else ylims[0])
        ax2.set_ylim(ylims if not isinstance(ylims, tuple) else ylims[1])

        ax1.set_xlabel(xtitle)
        ax1.set_ylabel(lab1 if lab1 is not None else ytitle, color='tab:blue')
        ax2.set_ylabel(lab2 if lab2 is not None else ytitle, color='tab:red')

        ax1.set_title(title)

        if show_labels:
            handles = []
            labels_leg = []
            if region_handle is not None:
                handles.append(region_handle)
                labels_leg.append(f"{rname}")
            if lab1 is not None:
                handles.append(p1)
                labels_leg.append(lab1)
            if lab2 is not None:
                handles.append(p2)
                labels_leg.append(lab2)
            if labels_leg:
                ax2.legend(handles, labels_leg, loc="upper left")

        if not x_show:
            ax1.set_xlabel("")
            ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if not y_show:
            for a in (ax1, ax2):
                a.set_ylabel("")
                a.tick_params(axis="y", which="both", left=False, right=False,
                              labelleft=False, labelright=False)

        _no_offset(ax1)
        _no_offset(ax2)

        if created_axes:
            _apply_layout(fig)
            if saveQ:
                fig.savefig(f"figures/plot_{sname}.{ext}", bbox_inches="tight")
            if showQ:
                plt.show()

        return fig, (ax1, ax2)

    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.figure

    region_handle = draw_region(ax)

    for i, arr in enumerate(arrays):
        if labels is not None:
            if i >= len(labels):
                raise ValueError("labels length must match number of arrays (or be None).")
            lab = labels[i] if show_labels else None
        else:
            lab = (f"array_{i+1}" if show_labels else None)

        ax.plot(xs[i], arr, label=lab)

    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)

    if logyQ:
        ax.set_yscale('log')
    if logxQ:
        ax.set_xscale('log')
    if invyQ:
        ax.invert_yaxis()

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    if show_labels:
        handles, labels_leg = ax.get_legend_handles_labels()
        if region_handle is not None:
            handles = [region_handle] + handles
            labels_leg = [f"{rname}"] + labels_leg
        if labels_leg:
            ax.legend(handles, labels_leg)

    if not x_show:
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if not y_show:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    _no_offset(ax)

    ax.set_title(title)

    if created_axes:
        _apply_layout(fig)
        if saveQ:
            fig.savefig(f"figures/plot_{sname}.{ext}", bbox_inches="tight")
        if showQ:
            plt.show()

# %%
def mplotf(marrays, mlabels=None, mx_array=None, mdual_axis=None, mresolution=None, mregion_high=None, mrname=None,
           minvyQ=None, mfigsize=None, mxlims=None, mylims=None, mxtitle=None, mytitle=None,
           my_show=None, mlogyQ=None, mscale_matchQ=None, msaveQ=None, msname=None, mext=None,
           showgeneQ=False, gtf_gz_path="data/other/gencode/gencode.v49.annotation.gtf.gz", gene_chrom=1,
           gene_denseQ=True, gene_show_names=True, gene_show_direction=False,
           figsize=(10, 4), layout_rect=(0.12, 0.15, 0.98, 0.95), hspace=0.15):

    k = len(marrays)
    W, H = figsize

    def at(v, i, default):
        if isinstance(v, (list, tuple)):
            return v[i] if i < len(v) else default
        return v if v is not None else default

    def first_len():
        g0 = marrays[0]
        return len(g0[0]) if isinstance(g0, (list, tuple)) else len(g0)

    n0 = first_len()
    x0_in = at(mx_array, 0, None)
    if x0_in is None:
        res0 = at(mresolution, 0, 1)
        xk = np.arange(n0, dtype=float) * float(res0)  # kb
    else:
        xk = np.asarray(x0_in, float)
        if xk.size != n0:
            raise ValueError("First mx_array length must match the length of the first plotted array.")

    xk_start = float(np.nanmin(xk))
    xk_end   = float(np.nanmax(xk))

    gene_start_bp = int(np.floor(xk_start * 1000.0))
    gene_end_bp   = int(np.ceil(xk_end   * 1000.0))

    nrows = k + (1 if showgeneQ else 0)
    fig, axs = plt.subplots(nrows, 1, figsize=(W, H * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    gi = 0
    if showgeneQ:
        chrom_try = f"chr{gene_chrom}" if isinstance(gene_chrom, (int, np.integer)) else str(gene_chrom)

        genes = load_gene_models_from_gtf_gz(
            gtf_gz_path, chrom_try, gene_start_bp, gene_end_bp,
            keep_gene_types="protein_coding", max_genes=200
        )
        if len(genes) == 0:
            chrom_alt = chrom_try[3:] if chrom_try.startswith("chr") else "chr" + chrom_try
            genes = load_gene_models_from_gtf_gz(
                gtf_gz_path, chrom_alt, gene_start_bp, gene_end_bp,
                keep_gene_types="protein_coding", max_genes=200
            )
            if len(genes) > 0:
                chrom_try = chrom_alt

        genes_kb = []
        for g in genes:
            gg = dict(g)
            gg["start"] = gg["start"] / 1000.0
            gg["end"]   = gg["end"]   / 1000.0
            if "exons" in gg:
                gg["exons"] = [(es / 1000.0, ee / 1000.0) for es, ee in gg["exons"]]
            genes_kb.append(gg)

        plot_gene_track(
            genes_kb, chrom_try, xk_start, xk_end,
            ax=axs[0],
            show_names=gene_show_names,
            show_direction=gene_show_direction,
            denseQ=gene_denseQ,
            x_show=False
        )

        axs[0].set_title("")
        axs[0].grid(False)
        gi = 1

    for i in range(k):
        g = marrays[i]
        ax = axs[i + gi]
        is_list = isinstance(g, (list, tuple))
        n_in = len(g) if is_list else 1

        dualQ = bool(at(mdual_axis, i, False)) and is_list and n_in == 2
        args = tuple(g) if is_list else (g,)

        plotf(*args,
              labels=at(mlabels, i, None),
              x_array=at(mx_array, i, None),
              dual_axis=dualQ,
              resolution=at(mresolution, i, 1),
              region_high=at(mregion_high, i, None),
              rname=at(mrname, i, 'Region'),
              invyQ=at(minvyQ, i, False),
              xlims=at(mxlims, i, (None, None)),
              ylims=at(mylims, i, (None, None)),
              xtitle=at(mxtitle, i, 'Index'),
              ytitle=at(mytitle, i, 'Value'),
              x_show=(i == k - 1),   # bottom plot only (among plotf panels)
              y_show=at(my_show, i, True),
              logyQ=at(mlogyQ, i, False),
              scale_matchQ=at(mscale_matchQ, i, False),
              saveQ=False,
              sname=at(msname, i, 'test'),
              ext=at(mext, i, 'pdf'),
              layout_rect=layout_rect,
              fig=fig,
              ax=ax,
              showQ=False)

    axs[-1].set_xlim(xk_start, xk_end)

    # If the gene axis used set_xticks([]), it would have cleared the shared ticks.
    # Restore auto ticks + ensure bottom axis is visible.
    if showgeneQ:
        axs[-1].xaxis.set_major_locator(mticker.AutoLocator())
        axs[-1].xaxis.set_minor_locator(mticker.NullLocator())
        axs[-1].tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

    fig.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1],
                        right=layout_rect[2], top=layout_rect[3], hspace=hspace)

    if at(msaveQ, 0, False):
        fig.savefig(f"figures/plot_{at(msname, 0, 'test')}.{at(mext, 0, 'pdf')}", bbox_inches="tight")

    plt.show()

# %%
def ploth(*arrays, labels=None, bin_size=5, alpha=0.5, density=False, statsQ=True,
          figsize=(10,4), xtitle='Index', ytitle='Value', xmax=None, saveQ=False, sname='test', ext='pdf'):

    arrays = [np.asarray(a) for a in arrays]
    max_val = max(np.nanmax(a) for a in arrays)

    if bin_size is None:
        bins = 50
    else:
        bins = np.arange(0, max_val + bin_size, bin_size)

    plt.figure(figsize=figsize)

    legend_entries = []

    for i, arr in enumerate(arrays):
        base_label = labels[i] if labels and i < len(labels) else None

        if statsQ:
            stats = f"Î¼={np.nanmean(arr):.2f}\n Ïƒ={np.nanstd(arr):.2f}\n M={np.nanmedian(arr):.2f}"
            label = (base_label if base_label else "") + stats
        else:
            label = base_label

        legend_entries.append(label)
        plt.hist(arr, bins=bins, alpha=alpha, density=density,
                 label=label)

    plt.xlabel(xtitle)
    plt.ylabel("Frequency" if not density else "Probability density")
    plt.xlim(0,xmax)

    if labels is not None or statsQ:
        leg = plt.legend(loc="upper right")
        for t in leg.get_texts():
            t.set_ha('right')

    plt.tight_layout()

    if saveQ:
        plt.savefig(f"figures/histogram_{sname}.{ext}", bbox_inches='tight')
    plt.show()

# %%
def plotb(array, label=None, resolution=1, figsize=(10,4), xtitle='Index', ytitle='Value', saveQ=False, sname='test', ext='pdf'):
    arr = np.asarray(array)
    x = np.arange(len(arr)) * resolution

    plt.figure(figsize=figsize)
    plt.bar(x, arr, width=resolution, label=label if label else "array", color='orange')

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if label:
        plt.legend()

    plt.tight_layout()
    if saveQ:
        plt.savefig(f"figures/barplot_{sname}.{ext}", bbox_inches='tight')
    plt.show()

# %%
def plotd(x, y, frac=0.1, random_state=None, xlabel='X', ylabel='Y', title='Density', figsize=(10,4),
          xlim=None, ylim=None, saveQ=False, sname='test', ext='pdf'):

    # keep only valid positive values
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    n = len(x)
    if n == 0:
        print("No valid points to plot.")
        return

    # subsample
    np.random.seed(random_state)
    sample_size = int(frac * n)
    if sample_size < n:
        idx = np.random.choice(n, size=sample_size, replace=False)
        x = x[idx]
        y = y[idx]

    # KDE evaluated on a regular grid
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)

    # determine limits if not provided
    if xlim is None:
        xmin, xmax = np.min(x), np.max(x)
    else:
        xmin, xmax = xlim

    if ylim is None:
        ymin, ymax = np.min(y), np.max(y)
    else:
        ymin, ymax = ylim

    # grid resolution
    X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    plt.figure(figsize=figsize)
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.yscale('log')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(label='Density')
    plt.tight_layout()

    if saveQ:
        plt.savefig(f"figures/dens_{sname}.{ext}", bbox_inches='tight')
    
    plt.show()

# %%
def plotw(*arrays, labels=None, figsize=(10,4), ylims=(None, None),
          xtitle='', ytitle="Value", logyQ=False, invyQ=False,
          violinQ=False, saveQ=False, sname="test", ext='pdf'):

    arrays = [np.asarray(a, float) for a in arrays]
    arrays = [a[np.isfinite(a)] for a in arrays]

    if any(a.size == 0 for a in arrays):
        raise ValueError("All input arrays must contain at least one finite value.")

    n = len(arrays)
    if labels is not None and len(labels) != n:
        raise ValueError("labels length must match number of arrays (or be None).")

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    if violinQ:
        parts = ax.violinplot(
            arrays,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.3)
            pc.set_facecolor("tab:blue")
            pc.set_edgecolor("none")

    ax.boxplot(
        arrays,
        vert=True,
        whis=1.5,
        showfliers=True,
        tick_labels=labels,
    )

    if labels is None:
        ax.set_xticks([])

    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)

    if logyQ:
        ax.set_yscale("log")
    if invyQ:
        ax.invert_yaxis()

    ax.set_ylim(ylims)

    plt.tight_layout()
    if saveQ:
        fig.savefig(f"figures/bplot_{sname}.{ext}", bbox_inches="tight")
    plt.show()

# %%
def plothm(
    field,
    resolution_hm=None,  
    smooth_type="bicubic", 
    bscale=None,
    cmap="viridis",
    x_ticks=None,
    t_ticks=None,
    xtitle="Index (x)",
    ytitle="Index (y)",
    ctitle="Value",
    figsize=(10, 4),
    xlims=(None, None),
    ylims=(None, None),
    invyQ=False,
    x_show=True,
    y_show=True,
    region_high=None,
    rname="Region",
    saveQ=False,
    sname="test",
    ext="pdf",
    layout_rect=(0.12, 0.15, 0.98, 0.95),
    fig=None,
    ax=None,
    showQ=True,
):
    # 1. Prepare Data
    F = np.asarray(field, dtype=float)
    orig_nt, orig_nx = F.shape

    # 2. Handle Resolution Tweak (Resampling)
    if resolution_hm is not None:
        zoom_t = resolution_hm / orig_nt
        zoom_x = resolution_hm / orig_nx
        F_plot = zoom(F, (zoom_t, zoom_x), order=1)
        interp_to_use = smooth_type # Use your requested smoothing
    else:
        F_plot = F
        interp_to_use = "nearest" # RAW data rendering: no interpolation

    # 3. Handle Ticks
    if x_ticks is None:
        x = np.arange(orig_nx, dtype=float)
    else:
        x = np.asarray(x_ticks, dtype=float)

    if t_ticks is None:
        t = np.arange(orig_nt, dtype=float)
    else:
        t = np.asarray(t_ticks, dtype=float)

    # 4. Setup Figure/Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize) if fig is None else (fig, fig.add_subplot(111))
    else:
        fig = ax.figure if fig is None else fig

    # 5. Visibility / Limits Logic
    xmin, xmax = xlims
    ymin, ymax = ylims
    ix = np.ones(orig_nx, dtype=bool)
    it = np.ones(orig_nt, dtype=bool)
    if xmin is not None: ix &= x >= xmin
    if xmax is not None: ix &= x <= xmax
    if ymin is not None: it &= t >= ymin
    if ymax is not None: it &= t <= ymax

    F_vis = F[np.ix_(it, ix)]
    if bscale is None:
        vmin, vmax = np.nanmin(F_vis), np.nanmax(F_vis)
    else:
        vmin, vmax = bscale

    # 6. Draw Region Overlay
    def draw_region(ax_):
        if region_high is None: return None
        rh = np.sort(np.asarray(region_high, dtype=float).ravel())
        if rh.size == 0: return None
        dx = np.min(np.diff(x)) if x.size > 1 else 1.0
        spans, start, prev = [], rh[0], rh[0]
        for v in rh[1:]:
            if np.isclose(v, prev + dx):
                prev = v
            else:
                spans.append((start, prev + dx))
                start = prev = v
        spans.append((start, prev + dx))
        h0 = None
        for a, b in spans:
            h = ax_.axvspan(a, b, color="lightgrey", alpha=0.4, zorder=1)
            if h0 is None: h0 = h
        return h0

    region_handle = draw_region(ax)

    # 7. Render Heatmap
    im = ax.imshow(
        F_plot,
        aspect="auto",
        origin="lower",
        interpolation=interp_to_use, # Uses 'nearest' if resolution_hm is None
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[x[0], x[-1], t[0], t[-1]],
        zorder=0,
    )

    # 8. Formatting
    ax.set_xlabel(xtitle)
    ax.set_ylabel(ytitle)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if invyQ: ax.invert_yaxis()
    if not x_show:
        ax.set_xlabel("")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if not y_show:
        ax.set_ylabel("")
        ax.tick_params(axis="y", left=False, labelleft=False)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(ctitle)

    if region_handle is not None:
        ax.legend([region_handle], [rname], loc="upper left")

    fig.subplots_adjust(left=layout_rect[0], bottom=layout_rect[1], 
                        right=layout_rect[2], top=layout_rect[3])

    if saveQ:
        fig.savefig(f"figures/heatmap_{sname}.{ext}", bbox_inches="tight")

    if showQ:
        plt.show()

# %% [markdown]
# ### Operations

# %%
def smoothf(data, window=50):
    data = np.asarray(data, float)
    n = len(data)
    win = min(window, n)
    kernel = np.ones(win, float) / win
    size = n + win - 1
    fft_A = rfft(data, size)
    fft_B = rfft(kernel, size)
    conv_full = irfft(fft_A * fft_B, size)
    conv = conv_full[win//2 : win//2 + n]
    return conv

# %%
# rescale array
def rescale(arr, interval):
    new_min, new_max = interval
    arr = np.asarray(arr, dtype=float)
    old_min, old_max = np.nanmin(arr), np.nanmax(arr)
    if old_max == old_min:
        return np.full_like(arr, (new_min + new_max) / 2.0)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

# %%
def fast_shift_add(dst, src, shift, perQ=False):
    
    if shift == 0:
        dst += src
        return
    
    if perQ:
        # periodic wrap
        dst += np.roll(src, shift)
    else:
        # standard linear genome
        if shift > 0:
            dst[shift:] += src[:-shift]
        else:
            dst[:shift] += src[-shift:]

# %%
def match_yaxis_scales(ax1, ax2, y1, y2):
    y1 = np.asarray(y1, float)
    y2 = np.asarray(y2, float)

    # Normalise both curves
    y1n = (y1 - y1.min()) / (y1.max() - y1.min() + 1e-12)
    y2n = (y2 - y2.min()) / (y2.max() - y2.min() + 1e-12)

    # Best affine match: y2n ~ alpha*y1n + beta
    A = np.vstack([y1n, np.ones_like(y1n)]).T
    alpha, beta = np.linalg.lstsq(A, y2n, rcond=None)[0]

    y1_matched = alpha * y1n + beta

    # Combined normalised range
    lo = min(y1_matched.min(), y2n.min())
    hi = max(y1_matched.max(), y2n.max())

    # Map back to original scale
    y1_lo = lo * (y1.max() - y1.min()) + y1.min()
    y1_hi = hi * (y1.max() - y1.min()) + y1.min()

    y2_lo = lo * (y2.max() - y2.min()) + y2.min()
    y2_hi = hi * (y2.max() - y2.min()) + y2.min()

    # ------------------------------------------------------------------
    # SAFETY FOR LOG-SCALE AXES â€” enforce positive bounds > 0
    # ------------------------------------------------------------------
    eps = 1e-9
    if ax1.get_yscale() == 'log':
        y1_lo = max(y1_lo, eps)
        y1_hi = max(y1_hi, y1_lo * 1.1)

    if ax2.get_yscale() == 'log':
        y2_lo = max(y2_lo, eps)
        y2_hi = max(y2_hi, y2_lo * 1.1)

    # Apply final limits
    ax1.set_ylim(y1_lo, y1_hi)
    ax2.set_ylim(y2_lo, y2_hi)

# %%
# Mean squared error between two arrays
def mean_squared_error(a, b):
    return float(np.mean((a - b) ** 2))

# %%
# convert to H:M:S
def hms(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# %%
def arrint(a, b):
    return np.intersect1d(a, b)
def arrcomp(a, b):
    return np.setdiff1d(a, b)

# %%
def etaf(it, maxiter, start_time):
    elapsed = time.time() - start_time
    sims_left = maxiter - it
    avg_time_per_sim = elapsed / it
    eta_sec = sims_left * avg_time_per_sim
    print(f"[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta_sec)}", end="\r")   
def etaferr(it, maxiter, start_time, err):
    elapsed = time.time() - start_time
    sims_left = maxiter - it
    avg_time_per_sim = elapsed / it
    eta_sec = sims_left * avg_time_per_sim
    print(f"[{it}/{maxiter}]  Elapsed: {hms(elapsed)}  ETA: {hms(eta_sec)}  MSE: {err:.4e}", end="\r")   

# %% [markdown]
# ## Mapping functions

# %%
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

# %%
def map_firing_timing0(
    firing_rate,
    fork_speed=1.4,
    resolution=1,
    neighbourhood_range=2000,
    perQ=False
):
    x = np.asarray(firing_rate, dtype=float)
    fork_speed_per_bin = fork_speed / resolution
    L = max(1, int(neighbourhood_range / resolution))

    y = np.zeros_like(x)
    last_raw = np.zeros_like(x)
    last_exp = np.ones_like(x)
    unitary = x.copy()

    for k in range(L + 1):
        if k:
            fast_shift_add(unitary, x,  k, perQ=perQ)
            fast_shift_add(unitary, x, -k, perQ=perQ)

        exp2_raw = last_raw + unitary / fork_speed_per_bin
        exp2 = np.exp(-exp2_raw)
        y += (last_exp - exp2) / np.clip(unitary, 1e-80, None)
        last_raw, last_exp = exp2_raw, exp2

    return y

# %%
def map_firing_timing0(
    firing_rate,
    fork_speed=1.4,
    resolution=1,
    neighbourhood_range=2000,
    perQ=False
):
    x = np.asarray(firing_rate, dtype=float)
    fork_speed_per_bin = fork_speed / resolution
    L = max(1, int(neighbourhood_range / resolution))

    n = x.size
    if perQ:
        L = min(L, n // 2)

    y = np.zeros_like(x)
    last_raw = np.zeros_like(x)
    last_exp = np.ones_like(x)

    unitary = x.copy()

    for k in range(L + 1):

        if perQ:
            unitary = x.copy()

        if k:
            fast_shift_add(unitary, x,  k, perQ=perQ)
            fast_shift_add(unitary, x, -k, perQ=perQ)

        exp2_raw = last_raw + unitary / fork_speed_per_bin
        exp2 = np.exp(-exp2_raw)
        y += (last_exp - exp2) / np.clip(unitary, 1e-80, None)
        last_raw, last_exp = exp2_raw, exp2

    return y

# %%
def map_firing_timing(
    firing_rate,
    fork_speed=1.4,
    resolution=1,
    neighbourhood_range=2000,
    perQ=False
):
    x = np.asarray(firing_rate, dtype=float)
    n = x.size
    fork_speed_per_bin = fork_speed / resolution

    # Requested neighbourhood radius in bins
    L_req = max(1, int(neighbourhood_range / resolution))

    # Finite periodic cap to avoid repeated counting on the circle.
    # Your proposed R = ceil((n-3)/2) equals (n-2)//2 for integers.
    if perQ:
        R = max(0, (n - 2) // 2)
        L = min(L_req, R)
    else:
        L = L_req

    y = np.zeros_like(x)
    last_raw = np.zeros_like(x)
    last_exp = np.ones_like(x)
    unitary = x.copy()

    for k in range(L + 1):
        if k:
            fast_shift_add(unitary, x,  k, perQ=perQ)
            fast_shift_add(unitary, x, -k, perQ=perQ)

        exp2_raw = last_raw + unitary / fork_speed_per_bin
        exp2 = np.exp(-exp2_raw)
        y += (last_exp - exp2) / np.clip(unitary, 1e-80, None)
        last_raw, last_exp = exp2_raw, exp2

    return y

# %%
# T(x) â†’ RFD(x) via RFD = v dT/dx
def map_timing_directionality(
    timing,
    fork_speed=1.4,
    resolution=1.0,
    T0=0.0,
    smoothw=None,
    perQ=False,
):
    T = np.asarray(timing, dtype=float)

    dT_per_bin = np.empty_like(T)

    if perQ:
        # periodic finite difference: wrap last â†’ first
        dT_per_bin[:] = T - np.roll(T, 1)
    else:
        # original non-periodic behaviour
        dT_per_bin[0] = T[0] - T0
        dT_per_bin[1:] = np.diff(T)

    # RFD = v * dT/dx with dx = resolution
    rfd = (fork_speed / resolution) * dT_per_bin

    # optional smoothing of RFD
    rfd = smoothf(rfd, smoothw)

    # enforce biological bounds
    rfd = np.clip(rfd, -1.0, 1.0)

    return rfd

# %%
# RFD(x) â†’ T(x)  via  dT/dx = (1/v) RFD(x)
def map_directionality_timing(
    rfd,
    fork_speed=1.4,
    resolution=1.0,
    T0=0.0,
    smoothw=None,
    perQ=False,
):
    rfd = np.asarray(rfd, dtype=float)

    # optional smoothing of RFD before integration
    rfd = smoothf(rfd, smoothw)

    # each bin contributes Î”T = (resolution / v) * RFD
    dT_per_bin = (resolution / fork_speed) * rfd

    if perQ:
        # periodic integration over circle
        T = np.cumsum(dT_per_bin)
        # remove global offset by enforcing circular consistency
        shift = (T[-1] + dT_per_bin[0]) / len(T)
        T = T - shift * np.arange(len(T))
    else:
        # standard linear integration
        T = T0 + np.cumsum(dT_per_bin)

    return T

# %% [markdown]
# ## Simulation functions

# %%
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
    time_stats_densQ=False,  # NEW (densities handled in rsim; here kept for API symmetry)
    time_grid=None,
    max_rep_time=1200.0,
):
    rng = np.random.default_rng() if rng is None else rng

    resolution_space = float(resolution_space)
    if not np.isfinite(resolution_space) or resolution_space <= 0:
        raise ValueError("resolution_space must be a finite positive number.")

    resolution_time = float(resolution_time)
    if not np.isfinite(resolution_time) or resolution_time <= 0:
        raise ValueError("resolution_time must be a finite positive number.")

    def _as_rate_field(field, name, n_hint=None):
        arr = np.asarray(field, dtype=float)
        if arr.ndim == 0:
            if not np.isfinite(arr) or arr < 0:
                raise ValueError(f"{name} scalar must be finite and >= 0.")
            return "scalar", float(arr), None
        if arr.ndim == 1:
            if n_hint is not None and arr.size != n_hint:
                raise ValueError(f"{name} (n,) must have length {n_hint}, got {arr.size}.")
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} (n,) must be finite and >= 0.")
            return "x", arr.astype(float, copy=False), None
        if arr.ndim == 2:
            if n_hint is not None and arr.shape[1] != n_hint:
                raise ValueError(f"{name} (T,n) must have n={n_hint} columns, got {arr.shape[1]}.")
            if np.any(~np.isfinite(arr)) or np.any(arr < 0):
                raise ValueError(f"{name} (T,n) must be finite and >= 0.")
            return "xt", arr.astype(float, copy=False), int(arr.shape[0])
        raise ValueError(f"{name} must be scalar, (n,), or (T,n).")

    def _as_speed_field(field, name, n_hint=None):
        arr = np.asarray(field, dtype=float)
        if arr.ndim == 0:
            if not np.isfinite(arr) or arr <= 0:
                raise ValueError(f"{name} scalar must be finite and > 0.")
            return "scalar", float(arr), None
        if arr.ndim == 1:
            if n_hint is not None and arr.size != n_hint:
                raise ValueError(f"{name} (n,) must have length {n_hint}, got {arr.size}.")
            if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError(f"{name} (n,) must be finite and > 0.")
            return "x", arr.astype(float, copy=False), None
        if arr.ndim == 2:
            if n_hint is not None and arr.shape[1] != n_hint:
                raise ValueError(f"{name} (T,n) must have n={n_hint} columns, got {arr.shape[1]}.")
            if np.any(~np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError(f"{name} (T,n) must be finite and > 0.")
            return "xt", arr.astype(float, copy=False), int(arr.shape[0])
        raise ValueError(f"{name} must be scalar, (n,), or (T,n).")

    ori_arr = np.asarray(ori_rate, dtype=float)
    if ori_arr.ndim == 0:
        raise ValueError("ori_rate must be (n,) or (T,n) (scalar does not define genome length).")
    if ori_arr.ndim == 1:
        n = int(ori_arr.size)
    elif ori_arr.ndim == 2:
        n = int(ori_arr.shape[1])
    else:
        raise ValueError("ori_rate must be (n,) or (T,n).")

    I_kind, I_val, TI = _as_rate_field(ori_rate, "ori_rate", n_hint=n)
    v_kind, v_val, Tv = _as_speed_field(fork_speed, "fork_speed", n_hint=n)

    inv_v_x = None
    inv_v_xt = None
    if v_kind == "scalar":
        inv_v_scalar = 1.0 / v_val
    elif v_kind == "x":
        inv_v_x = 1.0 / v_val
    else:
        inv_v_xt = 1.0 / v_val
        Tv = inv_v_xt.shape[0]

    p = float(stall_rate)
    stall_left_mask = rng.random(n) < p
    stall_right_mask = rng.random(n) < p
    stall_delay_left = np.zeros(n, float)
    stall_delay_right = np.zeros(n, float)
    if p > 0.0:
        if np.isfinite(tau):
            if stall_left_mask.any():
                stall_delay_left[stall_left_mask] = rng.exponential(scale=tau, size=int(stall_left_mask.sum()))
            if stall_right_mask.any():
                stall_delay_right[stall_right_mask] = rng.exponential(scale=tau, size=int(stall_right_mask.sum()))
        else:
            stall_delay_left[stall_left_mask] = np.inf
            stall_delay_right[stall_right_mask] = np.inf

    def _row_for_time(t):
        if not np.isfinite(t) or t < 0.0:
            return 0
        r = int(t // resolution_time)
        if inv_v_xt is not None and r >= Tv:
            r = Tv - 1
        return r

    def _edge_forward(t_here, k_to):
        if inv_v_xt is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_xt[r, k_to] + stall_delay_left[k_to]
        if inv_v_x is not None:
            return resolution_space * inv_v_x[k_to] + stall_delay_left[k_to]
        return resolution_space * inv_v_scalar + stall_delay_left[k_to]

    def _edge_backward(t_here, k_to):
        if inv_v_xt is not None:
            r = _row_for_time(t_here)
            return resolution_space * inv_v_xt[r, k_to] + stall_delay_right[k_to]
        if inv_v_x is not None:
            return resolution_space * inv_v_x[k_to] + stall_delay_right[k_to]
        return resolution_space * inv_v_scalar + stall_delay_right[k_to]

    # firing proposals t0
    t0 = np.full(n, np.inf, float)
    if I_kind == "x":
        rate = I_val
        pos = rate > 0
        if np.any(pos):
            t0[pos] = rng.exponential(scale=1.0 / rate[pos])
    elif I_kind == "scalar":
        if I_val > 0:
            t0[:] = rng.exponential(scale=1.0 / I_val, size=n)
    else:
        u = rng.random(n)
        target = -np.log(u)
        H = np.cumsum(I_val * resolution_time, axis=0)
        ge = H >= target[None, :]
        hit = ge.any(axis=0)
        j = np.argmax(ge, axis=0).astype(np.int64, copy=False)
        if np.any(hit):
            jh = j[hit]
            kh = np.flatnonzero(hit).astype(np.int64, copy=False)
            prevH = np.zeros(kh.size, float)
            m0 = jh > 0
            if np.any(m0):
                prevH[m0] = H[jh[m0] - 1, kh[m0]]
            lam = I_val[jh, kh]
            frac = np.ones_like(lam)
            poslam = lam > 0
            frac[poslam] = (target[kh[poslam]] - prevH[poslam]) / (lam[poslam] * resolution_time)
            frac = np.clip(frac, 0.0, 1.0)
            t0[kh] = (jh.astype(float) + frac) * resolution_time

    # firing-factor gating (unchanged)
    use_factors = ffiring_nr is not None
    if use_factors:
        Amax = float(ffiring_nr)
        if not np.isfinite(Amax) or Amax <= 0:
            raise ValueError("ffiring_nr must be a finite positive number.")
        cost = 2.0
        if Amax < cost:
            raise ValueError("ffiring_nr must be >= 2 if modelling two forks per origin firing.")

        r = 0.0 if ffiring_forkQ else float(ffiring_recycle)
        if not ffiring_forkQ and (not np.isfinite(r) or r < 0):
            raise ValueError("ffiring_recycle must be a finite non-negative number.")

        def recover(A_now, dt):
            if r == 0.0 or dt <= 0.0:
                return A_now
            return Amax - (Amax - A_now) * np.exp(-r * dt)

        def wait_time_to_cost(A_now):
            if A_now >= cost:
                return 0.0
            if r == 0.0:
                return np.inf
            denom = Amax - A_now
            num = Amax - cost
            if denom <= 0.0 or num <= 0.0:
                return np.inf
            return (1.0 / r) * np.log(denom / num)

        def propagate_from_times(t):
            if not perQ:
                g = np.empty(n, float)
                src = np.empty(n, np.int32)
                g[0], src[0] = t[0], 0
                for k in range(1, n):
                    cand = g[k - 1] + _edge_forward(g[k - 1], k)
                    if t[k] <= cand:
                        g[k], src[k] = t[k], k
                    else:
                        g[k], src[k] = cand, src[k - 1]
                for k in range(n - 2, -1, -1):
                    cand = g[k + 1] + _edge_backward(g[k + 1], k)
                    if cand < g[k]:
                        g[k], src[k] = cand, src[k + 1]
                return g, src
            g = t.copy()
            src = np.arange(n, dtype=np.int32)
            visited = np.zeros(n, bool)
            pq = [(g[i], i) for i in range(n) if np.isfinite(g[i])]
            heapq.heapify(pq)
            while pq:
                time_u, u = heapq.heappop(pq)
                if visited[u] or time_u > g[u]:
                    continue
                visited[u] = True
                v = (u + 1) % n
                w = _edge_forward(time_u, v)
                if np.isfinite(w):
                    cand = time_u + w
                    if cand < g[v]:
                        g[v] = cand
                        src[v] = src[u]
                        heapq.heappush(pq, (cand, v))
                v = (u - 1) % n
                w = _edge_backward(time_u, v)
                if np.isfinite(w):
                    cand = time_u + w
                    if cand < g[v]:
                        g[v] = cand
                        src[v] = src[u]
                        heapq.heappush(pq, (cand, v))
            return g, src

        def schedule_with_releases(t_base, release_guess):
            t_adj = np.full(n, np.inf, float)
            A = Amax
            current_time = 0.0
            cand_idx = np.flatnonzero(np.isfinite(t_base)).astype(np.int64, copy=False)
            cand_heap = [(t_base[i], int(i)) for i in cand_idx]
            heapq.heapify(cand_heap)
            rel_heap = []
            firedQ = np.zeros(n, bool)
            while cand_heap or rel_heap:
                next_cand_t = cand_heap[0][0] if cand_heap else np.inf
                next_rel_t = rel_heap[0][0] if rel_heap else np.inf
                if next_rel_t <= next_cand_t:
                    te, add = heapq.heappop(rel_heap)
                    if te > current_time:
                        A = recover(A, te - current_time)
                        current_time = te
                    A = min(Amax, A + add)
                    continue
                ti, i = heapq.heappop(cand_heap)
                if firedQ[i]:
                    continue
                if ti > current_time:
                    A = recover(A, ti - current_time)
                    current_time = ti
                if A >= cost:
                    A -= cost
                    firedQ[i] = True
                    t_adj[i] = current_time
                    rt = release_guess[i]
                    if np.isfinite(rt) and rt >= current_time:
                        heapq.heappush(rel_heap, (rt, cost))
                else:
                    dtw = wait_time_to_cost(A)
                    if not np.isfinite(dtw):
                        firedQ[i] = True
                        t_adj[i] = np.inf
                        continue
                    heapq.heappush(cand_heap, (current_time + dtw, i))
            return t_adj

        if not ffiring_forkQ:
            t = t0.copy()
            heap_idx = np.flatnonzero(np.isfinite(t)).astype(np.int64, copy=False)
            heap = [(t[i], int(i)) for i in heap_idx]
            heapq.heapify(heap)
            done = np.zeros(n, bool)
            A = Amax
            current_time = 0.0
            t_adj = t.copy()
            while heap:
                ti, i = heapq.heappop(heap)
                if done[i]:
                    continue
                if ti > current_time:
                    A = recover(A, ti - current_time)
                    current_time = ti
                if A >= cost:
                    A -= cost
                    done[i] = True
                    t_adj[i] = current_time
                else:
                    dtw = wait_time_to_cost(A)
                    if not np.isfinite(dtw):
                        t_adj[i] = np.inf
                        done[i] = True
                        continue
                    current_time = current_time + dtw
                    A = recover(A, dtw)
                    heapq.heappush(heap, (current_time, i))
            t = t_adj
        else:
            t = t0.copy()
            release_guess = np.full(n, np.inf, float)
            t = schedule_with_releases(t0, release_guess)
            for _ in range(3):
                rep_time_tmp, src_tmp = propagate_from_times(t)
                if not perQ:
                    changes = np.flatnonzero(np.r_[True, src_tmp[1:] != src_tmp[:-1], True])
                    starts = changes[:-1]
                    ends = changes[1:] - 1
                    labels = src_tmp[starts].astype(np.int64, copy=False)
                    release_guess.fill(np.inf)
                    release_guess[labels] = np.maximum(rep_time_tmp[starts], rep_time_tmp[ends])
                else:
                    release_guess.fill(np.inf)
                    fired = np.flatnonzero((src_tmp == np.arange(n, dtype=np.int32)) & np.isfinite(t)).astype(np.int64, copy=False)
                    for ori in fired:
                        seg = np.flatnonzero(src_tmp == ori)
                        if seg.size:
                            release_guess[ori] = np.nanmax(rep_time_tmp[seg])
                t_new = schedule_with_releases(t0, release_guess)
                if np.allclose(np.nan_to_num(t_new, nan=np.inf, posinf=np.inf),
                               np.nan_to_num(t, nan=np.inf, posinf=np.inf),
                               rtol=0, atol=1e-12):
                    t = t_new
                    break
                t = t_new
    else:
        t = t0

    # propagation with prev
    if not perQ:
        g = np.empty(n, float)
        src = np.empty(n, np.int32)
        prev = np.full(n, -1, np.int32)
        g[0], src[0], prev[0] = t[0], 0, 0
        for k in range(1, n):
            cand = g[k - 1] + _edge_forward(g[k - 1], k)
            if t[k] <= cand:
                g[k], src[k], prev[k] = t[k], k, k
            else:
                g[k], src[k], prev[k] = cand, src[k - 1], k - 1
        for k in range(n - 2, -1, -1):
            cand = g[k + 1] + _edge_backward(g[k + 1], k)
            if cand < g[k]:
                g[k], src[k], prev[k] = cand, src[k + 1], k + 1
    else:
        g = t.copy()
        src = np.arange(n, dtype=np.int32)
        prev = np.arange(n, dtype=np.int32)
        visited = np.zeros(n, bool)
        pq = [(g[i], i) for i in range(n) if np.isfinite(g[i])]
        heapq.heapify(pq)
        while pq:
            time_u, u = heapq.heappop(pq)
            if visited[u] or time_u > g[u]:
                continue
            visited[u] = True
            v = (u + 1) % n
            w = _edge_forward(time_u, v)
            if np.isfinite(w):
                cand = time_u + w
                if cand < g[v]:
                    g[v] = cand
                    src[v] = src[u]
                    prev[v] = u
                    heapq.heappush(pq, (cand, v))
            v = (u - 1) % n
            w = _edge_backward(time_u, v)
            if np.isfinite(w):
                cand = time_u + w
                if cand < g[v]:
                    g[v] = cand
                    src[v] = src[u]
                    prev[v] = u
                    heapq.heappush(pq, (cand, v))

    idx = np.arange(n, dtype=np.int32)
    fired_mask = (src == idx) & np.isfinite(t)
    fired_idx = np.flatnonzero(fired_mask).astype(np.int64, copy=False)

    fork_directionality = np.zeros(n, float)
    if not perQ:
        fork_directionality[prev == (idx - 1)] = 1.0
        fork_directionality[prev == (idx + 1)] = -1.0
    else:
        fork_directionality[prev == ((idx - 1) % n)] = 1.0
        fork_directionality[prev == ((idx + 1) % n)] = -1.0

    fired_times_min = t[fired_idx]
    fired_pos_kb = fired_idx.astype(float) * resolution_space

    if fired_idx.size >= 2:
        fired_sorted = np.sort(fired_idx)
        diffs = np.diff(fired_sorted)
        if perQ:
            wrap = n - fired_sorted[-1] + fired_sorted[0]
            interorigin_dist_kb = np.concatenate([diffs, [wrap]]).astype(float) * resolution_space
            if interorigin_dist_kb.size != fired_sorted.size:
                raise RuntimeError("perQ=True: expected len(IODs) == number of fired origins.")
        else:
            interorigin_dist_kb = diffs.astype(float) * resolution_space
    else:
        interorigin_dist_kb = np.array([], float)

    replication_time_min = g
    if max_rep_time is not None:
        replication_time_min = np.minimum(replication_time_min, float(max_rep_time))

    # --- time statistics (1D only, stays here) ---
    time_stats = None
    if time_statsQ:
        if time_grid is None:
            raise ValueError("time_statsQ=True requires a time_grid array.")
        time_grid = np.asarray(time_grid, dtype=float)
        t_max_grid = float(time_grid[-1])

        def count_from_intervals(starts, ends):
            starts = np.asarray(starts, float)
            ends = np.asarray(ends, float)
            if starts.size == 0:
                return np.zeros_like(time_grid, float)
            starts.sort()
            ends.sort()
            started = np.searchsorted(starts, time_grid, side="right")
            ended = np.searchsorted(ends, time_grid, side="left")
            return (started - ended).astype(float)

        # build fork/stall intervals via traversal edges (cheap vs mÃ—n)
        pk = prev.astype(np.int64, copy=False)
        good = (pk >= 0) & np.isfinite(replication_time_min) & np.isfinite(replication_time_min[pk])
        cols = np.flatnonzero(good).astype(np.int64, copy=False)
        edge_start = replication_time_min[pk[cols]]
        edge_end = replication_time_min[cols]

        fork_ok = np.isfinite(edge_start) & np.isfinite(edge_end) & (edge_start < edge_end) & (edge_start <= t_max_grid)
        fork_starts = edge_start[fork_ok].tolist()
        fork_ends = np.minimum(edge_end[fork_ok], t_max_grid).tolist()

        d = fork_directionality[cols]
        delay = np.where(d > 0, stall_delay_left[cols], stall_delay_right[cols])
        stall_ok = np.isfinite(delay) & (delay > 0.0) & np.isfinite(edge_end) & (edge_end <= t_max_grid)
        stall_starts = (edge_end[stall_ok] - delay[stall_ok]).tolist()
        stall_ends = np.minimum(edge_end[stall_ok], t_max_grid).tolist()

        total_forks = count_from_intervals(fork_starts, fork_ends)
        stalled_forks = count_from_intervals(stall_starts, stall_ends)
        active_forks = total_forks - stalled_forks

        # firing_factors 1D (as before)
        firing_factors = None
        if use_factors:
            Amax = float(ffiring_nr)
            cost = 2.0
            firing_factors = np.empty_like(time_grid, float)

            if ffiring_forkQ:
                release = np.full(n, np.inf, float)
                if not perQ:
                    changes = np.flatnonzero(np.r_[True, src[1:] != src[:-1], True])
                    starts = changes[:-1]
                    ends = changes[1:] - 1
                    labels = src[starts].astype(np.int64, copy=False)
                    release[labels] = np.maximum(replication_time_min[starts], replication_time_min[ends])
                else:
                    for ori in fired_idx:
                        seg = np.flatnonzero(src == ori)
                        if seg.size:
                            release[ori] = np.nanmax(replication_time_min[seg])

                fire_events = np.sort(fired_times_min[np.isfinite(fired_times_min)])
                rel_events = np.sort(release[fired_idx][np.isfinite(release[fired_idx])])
                A = Amax
                fi = 0
                ri = 0
                for j, tj in enumerate(time_grid):
                    while fi < fire_events.size and fire_events[fi] <= tj:
                        A = max(0.0, A - cost)
                        fi += 1
                    while ri < rel_events.size and rel_events[ri] <= tj:
                        A = min(Amax, A + cost)
                        ri += 1
                    firing_factors[j] = A
            else:
                r = float(ffiring_recycle)
                A = Amax
                last_t = 0.0
                events = np.sort(fired_times_min[np.isfinite(fired_times_min)])

                def _recover(A_now, dt):
                    if r == 0.0 or dt <= 0.0:
                        return A_now
                    return Amax - (Amax - A_now) * np.exp(-r * dt)

                ei = 0
                for j, tj in enumerate(time_grid):
                    while ei < events.size and events[ei] <= tj:
                        te = events[ei]
                        A = _recover(A, te - last_t)
                        A = max(0.0, A - cost)
                        last_t = te
                        ei += 1
                    firing_factors[j] = _recover(A, tj - last_t)

        time_stats = dict(
            time_grid=time_grid,
            total_forks=total_forks,
            active_forks=active_forks,
            stalled_forks=stalled_forks,
            firing_factors=firing_factors,
        )

    return (
        fired_idx.astype(np.int64, copy=False),
        fired_pos_kb,
        fired_times_min,
        replication_time_min,
        interorigin_dist_kb,
        time_stats,
        fork_directionality,
        # NEW minimal extra return for post dens compilation (no mÃ—n arrays):
        t, src, prev,
    )


def rsim(
    ori_rate,
    fork_speed,
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
    time_stats_densQ=False,   # NEW
    time_grid=np.arange(0.0, 1500.0 + 1.0, 1.0),
    max_rep_time=1200.0,
    seed=None,
    verbose=True,
    print_every=1,
    dens_block_n=4096,         # NEW (space block size)
    dens_memmap_dir=None,      # NEW
):
    ori_arr = np.asarray(ori_rate, dtype=float)
    if ori_arr.ndim == 1:
        n = int(ori_arr.size)
    elif ori_arr.ndim == 2:
        n = int(ori_arr.shape[1])
    else:
        raise ValueError("ori_rate must be (n,) or (T,n).")

    sim_number = int(sim_number)

    ss = np.random.SeedSequence(seed) if seed is not None else np.random.SeedSequence()
    generators = [np.random.default_rng(s) for s in ss.spawn(sim_number)]

    sum_rep_time = np.zeros(n, dtype=float)
    count_rep_time = np.zeros(n, dtype=np.int64)

    rep_times_per_sim = np.empty((sim_number, n), dtype=float)
    nr_oris = np.empty(sim_number, dtype=np.int64)

    all_iods = []
    fire_counts = np.zeros(n, dtype=np.int64)

    sum_fdir = np.zeros(n, dtype=float)
    count_fdir = np.zeros(n, dtype=np.int64)

    if time_statsQ:
        if time_grid is None:
            raise ValueError("time_statsQ=True requires a time_grid array.")
        time_grid = np.asarray(time_grid, dtype=float)
        m = int(time_grid.size)
        sum_total_forks = np.zeros(m, float)
        sum_active_forks = np.zeros(m, float)
        sum_stalled_forks = np.zeros(m, float)
        sum_firing_factors = np.zeros(m, float)
        count_firing_factors = 0
        dt_grid_out = float(time_grid[1] - time_grid[0]) if m >= 2 else 1.0
    else:
        time_grid = None
        dt_grid_out = None
        m = None

    # --- allocate density accumulators (disk-backed) ---
    if time_stats_densQ:
        if time_grid is None:
            raise ValueError("time_stats_densQ=True requires time_grid (even if time_statsQ=False).")
        time_grid = np.asarray(time_grid, dtype=float)
        m = int(time_grid.size)
        if dens_memmap_dir is None:
            dens_memmap_dir = tempfile.gettempdir()
        os.makedirs(dens_memmap_dir, exist_ok=True)

        def _mm(name, shape, dtype):
            path = os.path.join(dens_memmap_dir, f"{name}_{int(time.time()*1e6)}.dat")
            return np.memmap(path, mode="w+", dtype=dtype, shape=shape)

        # store sums as float32 (you can change to uint16 if sim_number <= 65535)
        sum_replicated_fraction = _mm("sum_replicated_fraction", (m, n), np.float32)
        sum_initiation_events = _mm("sum_initiation_events", (m, n), np.float32)
        sum_coalescence_events = _mm("sum_coalescence_events", (m, n), np.float32)
        sum_right_moving_fork_density = _mm("sum_right_moving_fork_density", (m, n), np.float32)
        sum_left_moving_fork_density = _mm("sum_left_moving_fork_density", (m, n), np.float32)

        sum_replicated_fraction[:] = 0.0
        sum_initiation_events[:] = 0.0
        sum_coalescence_events[:] = 0.0
        sum_right_moving_fork_density[:] = 0.0
        sum_left_moving_fork_density[:] = 0.0

        dens_block_n = int(dens_block_n)
        if dens_block_n <= 0:
            raise ValueError("dens_block_n must be a positive integer.")
    else:
        sum_replicated_fraction = None
        sum_initiation_events = None
        sum_coalescence_events = None
        sum_right_moving_fork_density = None
        sum_left_moving_fork_density = None

    start_time = time.time()

    for i, g in enumerate(generators, start=1):
        fired_idx, _, _, rep_time, iod, time_stats, fdir, t_fire, src, prev = simulate_replication(
            ori_rate=ori_rate,
            fork_speed=fork_speed,
            resolution_space=resolution_space,
            resolution_time=resolution_time,
            rng=g,
            perQ=perQ,
            stall_rate=stall_rate,
            tau=tau,
            ffiring_nr=ffiring_nr,
            ffiring_recycle=ffiring_recycle,
            ffiring_forkQ=ffiring_forkQ,
            time_statsQ=time_statsQ,
            time_stats_densQ=time_stats_densQ,
            time_grid=time_grid,
            max_rep_time=max_rep_time,
        )

        rep_times_per_sim[i - 1, :] = rep_time
        nr_oris[i - 1] = int(fired_idx.size)

        finite_mask = np.isfinite(rep_time)
        sum_rep_time[finite_mask] += rep_time[finite_mask]
        count_rep_time[finite_mask] += 1

        if iod.size:
            all_iods.append(iod)
        if fired_idx.size:
            fire_counts[fired_idx] += 1

        sum_fdir[finite_mask] += fdir[finite_mask]
        count_fdir[finite_mask] += 1

        # --- 1D time series accumulate (RAM) ---
        if time_statsQ and time_stats is not None:
            sum_total_forks += time_stats["total_forks"]
            sum_active_forks += time_stats["active_forks"]
            sum_stalled_forks += time_stats["stalled_forks"]
            ff = time_stats.get("firing_factors", None)
            if ff is not None:
                sum_firing_factors += ff
                count_firing_factors += 1

        # --- 2D densities accumulate (disk memmap, block-by-block, no giant temporaries) ---
        if time_stats_densQ:
            t_max_grid = float(time_grid[-1])

            # (a) replicated_fraction: block over x
            # adds 0/1 per (t,x)
            for x0 in range(0, n, dens_block_n):
                x1 = min(n, x0 + dens_block_n)
                # (m, xb) bool, xb small -> OK
                blk = (rep_time[x0:x1][None, :] <= time_grid[:, None])
                sum_replicated_fraction[:, x0:x1] += blk.astype(np.float32)

            # (b) initiation events: sparse binning (no x-blocking needed)
            tf = t_fire
            ok = np.isfinite(tf) & (tf <= t_max_grid)
            if np.any(ok):
                cols = np.flatnonzero(ok).astype(np.int64, copy=False)
                bins = np.searchsorted(time_grid, tf[ok], side="left").astype(np.int64, copy=False)
                np.add.at(sum_initiation_events, (bins, cols), 1.0)

            # (c) coalescence events: sparse binning on boundaries
            if not perQ:
                boundary_k = (np.flatnonzero(src[1:] != src[:-1]) + 1).astype(np.int64, copy=False)
                if boundary_k.size:
                    km1 = boundary_k - 1
                    t_meet = np.maximum(rep_time[km1], rep_time[boundary_k])
                    okm = np.isfinite(t_meet) & (t_meet <= t_max_grid)
                    if np.any(okm):
                        bins = np.searchsorted(time_grid, t_meet[okm], side="left").astype(np.int64, copy=False)
                        cols = boundary_k[okm].astype(np.int64, copy=False)
                        np.add.at(sum_coalescence_events, (bins, cols), 1.0)
            else:
                boundary_k = np.flatnonzero(src[np.r_[1:n, 0]] != src).astype(np.int64, copy=False)
                if boundary_k.size:
                    km1 = (boundary_k - 1) % n
                    t_meet = np.maximum(rep_time[km1], rep_time[boundary_k])
                    okm = np.isfinite(t_meet) & (t_meet <= t_max_grid)
                    if np.any(okm):
                        bins = np.searchsorted(time_grid, t_meet[okm], side="left").astype(np.int64, copy=False)
                        cols = boundary_k[okm].astype(np.int64, copy=False)
                        np.add.at(sum_coalescence_events, (bins, cols), 1.0)

            # (d) fork densities: event-interval updates into memmap using sparse add.at
            # Instead of allocating (m+1,n) diffs, we directly update the (m,n) densities via bin ranges per column.
            # This is still efficient because each column has O(1) edges; total edges ~ n.
            # We do it blockwise in x to keep cache friendly.

            pk = prev.astype(np.int64, copy=False)
            good = (pk >= 0) & np.isfinite(rep_time) & np.isfinite(rep_time[pk])
            cols0 = np.flatnonzero(good).astype(np.int64, copy=False)
            if cols0.size:
                t_end = rep_time[cols0]
                t_start = rep_time[pk[cols0]]
                ok2 = (t_start < t_end) & (t_start <= t_max_grid)
                cols0 = cols0[ok2]
                if cols0.size:
                    t_start = t_start[ok2]
                    t_end = np.minimum(t_end[ok2], t_max_grid)
                    j0 = np.searchsorted(time_grid, t_start, side="left").astype(np.int64, copy=False)
                    j1 = np.searchsorted(time_grid, t_end, side="left").astype(np.int64, copy=False)
                    ok3 = j1 > j0
                    cols0 = cols0[ok3]
                    if cols0.size:
                        j0 = j0[ok3]
                        j1 = j1[ok3]
                        d = fdir[cols0]
                        Rm = d > 0
                        Lm = d < 0

                        # update ranges: for each (col, j0:j1) add +1
                        # do in blocks of indices to keep python overhead bounded
                        def _range_add(target_mm, cols, a, b):
                            # cols,a,b 1D arrays same length
                            for kk in range(cols.size):
                                c = int(cols[kk])
                                target_mm[a[kk]:b[kk], c] += 1.0

                        if np.any(Rm):
                            _range_add(sum_right_moving_fork_density, cols0[Rm], j0[Rm], j1[Rm])
                        if np.any(Lm):
                            _range_add(sum_left_moving_fork_density, cols0[Lm], j0[Lm], j1[Lm])

        if verbose and (i % int(print_every) == 0 or i == int(sim_number)):
            etaf(it=i, maxiter=int(sim_number), start_time=start_time)

    all_iods_kb = np.concatenate(all_iods) if all_iods else np.array([], dtype=float)

    avg_rep_time_min = np.full(n, np.inf, dtype=float)
    nonzero = count_rep_time > 0
    avg_rep_time_min[nonzero] = sum_rep_time[nonzero] / count_rep_time[nonzero].astype(float)

    fork_directionality = np.full(n, np.nan, float)
    fd_ok = count_fdir > 0
    fork_directionality[fd_ok] = sum_fdir[fd_ok] / count_fdir[fd_ok].astype(float)

    s_phase_durations_min = rep_times_per_sim.max(axis=1)
    efficiencies = fire_counts.astype(float) / float(sim_number)

    # --- finalise outputs ---
    if time_statsQ:
        invS = 1.0 / float(sim_number)
        avg_total_forks = sum_total_forks * invS
        avg_active_forks = sum_active_forks * invS
        avg_stalled_forks = sum_stalled_forks * invS
        avg_firing_factors = (sum_firing_factors / float(count_firing_factors)) if count_firing_factors > 0 else None
    else:
        avg_total_forks = None
        avg_active_forks = None
        avg_stalled_forks = None
        avg_firing_factors = None

    if time_stats_densQ:
        # divide memmaps in place; show ETA for this â€œpostâ€ step
        post_start = time.time()
        for j, x0 in enumerate(range(0, n, dens_block_n), start=1):
            x1 = min(n, x0 + dens_block_n)
            sum_replicated_fraction[:, x0:x1] /= float(sim_number)
            sum_initiation_events[:, x0:x1] /= float(sim_number)
            sum_coalescence_events[:, x0:x1] /= float(sim_number)
            sum_right_moving_fork_density[:, x0:x1] /= float(sim_number)
            sum_left_moving_fork_density[:, x0:x1] /= float(sim_number)
            if verbose:
                etaf(it=j, maxiter=(n + dens_block_n - 1) // dens_block_n, start_time=post_start)

        avg_replicated_fraction = sum_replicated_fraction
        avg_initiation_events = sum_initiation_events
        avg_coalescence_events = sum_coalescence_events
        avg_right_moving_fork_density = sum_right_moving_fork_density
        avg_left_moving_fork_density = sum_left_moving_fork_density
        dt_grid_out = float(time_grid[1] - time_grid[0]) if time_grid is not None and time_grid.size >= 2 else 1.0
    else:
        avg_replicated_fraction = None
        avg_initiation_events = None
        avg_coalescence_events = None
        avg_right_moving_fork_density = None
        avg_left_moving_fork_density = None

    return dict(
        replication_timing=avg_rep_time_min,
        rep_times_per_sim=rep_times_per_sim,
        s_phase_duration=s_phase_durations_min,
        num_oris=nr_oris.tolist(),
        inter_origin_distances=all_iods_kb,
        efficiency=efficiencies,
        fork_directionality=fork_directionality,
        time_grid=time_grid,
        dt_grid=dt_grid_out,
        replicated_fraction=avg_replicated_fraction,
        initiation_events=avg_initiation_events,
        coalescence_events=avg_coalescence_events,
        right_moving_fork_density=avg_right_moving_fork_density,
        left_moving_fork_density=avg_left_moving_fork_density,
        total_forks=avg_total_forks,
        active_forks=avg_active_forks,
        stalled_forks=avg_stalled_forks,
        firing_factors=avg_firing_factors,
    )

# %% [markdown]
# ## Master functions

# %%
class NameMapping:
    def __init__(self):
        self.long_from_short = {
            'fr':'firing_rate',
            'eff':'efficiency',
            'rt':'replication_timing',
            'iod':'inter_origin_distances',
            'rfd':'fork_directionality',
            'fs':'fork_speed',
            'fss':'fork_speeds',
            'nori':'num_oris',
            'ex':'example',
            'rg':'region'
        }
        self.short_from_long = {v:k for k,v in self.long_from_short.items()}
    
    def __getitem__(self, key):
        key = str(key)
        if key in self.long_from_short:   # short â†’ long
            return self.long_from_short[key]
        if key in self.short_from_long:   # long â†’ long (canonical)
            return key
        raise KeyError(f"Unknown key: {key}")

mapt = NameMapping()

# %%
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

# %%
rmap = DataTypeMapper()
rmap.register("firing_rate", "replication_timing", map_firing_timing)
rmap.register("replication_timing", "firing_rate", map_timing_firing)
rmap.register("replication_timing", "fork_directionality", map_timing_directionality)
rmap.register("fork_directionality", "replication_timing", map_directionality_timing)

# %%
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
        data1 = np.loadtxt(f"data/{mapt[d1]}/{mapt[d1]}_{cell_line}_chr{chr_number}.txt", dtype=float)
    data2 = rmap.map(d1, d2, data1, fork_speed=fork_speed, resolution=resolution, **kwargs)

    if saveQ:
        os.makedirs(f"data/{mapt[d2]}", exist_ok=True)
        np.savetxt(f"data/{mapt[d2]}/{mapt[d2]}_{cell_line}_chr{chr_number}.txt", data2)

    return data2

# %% [markdown]
# ## Examples

# %%
pass

