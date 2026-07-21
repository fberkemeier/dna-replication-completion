"""Data-source helpers for the replication timing notebook."""

from pathlib import Path
from urllib.request import urlretrieve
import gzip
import shutil

import numpy as np
import pandas as pd
import pybigtools

from replication_src import (
    DATA_DIR,
    TIMING_DIR,
    safe_filename,
    timing_cache_path,
    timing_curve_table,
)


DEFAULT_UCSC_REPLISEQ_BASE_URL = (
    "http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/"
    "wgEncodeUwRepliSeq"
)
DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES = {
    "BG02ES": "Bg02es",
    "BJ": "Bj",
    "HELAS3": "Helas3",
    "HEPG2": "Hepg2",
    "HUVEC": "Huvec",
    "IMR90": "Imr90",
    "K562": "K562",
    "MCF7": "Mcf7",
    "NHEK": "Nhek",
    "SKNSH": "Sknsh",
}

DEFAULT_ZHAO_HCT116_GSE_ACCESSION = "GSE137764"
DEFAULT_ZHAO_HCT116_REPLISEQ_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE137nnn/GSE137764/suppl/"
    "GSE137764_HCT_GaussiansGSE137764_mooth_scaled_autosome.mat.gz"
)
DEFAULT_ZHAO_HCT116_REPLISEQ_FILENAME_GZ = (
    "GSE137764_HCT_GaussiansGSE137764_mooth_scaled_autosome.mat.gz"
)
DEFAULT_ZHAO_HCT116_REPLISEQ_FILENAME = (
    "GSE137764_HCT_GaussiansGSE137764_mooth_scaled_autosome.mat"
)
DEFAULT_ZHAO_HCT116_SOURCE_RESOLUTION = 50_000
DEFAULT_ZHAO_HCT116_ANALYSIS_RESOLUTION = 1_000
DEFAULT_ZHAO_HCT116_S_PHASE_BINS = [f"S{i}" for i in range(1, 17)]


def ucsc_repliseq_cell_line_name(
    cell_line,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    key = str(cell_line).upper().replace("-", "").replace("_", "").replace(" ", "")
    return cell_line_names.get(key, str(cell_line))


def ucsc_repliseq_wavelet_filename(
    cell_line,
    replicate=1,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    ucsc_cell_line = ucsc_repliseq_cell_line_name(
        cell_line,
        cell_line_names=cell_line_names,
    )
    return f"wgEncodeUwRepliSeq{ucsc_cell_line}WaveSignalRep{replicate}.bigWig"


def ucsc_repliseq_wavelet_url(
    cell_line,
    base_url=DEFAULT_UCSC_REPLISEQ_BASE_URL,
    replicate=1,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    filename = ucsc_repliseq_wavelet_filename(
        cell_line,
        replicate=replicate,
        cell_line_names=cell_line_names,
    )
    return f"{base_url}/{filename}"


def ucsc_repliseq_wavelet_path(
    cell_line,
    replicate=1,
    data_dir=DATA_DIR,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    return Path(data_dir) / ucsc_repliseq_wavelet_filename(
        cell_line,
        replicate=replicate,
        cell_line_names=cell_line_names,
    )


def download_ucsc_repliseq_wavelet_bigwig(
    cell_line,
    base_url=DEFAULT_UCSC_REPLISEQ_BASE_URL,
    replicate=1,
    data_dir=DATA_DIR,
    overwrite=False,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = ucsc_repliseq_wavelet_path(
        cell_line,
        replicate=replicate,
        data_dir=data_dir,
        cell_line_names=cell_line_names,
    )
    url = ucsc_repliseq_wavelet_url(
        cell_line,
        base_url=base_url,
        replicate=replicate,
        cell_line_names=cell_line_names,
    )

    if overwrite or not path.exists():
        print(f"Downloading {url}")
        urlretrieve(url, path)
    else:
        print(f"Already available: {path}")

    return path


def download_ucsc_repliseq_wavelet_bigwigs(
    cell_lines,
    base_url=DEFAULT_UCSC_REPLISEQ_BASE_URL,
    replicate=1,
    data_dir=DATA_DIR,
    overwrite=False,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    return {
        cell_line: download_ucsc_repliseq_wavelet_bigwig(
            cell_line,
            base_url=base_url,
            replicate=replicate,
            data_dir=data_dir,
            overwrite=overwrite,
            cell_line_names=cell_line_names,
        )
        for cell_line in cell_lines
    }


def ucsc_repliseq_bigwig_status_table(bigwigs, replicate=1):
    return pd.DataFrame([
        {
            "cell_line": cell_line,
            "replicate": replicate,
            "local_bigwig": str(path),
            "exists": Path(path).exists(),
        }
        for cell_line, path in bigwigs.items()
    ])


def require_local_ucsc_repliseq_wavelet_bigwig(
    cell_line,
    replicate=1,
    data_dir=DATA_DIR,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    path = ucsc_repliseq_wavelet_path(
        cell_line,
        replicate=replicate,
        data_dir=data_dir,
        cell_line_names=cell_line_names,
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the A1 download cell first.")
    return path


def load_ucsc_repliseq_wavelet_array(
    cell_line,
    chrom,
    resolution=1_000,
    replicate=1,
    data_dir=DATA_DIR,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    path = require_local_ucsc_repliseq_wavelet_bigwig(
        cell_line=cell_line,
        replicate=replicate,
        data_dir=data_dir,
        cell_line_names=cell_line_names,
    )
    bw = pybigtools.open(str(path))
    try:
        chrom_sizes = dict(bw.chroms())
        if chrom not in chrom_sizes:
            available = ", ".join(sorted(chrom_sizes))
            raise KeyError(
                f"{chrom} is not present in {path.name}. "
                f"Available chromosomes: {available}"
            )

        chrom_end = chrom_sizes[chrom] - (chrom_sizes[chrom] % resolution)
        positions = np.arange(0, chrom_end, resolution, dtype=int)
        signal = np.asarray(
            bw.values(
                chrom,
                0,
                chrom_end,
                bins=len(positions),
                summary="mean",
                exact=True,
                missing=np.nan,
            ),
            dtype=float,
        )
    finally:
        bw.close()

    return signal, positions, path


def ucsc_repliseq_line_config(cell_line, chrom, positions, resolution, bigwig_path):
    start = int(positions[0])
    end = int(positions[-1] + resolution)
    key = f"{cell_line}_line_{safe_filename(chrom)}"

    return {
        "key": key,
        "label": f"{cell_line}, {chrom} UCSC Repli-seq wavelet profile",
        "short_label": f"{cell_line} {chrom} UCSC wavelet",
        "point_label": f"{cell_line} {chrom}",
        "cell_line": cell_line,
        "chrom": chrom,
        "requested_chrom": chrom,
        "start": start,
        "end": end,
        "resolution": resolution,
        "fit_periodic": False,
        "sim_periodic": False,
        "bound_geometries": ["line"],
        "analysis_type": "ucsc_wavelet_line_profile",
        "source_bigwig": str(bigwig_path),
    }


def ucsc_repliseq_timing_table(
    cfg,
    positions,
    wavelet_signal,
    raw_wavelet_signal=None,
    centromere_fill_mask=None,
):
    positions = np.asarray(positions, dtype=np.int64)
    wavelet_signal = np.asarray(wavelet_signal, dtype=float)
    if raw_wavelet_signal is None:
        raw_wavelet_signal = wavelet_signal
    raw_wavelet_signal = np.asarray(raw_wavelet_signal, dtype=float)
    if centromere_fill_mask is None:
        centromere_fill_mask = np.zeros(wavelet_signal.size, dtype=bool)
    centromere_fill_mask = np.asarray(centromere_fill_mask, dtype=bool)
    resolution = int(cfg["resolution"])

    return pd.DataFrame({
        "cell_line": cfg["cell_line"],
        "analysis_type": cfg["analysis_type"],
        "chrom": cfg["chrom"],
        "requested_chrom": cfg.get("requested_chrom", cfg["chrom"]),
        "region_start_bp": int(cfg["start"]),
        "region_end_bp": int(cfg["end"]),
        "resolution_bp": resolution,
        "bin_start_bp": positions,
        "bin_end_bp": positions + resolution,
        "source_signal": "UCSC WaveSignal",
        "centromere_handling": "none_in_cache",
        "timing_wavelet_signal_raw": raw_wavelet_signal,
        "timing_wavelet_signal": wavelet_signal,
        "centromere_interpolated": centromere_fill_mask,
        "timing_s_phase_midpoint": wavelet_signal,
    })


def build_ucsc_repliseq_line_datasets(
    cell_lines,
    chroms,
    resolution=1_000,
    replicate=1,
    data_dir=DATA_DIR,
    timing_dir=TIMING_DIR,
    overwrite_timing_cache=False,
    cell_line_names=DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES,
):
    configs = {}
    arrays = {}
    rows = []

    for cell_line in cell_lines:
        for chrom in chroms:
            signal, positions, bigwig = load_ucsc_repliseq_wavelet_array(
                cell_line=cell_line,
                chrom=chrom,
                resolution=resolution,
                replicate=replicate,
                data_dir=data_dir,
                cell_line_names=cell_line_names,
            )
            fill_mask = np.zeros(signal.size, dtype=bool)

            cfg = ucsc_repliseq_line_config(
                cell_line,
                chrom,
                positions,
                resolution,
                bigwig,
            )
            key = cfg["key"]
            cache_path = timing_cache_path(cfg, timing_dir=timing_dir)
            timing_table = ucsc_repliseq_timing_table(
                cfg,
                positions,
                signal,
                raw_wavelet_signal=signal,
                centromere_fill_mask=fill_mask,
            )
            cache_needs_update = True
            if cache_path.exists() and not overwrite_timing_cache:
                cache_columns = set(pd.read_csv(cache_path, nrows=0).columns)
                required_columns = {
                    "timing_wavelet_signal_raw",
                    "centromere_interpolated",
                    "centromere_handling",
                }
                cache_needs_update = not required_columns.issubset(cache_columns)

            if overwrite_timing_cache or not cache_path.exists() or cache_needs_update:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                timing_table.to_csv(cache_path, index=False)

            configs[key] = cfg
            arrays[key] = {
                "signal": signal,
                "raw_signal": signal,
                "positions": positions,
                "bigwig": bigwig,
                "cache_path": cache_path,
                "centromere_fill_mask": fill_mask,
            }
            rows.append({
                "key": key,
                "cell_line": cell_line,
                "chromosome": chrom,
                "resolution_bp": resolution,
                "bins": signal.size,
                "centromere_interpolated_bins": int(fill_mask.sum()),
                "centromere_interpolated_bp": int(fill_mask.sum() * resolution),
                "source_bigwig": str(bigwig),
                "raw_timing_csv": str(cache_path),
            })

    return configs, arrays, pd.DataFrame(rows)


def default_zhao_hct116_repliseq_paths(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    return (
        data_dir / DEFAULT_ZHAO_HCT116_REPLISEQ_FILENAME_GZ,
        data_dir / DEFAULT_ZHAO_HCT116_REPLISEQ_FILENAME,
    )


def ensure_zhao_hct116_repliseq(
    url=DEFAULT_ZHAO_HCT116_REPLISEQ_URL,
    gz_path=None,
    table_path=None,
    data_dir=DATA_DIR,
    overwrite=False,
):
    if gz_path is None or table_path is None:
        default_gz_path, default_table_path = default_zhao_hct116_repliseq_paths(
            data_dir=data_dir,
        )
        gz_path = default_gz_path if gz_path is None else Path(gz_path)
        table_path = default_table_path if table_path is None else Path(table_path)
    else:
        gz_path = Path(gz_path)
        table_path = Path(table_path)

    gz_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite or not gz_path.exists():
        print(f"Downloading {DEFAULT_ZHAO_HCT116_GSE_ACCESSION} HCT116 Repli-seq matrix -> {gz_path}")
        urlretrieve(url, gz_path)
    else:
        print(f"Already available: {gz_path}")

    if overwrite or not table_path.exists():
        print(f"Decompressing -> {table_path}")
        with gzip.open(gz_path, "rb") as src, open(table_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        print(f"Already decompressed: {table_path}")

    return table_path


def load_zhao_hct116_repliseq_matrix(
    path,
    s_phase_bins=DEFAULT_ZHAO_HCT116_S_PHASE_BINS,
    source_resolution=DEFAULT_ZHAO_HCT116_SOURCE_RESOLUTION,
):
    rows = []
    with open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(line.rstrip("\r\n").split("\t"))

    if len(rows) != 3 + len(s_phase_bins):
        raise ValueError(f"Expected {3 + len(s_phase_bins)} rows in {path}, found {len(rows)}")

    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows[:3]):
        raise ValueError(
            "The coordinate rows in the Zhao HCT116 matrix have inconsistent column counts"
        )

    chroms = np.asarray(rows[0], dtype=str)
    starts = np.asarray(rows[1], dtype=np.int64)
    ends = np.asarray(rows[2], dtype=np.int64)

    signal_rows = []
    for row in rows[3:]:
        if len(row) != n_cols:
            raise ValueError(
                "A signal row in the Zhao HCT116 matrix has an unexpected column count"
            )
        signal_rows.append([float(value) if value else np.nan for value in row])

    signals = np.asarray(signal_rows, dtype=float)
    return {
        "chroms": chroms,
        "starts": starts,
        "ends": ends,
        "signals": signals,
        "s_phase_bins": list(s_phase_bins),
        "source_resolution": source_resolution,
    }


def zhao_hct116_timing_from_signals(signals):
    signals = np.asarray(signals, dtype=float)
    bin_midpoints = np.arange(1, signals.shape[0] + 1, dtype=float)
    clean_signals = np.where(np.isfinite(signals), np.maximum(signals, 0.0), 0.0)
    weights = clean_signals.sum(axis=0)
    timing = np.full(signals.shape[1], np.nan, dtype=float)
    usable = weights > 0
    timing[usable] = (
        clean_signals[:, usable] * bin_midpoints[:, None]
    ).sum(axis=0) / weights[usable]
    return timing


def zhao_hct116_region_timing(
    cfg,
    matrix,
    output_resolution=DEFAULT_ZHAO_HCT116_ANALYSIS_RESOLUTION,
):
    chrom = cfg.get("requested_chrom", cfg["chrom"])
    chrom_mask = matrix["chroms"] == chrom
    if not np.any(chrom_mask):
        available = ", ".join(sorted(set(matrix["chroms"])))
        raise KeyError(
            f"{chrom} is not present in the Zhao HCT116 matrix. "
            f"Available chromosomes: {available}"
        )

    source_centers = (
        matrix["starts"][chrom_mask] + matrix["ends"][chrom_mask]
    ) / 2.0
    source_timing = zhao_hct116_timing_from_signals(matrix["signals"][:, chrom_mask])
    finite = np.isfinite(source_timing)
    if finite.sum() < 2:
        raise ValueError(f"Not enough finite HCT116 timing values for {chrom}")

    positions = np.arange(
        int(cfg["start"]),
        int(cfg["end"]),
        int(output_resolution),
        dtype=np.int64,
    )
    target_centers = positions + output_resolution / 2.0
    timing = np.interp(target_centers, source_centers[finite], source_timing[finite])
    return positions, timing


def build_zhao_hct116_periodic_configs(
    regions,
    resolution=DEFAULT_ZHAO_HCT116_ANALYSIS_RESOLUTION,
    accession=DEFAULT_ZHAO_HCT116_GSE_ACCESSION,
    source_resolution=DEFAULT_ZHAO_HCT116_SOURCE_RESOLUTION,
    cell_line="HCT116",
):
    configs = {}
    for region in regions:
        requested_chrom = region["chrom"]
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
            "chrom": requested_chrom,
            "requested_chrom": requested_chrom,
            "start": int(region["start"]),
            "end": int(region["end"]),
            "resolution": int(resolution),
            "fit_periodic": True,
            "sim_periodic": True,
            "bound_geometries": ["torus"],
            "analysis_type": "zhao_hct116_periodic_interval",
            "data_source": accession,
            "source_resolution": source_resolution,
            "timing_method": "weighted_mean_S1_to_S16_then_linear_interpolation",
        }
    return configs


def save_zhao_hct116_timing_csv(
    cfg,
    matrix,
    timing_dir=TIMING_DIR,
    overwrite=False,
):
    path = timing_cache_path(cfg, timing_dir=timing_dir)
    if path.exists() and not overwrite:
        print(f"Timing CSV already exists: {path}")
        return path, pd.read_csv(path), "exists"

    positions, timing_raw = zhao_hct116_region_timing(
        cfg,
        matrix,
        output_resolution=cfg["resolution"],
    )
    table = timing_curve_table(cfg, positions, timing_raw)
    table["source_accession"] = cfg["data_source"]
    table["source_resolution_bp"] = cfg["source_resolution"]
    table["timing_method"] = cfg["timing_method"]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"Wrote raw timing CSV: {path} ({len(table):,} bins)")
    return path, table, "written"


def preprocess_zhao_hct116_timing_collection(
    configs,
    matrix,
    timing_dir=TIMING_DIR,
    overwrite=False,
):
    rows = []
    for key, cfg in configs.items():
        path, table, status = save_zhao_hct116_timing_csv(
            cfg,
            matrix,
            timing_dir=timing_dir,
            overwrite=overwrite,
        )
        rows.append({
            "dataset_key": key,
            "dataset_label": cfg["label"],
            "raw_timing_csv": str(path),
            "bins": len(table),
            "status": status,
        })
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_UCSC_REPLISEQ_BASE_URL",
    "DEFAULT_UCSC_REPLISEQ_CELL_LINE_NAMES",
    "DEFAULT_ZHAO_HCT116_ANALYSIS_RESOLUTION",
    "DEFAULT_ZHAO_HCT116_GSE_ACCESSION",
    "DEFAULT_ZHAO_HCT116_REPLISEQ_FILENAME",
    "DEFAULT_ZHAO_HCT116_REPLISEQ_FILENAME_GZ",
    "DEFAULT_ZHAO_HCT116_REPLISEQ_URL",
    "DEFAULT_ZHAO_HCT116_S_PHASE_BINS",
    "DEFAULT_ZHAO_HCT116_SOURCE_RESOLUTION",
    "build_ucsc_repliseq_line_datasets",
    "build_zhao_hct116_periodic_configs",
    "default_zhao_hct116_repliseq_paths",
    "download_ucsc_repliseq_wavelet_bigwig",
    "download_ucsc_repliseq_wavelet_bigwigs",
    "ensure_zhao_hct116_repliseq",
    "load_ucsc_repliseq_wavelet_array",
    "load_zhao_hct116_repliseq_matrix",
    "preprocess_zhao_hct116_timing_collection",
    "require_local_ucsc_repliseq_wavelet_bigwig",
    "save_zhao_hct116_timing_csv",
    "ucsc_repliseq_bigwig_status_table",
    "ucsc_repliseq_cell_line_name",
    "ucsc_repliseq_line_config",
    "ucsc_repliseq_timing_table",
    "ucsc_repliseq_wavelet_filename",
    "ucsc_repliseq_wavelet_path",
    "ucsc_repliseq_wavelet_url",
    "zhao_hct116_region_timing",
    "zhao_hct116_timing_from_signals",
]
