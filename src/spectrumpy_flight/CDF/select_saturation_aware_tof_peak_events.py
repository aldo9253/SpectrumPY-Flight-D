#!/usr/bin/env python3
"""Select L1B events with saturation-aware TOF peak-width criteria.

TOF H can saturate for large pulses, which makes its FWHM artificially wide.
For saturated TOF H peaks this script replaces the TOF H FWHM with the nearest
TOF M peak FWHM in the same event, if one is found within the match window.
TOF M and TOF L peaks use their own FWHM values directly.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cdflib
import numpy as np
import pandas as pd

from subset_l1b_peak_categories import (
    load_event_list,
    load_summary_events,
    matched_epochs,
    verify_subset,
    write_subset,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_L1B_DIR = SCRIPT_DIR / "l1b"
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR / "l1b_peak_analysis"
DEFAULT_SUMMARY = DEFAULT_ANALYSIS_DIR / "l1b_tof_peak_fwhm_7sig_event_summary.csv"
DEFAULT_PEAKS = DEFAULT_ANALYSIS_DIR / "l1b_tof_peak_fwhm_7sig_peaks.csv"
DEFAULT_EVENT_LIST = DEFAULT_ANALYSIS_DIR / "IDEX_Event_List.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "category_cdfs"
TOF_CHANNELS = ("tof_high", "tof_mid", "tof_low")
TOF_HIGH_SATURATION_HEIGHT_PC = 0.14
TOF_HIGH_FLAT_TOP_TOLERANCE_PC = 1.0e-6
TOF_HIGH_FLAT_TOP_MIN_SAMPLES = 3


@dataclass(frozen=True)
class Criterion:
    name: str
    min_qualifying_peaks: int
    min_width_us: float
    cdf_name: str


CRITERIA = (
    Criterion(
        name="sat_aware_tof_7sig_ge2_peaks_fwhm_ge_20ns",
        min_qualifying_peaks=2,
        min_width_us=0.020,
        cdf_name="imap_idex_l1b_sat_aware_tof_7sig_ge2peaks_fwhm_ge_20ns_v001.cdf",
    ),
    Criterion(
        name="sat_aware_tof_7sig_ge1_peak_fwhm_ge_40ns",
        min_qualifying_peaks=1,
        min_width_us=0.040,
        cdf_name="imap_idex_l1b_sat_aware_tof_7sig_ge1peak_fwhm_ge_40ns_v001.cdf",
    ),
    Criterion(
        name="sat_aware_tof_7sig_ge1_peak_fwhm_ge_20ns",
        min_qualifying_peaks=1,
        min_width_us=0.020,
        cdf_name="imap_idex_l1b_sat_aware_tof_7sig_ge1peak_fwhm_ge_20ns_v001.cdf",
    ),
)


def source_paths(l1b_dir: Path) -> dict[str, Path]:
    return {path.name: path for path in l1b_dir.glob("imap_idex_l1b_sci-10days_*.cdf")}


def load_tof_high_waveform(path: Path, record_index: int) -> np.ndarray:
    cdf = cdflib.CDF(str(path))
    return np.asarray(cdf.varget("TOF_High"), dtype=float)[record_index]


def has_flat_saturated_top(waveform: np.ndarray, sample_index: int) -> bool:
    samples = np.asarray(waveform, dtype=float)
    finite = np.isfinite(samples)
    if not np.any(finite) or sample_index < 0 or sample_index >= samples.size:
        return False
    peak_value = float(samples[sample_index])
    rail_value = float(np.nanmax(samples[finite]))
    if not np.isfinite(peak_value) or not np.isfinite(rail_value):
        return False
    if peak_value < rail_value - TOF_HIGH_FLAT_TOP_TOLERANCE_PC:
        return False

    near_top = finite & (samples >= rail_value - TOF_HIGH_FLAT_TOP_TOLERANCE_PC)
    left = sample_index
    while left > 0 and near_top[left - 1]:
        left -= 1
    right = sample_index
    while right < samples.size - 1 and near_top[right + 1]:
        right += 1
    return (right - left + 1) >= TOF_HIGH_FLAT_TOP_MIN_SAMPLES


def add_saturation_aware_widths(peaks: pd.DataFrame, l1b_dir: Path, match_window_us: float) -> pd.DataFrame:
    data = peaks[peaks["channel"].isin(TOF_CHANNELS)].copy()
    for column in ("epoch_tt2000", "record_index", "sample_index"):
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")
    for column in ("peak_time_us", "peak_height", "peak_prominence", "width_us"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["tof_high_saturated"] = False
    data["tof_mid_match_found"] = False
    data["tof_mid_match_peak_index"] = pd.NA
    data["tof_mid_match_time_us"] = math.nan
    data["tof_mid_match_width_us"] = math.nan
    data["effective_width_source"] = data["channel"]
    data["effective_width_us"] = data["width_us"]

    paths = source_paths(l1b_dir)
    waveform_cache: dict[tuple[str, int], np.ndarray] = {}
    high_indices = data.index[data["channel"].eq("tof_high")].tolist()
    grouped_mid = {
        int(epoch): group.sort_values("peak_time_us")
        for epoch, group in data[data["channel"].eq("tof_mid")].groupby("epoch_tt2000", dropna=True)
    }

    for index in high_indices:
        row = data.loc[index]
        source_file = str(row["source_file"])
        record_index = int(row["record_index"])
        sample_index = int(row["sample_index"])
        key = (source_file, record_index)
        if key not in waveform_cache:
            waveform_cache[key] = load_tof_high_waveform(paths[source_file], record_index)

        saturated = bool(row["peak_height"] >= TOF_HIGH_SATURATION_HEIGHT_PC) or has_flat_saturated_top(
            waveform_cache[key],
            sample_index,
        )
        if not saturated:
            continue

        data.at[index, "tof_high_saturated"] = True
        epoch = int(row["epoch_tt2000"])
        mid_peaks = grouped_mid.get(epoch)
        if mid_peaks is None or mid_peaks.empty:
            data.at[index, "effective_width_source"] = "tof_mid_replacement_missing"
            data.at[index, "effective_width_us"] = math.nan
            continue

        deltas = (mid_peaks["peak_time_us"] - float(row["peak_time_us"])).abs()
        nearest_index = deltas.idxmin()
        if float(deltas.loc[nearest_index]) > match_window_us:
            data.at[index, "effective_width_source"] = "tof_mid_replacement_missing"
            data.at[index, "effective_width_us"] = math.nan
            continue

        nearest = mid_peaks.loc[nearest_index]
        data.at[index, "tof_mid_match_found"] = True
        data.at[index, "tof_mid_match_peak_index"] = int(nearest["peak_index"])
        data.at[index, "tof_mid_match_time_us"] = float(nearest["peak_time_us"])
        data.at[index, "tof_mid_match_width_us"] = float(nearest["width_us"])
        data.at[index, "effective_width_source"] = "tof_mid_replacement"
        data.at[index, "effective_width_us"] = float(nearest["width_us"])

    return data


def event_rows_for_epochs(summary_events: pd.DataFrame, epochs: set[int]) -> pd.DataFrame:
    return (
        summary_events[summary_events["epoch_tt2000"].isin(epochs)]
        .sort_values("epoch_tt2000")
        .reset_index(drop=True)
    )


def event_flags(summary_events: pd.DataFrame, event_list_path: Path, tolerance_seconds: float) -> pd.DataFrame:
    flags = summary_events[["epoch_tt2000"]].drop_duplicates().copy()
    flags["in_triggered_dust_list"] = False
    flags["in_pulser_list"] = False
    if not event_list_path.exists():
        return flags

    event_list = load_event_list(event_list_path)
    manual_rows = event_list[
        event_list["Event type"].str.casefold().eq("triggered")
        & event_list["Dust category"].str.casefold().eq("dust")
    ]
    pulser_rows = event_list[event_list["Event type"].str.casefold().eq("pulser")]
    tolerance_ns = int(tolerance_seconds * 1_000_000_000)
    manual_epochs = matched_epochs(summary_events, manual_rows, tolerance_ns=tolerance_ns)
    pulser_epochs = matched_epochs(summary_events, pulser_rows, tolerance_ns=tolerance_ns)
    flags.loc[flags["epoch_tt2000"].isin(manual_epochs), "in_triggered_dust_list"] = True
    flags.loc[flags["epoch_tt2000"].isin(pulser_epochs), "in_pulser_list"] = True
    return flags


def criterion_event_table(
    criterion: Criterion,
    effective_peaks: pd.DataFrame,
    summary_events: pd.DataFrame,
    flags: pd.DataFrame,
) -> pd.DataFrame:
    qualifying = effective_peaks[effective_peaks["effective_width_us"] >= criterion.min_width_us].copy()
    counts = (
        qualifying.groupby("epoch_tt2000", as_index=False)
        .agg(
            qualifying_peak_count=("effective_width_us", "size"),
            qualifying_channel_count=("channel", "nunique"),
            max_effective_width_us=("effective_width_us", "max"),
            saturated_tof_high_qualifying_count=("tof_high_saturated", "sum"),
            tof_mid_replacement_qualifying_count=(
                "effective_width_source",
                lambda values: int(values.eq("tof_mid_replacement").sum()),
            ),
        )
    )
    selected = counts[counts["qualifying_peak_count"] >= criterion.min_qualifying_peaks].copy()
    events = summary_events.merge(selected, on="epoch_tt2000", how="inner").merge(flags, on="epoch_tt2000", how="left")
    events["criterion"] = criterion.name
    events["min_qualifying_peaks"] = criterion.min_qualifying_peaks
    events["min_width_us"] = criterion.min_width_us
    return events.sort_values("epoch_tt2000").reset_index(drop=True)


def print_breakdown(name: str, events: pd.DataFrame, total_events: int) -> None:
    pulser_count = int(events["in_pulser_list"].fillna(False).sum()) if not events.empty else 0
    manual_count = int(events["in_triggered_dust_list"].fillna(False).sum()) if not events.empty else 0
    nonpulser_count = int((~events["in_pulser_list"].fillna(False)).sum()) if not events.empty else 0
    print(
        f"{name}: {len(events)} / {total_events} events "
        f"({nonpulser_count} non-pulser, {pulser_count} pulser-list, {manual_count} triggered-dust-list)"
    )
    if not events.empty:
        month_counts = events["epoch_utc"].astype(str).str.slice(0, 7).value_counts().sort_index()
        print("  by month: " + ", ".join(f"{month}={count}" for month, count in month_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1b-dir", type=Path, default=DEFAULT_L1B_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--event-list", type=Path, default=DEFAULT_EVENT_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--match-window-us", type=float, default=0.050)
    parser.add_argument("--event-list-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--skip-cdfs", action="store_true")
    args = parser.parse_args()

    summary_events = load_summary_events(args.summary)
    peaks = pd.read_csv(args.peaks)
    effective_peaks = add_saturation_aware_widths(peaks, args.l1b_dir, args.match_window_us)
    flags = event_flags(summary_events, args.event_list, args.event_list_tolerance_seconds)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    effective_path = args.analysis_dir / "tof_saturation_aware_fwhm_7sig_peak_metrics.csv"
    effective_peaks.to_csv(effective_path, index=False)
    print(f"Wrote {effective_path}")

    all_selected: list[pd.DataFrame] = []
    for criterion in CRITERIA:
        events = criterion_event_table(criterion, effective_peaks, summary_events, flags)
        csv_path = args.analysis_dir / f"{criterion.name}_events.csv"
        events.to_csv(csv_path, index=False)
        print_breakdown(criterion.name, events, len(summary_events))
        print(f"  wrote {csv_path}")
        if not args.skip_cdfs:
            cdf_path = args.output_dir / criterion.cdf_name
            write_subset(args.l1b_dir, cdf_path, events, preserve_compression=False)
            verify_subset(cdf_path, events)
            print(f"  wrote {cdf_path}")
        all_selected.append(events)

    manifest = pd.concat(all_selected, ignore_index=True) if all_selected else pd.DataFrame()
    manifest_path = args.analysis_dir / "tof_saturation_aware_fwhm_7sig_criteria_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
