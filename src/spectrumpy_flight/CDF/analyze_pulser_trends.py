#!/usr/bin/env python3
"""Trend pulser response amplitudes versus injection level and mission time.

The pulser population is selected from two sources by default: the same L1B
quicklook-style trigger classifier used by the noise-capture analysis, plus
``IDEX_Event_List.csv`` rows whose Event type is Pulser. The event list is
important for early injection tests where the HG trigger level differs from the
later quicklook pulser threshold while the L2A charge sequence is clearly a
pulser sweep. Quicklook-only records are kept only when the L2A ion-grid charge
is in the pulser-like charge range by default.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cdflib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cdflib import cdfwrite

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_L1B_DIR = SCRIPT_DIR / "l1b"
DEFAULT_L2A_DIR = SCRIPT_DIR / "l2a"
DEFAULT_EVENT_LIST = SCRIPT_DIR / "l1b_peak_analysis" / "IDEX_Event_List.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pulser_trend_analysis"
FILL_LIMIT = -1.0e30

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_l1b_noise_captures import CHANNELS, CHANNEL_COLORS, classify_event, clean_waveform, epoch_to_datetime
from subset_l1b_peak_categories import cdf_info_value, concat_records, global_attrs_for_output, var_spec, verify_subset

TRIGGER_CANDIDATES = (
    "trigger_origin",
    "trigger_mode_hg",
    "trigger_mode_mg",
    "trigger_mode_lg",
    "trigger_level_hg",
    "trigger_level_mg",
    "trigger_level_lg",
    "idx__txhdrtrigid",
    "idx__txhdrhgtrigmode",
    "idx__txhdrmgtrigmode",
    "idx__txhdrlgtrigmode",
    "idx__txhdrhgtrigctrl1",
    "idx__txhdrmgtrigctrl1",
    "idx__txhdrlgtrigctrl1",
)

L2A_VALUE_VARS = (
    "target_low_impact_charge",
    "target_high_impact_charge",
    "ion_grid_impact_charge",
    "target_low_chi_squared",
    "target_high_chi_squared",
    "ion_grid_chi_squared",
    "target_low_reduced_chi_squared",
    "target_high_reduced_chi_squared",
    "ion_grid_reduced_chi_squared",
    "tof_snr",
)

FIT_CHANNELS: tuple[tuple[str, str, str], ...] = (
    ("Target H", "Target_High", "target_high_fit_results"),
    ("Target L", "Target_Low", "target_low_fit_results"),
    ("Ion Grid", "Ion_Grid", "ion_grid_fit_results"),
)

PULSER_CDF_VARIABLES = (
    "epoch",
    "time_high_sample_rate",
    "time_low_sample_rate",
    "TOF_High",
    "TOF_Mid",
    "TOF_Low",
    "Target_High",
    "Target_Low",
    "Ion_Grid",
    "trigger_origin",
    "trigger_mode_hg",
    "trigger_level_hg",
    "trigger_mode_mg",
    "trigger_level_mg",
    "trigger_mode_lg",
    "trigger_level_lg",
    "shcoarse",
    "shfine",
    "aid",
    "ephemeris_position_x",
    "ephemeris_position_y",
    "ephemeris_position_z",
    "ephemeris_velocity_x",
    "ephemeris_velocity_y",
    "ephemeris_velocity_z",
    "longitude",
    "latitude",
    "spin_phase",
    "solar_longitude",
)

PULSER_L2A_CDF_VARIABLES = (
    "target_low_fit_parameters",
    "target_low_fit_results",
    "target_low_impact_charge",
    "target_low_chi_squared",
    "target_low_reduced_chi_squared",
    "target_high_fit_parameters",
    "target_high_fit_results",
    "target_high_impact_charge",
    "target_high_chi_squared",
    "target_high_reduced_chi_squared",
    "ion_grid_fit_parameters",
    "ion_grid_fit_results",
    "ion_grid_impact_charge",
    "ion_grid_chi_squared",
    "ion_grid_reduced_chi_squared",
    "tof_peak_fit_parameters",
    "tof_peak_area_under_fit",
    "tof_peak_chi_square",
    "tof_peak_reduced_chi_square",
    "tof_snr",
    "peak_fit_parameter_labels",
    "target_fit_parameter_labels",
    "peak_fit_parameter_index",
    "target_fit_parameter_index",
)


def apply_style() -> None:
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    try:
        from plot_style import apply_plot_style
    except Exception:
        plt.style.use("default")
    else:
        apply_plot_style("light")


def cdf_sources(input_dir: Path, level: str) -> list[Path]:
    sources = sorted(input_dir.glob(f"imap_idex_{level}_sci-10days_*.cdf"))
    if not sources:
        raise FileNotFoundError(f"No {level.upper()} 10-day CDF files found in {input_dir}")
    return sources


def valid_scalar(value: object) -> float:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return math.nan
    if not np.isfinite(scalar) or scalar <= FILL_LIMIT:
        return math.nan
    return scalar


def encoded_utc(epoch: int) -> str:
    encoded = cdflib.cdfepoch.encode_tt2000(int(epoch))
    if isinstance(encoded, list):
        encoded = encoded[0]
    return str(encoded)


def load_l1b_event_index(sources: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in sources:
        cdf = cdflib.CDF(str(source))
        cdf_vars = {"epoch": np.asarray(cdf.varget("epoch"), dtype=np.int64)}
        for varname in TRIGGER_CANDIDATES:
            try:
                cdf_vars[varname] = np.asarray(cdf.varget(varname))
            except Exception:
                continue
        for record_index, epoch in enumerate(cdf_vars["epoch"]):
            classification = classify_event(cdf_vars, record_index)
            rows.append(
                {
                    "source_file": source.name,
                    "record_index": int(record_index),
                    "epoch_tt2000": int(epoch),
                    "epoch_utc": encoded_utc(int(epoch)),
                    "event_time": epoch_to_datetime(int(epoch)),
                    "quicklook_classification": classification,
                }
            )
    return pd.DataFrame(rows).sort_values("epoch_tt2000").reset_index(drop=True)


def load_event_list_pulser_epochs(
    event_list: Path,
    l1b_events: pd.DataFrame,
    tolerance_seconds: float,
) -> set[int]:
    if not event_list.exists():
        return set()
    events = pd.read_csv(event_list, header=1)
    event_type = events.get("Event type")
    event_time = events.get("Time")
    if event_type is None or event_time is None:
        return set()
    pulser_times = pd.to_datetime(
        events.loc[event_type.astype(str).str.strip().str.casefold().eq("pulser"), "Time"],
        errors="coerce",
    ).dropna()
    if pulser_times.empty:
        return set()

    sorted_events = l1b_events[["epoch_tt2000", "event_time"]].sort_values("event_time").reset_index(drop=True)
    event_ns = sorted_events["event_time"].astype("datetime64[ns]").astype(np.int64).to_numpy()
    matched: set[int] = set()
    tolerance_ns = int(tolerance_seconds * 1_000_000_000)
    for pulser_time in pulser_times:
        target = np.datetime64(pulser_time.to_datetime64()).astype("datetime64[ns]").astype(np.int64)
        insertion = int(np.searchsorted(event_ns, target))
        candidates = [idx for idx in (insertion - 1, insertion) if 0 <= idx < len(sorted_events)]
        if not candidates:
            continue
        best_idx = min(candidates, key=lambda idx: abs(int(event_ns[idx]) - int(target)))
        if abs(int(event_ns[best_idx]) - int(target)) <= tolerance_ns:
            matched.add(int(sorted_events.loc[best_idx, "epoch_tt2000"]))
    return matched


def load_l2a_table(sources: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source in sources:
        cdf = cdflib.CDF(str(source))
        epochs = np.asarray(cdf.varget("epoch"), dtype=np.int64)
        values: dict[str, np.ndarray] = {}
        for varname in L2A_VALUE_VARS:
            try:
                values[varname] = np.asarray(cdf.varget(varname))
            except Exception:
                continue
        for record_index, epoch in enumerate(epochs):
            row: dict[str, object] = {
                "l2a_source_file": source.name,
                "l2a_record_index": int(record_index),
                "epoch_tt2000": int(epoch),
            }
            for varname, arr in values.items():
                row[f"l2a_{varname}"] = valid_scalar(arr[record_index])
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("epoch_tt2000")


def infer_injection_level(ion_grid_charge: float) -> float:
    if not np.isfinite(ion_grid_charge) or ion_grid_charge < 0.05:
        return math.nan
    thresholds = np.asarray([0.43, 0.72, 1.00, 1.30], dtype=float)
    return float(np.searchsorted(thresholds, ion_grid_charge, side="right") + 1)


def add_test_and_level_columns(events: pd.DataFrame, test_gap_minutes: float) -> pd.DataFrame:
    events = events.sort_values("event_time").reset_index(drop=True).copy()
    gaps = events["event_time"].diff().dt.total_seconds().fillna(np.inf)
    events["seconds_since_previous_pulser"] = gaps.replace(np.inf, np.nan)
    events["test_id"] = (gaps > test_gap_minutes * 60.0).cumsum().astype(int)
    events["injection_index"] = events.groupby("test_id").cumcount() + 1
    test_start = events.groupby("test_id")["event_time"].transform("min")
    events["test_start_time"] = test_start
    events["test_start_utc"] = test_start.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    events["l2a_inferred_injection_level"] = events["l2a_ion_grid_impact_charge"].map(infer_injection_level)
    events["pulser_like_l2a_charge"] = events["l2a_inferred_injection_level"].notna()
    events["injection_level"] = events["l2a_inferred_injection_level"]
    missing_level = events["injection_level"].isna()
    events["injection_level_source"] = "l2a_ion_grid_charge"
    events.loc[missing_level, "injection_level"] = events.loc[missing_level, "injection_index"].astype(float)
    events.loc[missing_level, "injection_level_source"] = "sequence_fallback"
    events["injection_level"] = events["injection_level"].astype(int)
    events["injection_repeat_index"] = events.groupby(["test_id", "injection_level"]).cumcount() + 1
    events["test_event_count"] = events.groupby("test_id")["epoch_tt2000"].transform("size")
    return events


def baseline_slice(time_axis: np.ndarray, waveform: np.ndarray, baseline_samples: int) -> np.ndarray:
    finite_time = np.asarray(time_axis, dtype=float).ravel()
    finite_wave = clean_waveform(waveform)
    if finite_time.size == finite_wave.size:
        before_trigger = finite_wave[np.isfinite(finite_wave) & np.isfinite(finite_time) & (finite_time < 0.0)]
        if before_trigger.size >= max(10, baseline_samples // 4):
            return before_trigger[:baseline_samples]
    return finite_wave[np.isfinite(finite_wave)][:baseline_samples]


def waveform_metrics(waveform: np.ndarray, time_axis: np.ndarray, baseline_samples: int) -> dict[str, float]:
    wave = clean_waveform(waveform)
    baseline_values = baseline_slice(time_axis, wave, baseline_samples)
    baseline = float(np.nanmedian(baseline_values)) if baseline_values.size else math.nan
    residual = wave - baseline
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {
            "baseline": baseline,
            "positive_peak_amplitude": math.nan,
            "negative_peak_amplitude": math.nan,
            "absolute_peak_amplitude": math.nan,
            "positive_peak_time_us": math.nan,
            "absolute_peak_time_us": math.nan,
        }

    positive_index = int(np.nanargmax(residual))
    absolute_index = int(np.nanargmax(np.abs(residual)))
    time = np.asarray(time_axis, dtype=float).ravel()
    return {
        "baseline": baseline,
        "positive_peak_amplitude": float(np.nanmax(residual)),
        "negative_peak_amplitude": float(np.nanmin(residual)),
        "absolute_peak_amplitude": float(np.nanmax(np.abs(residual))),
        "positive_peak_time_us": float(time[positive_index]) if positive_index < time.size else math.nan,
        "absolute_peak_time_us": float(time[absolute_index]) if absolute_index < time.size else math.nan,
    }


def collect_channel_metrics(
    selected_events: pd.DataFrame,
    l1b_sources: list[Path],
    baseline_samples: int,
) -> pd.DataFrame:
    source_paths = {source.name: source for source in l1b_sources}
    rows: list[dict[str, object]] = []
    event_columns = [
        "source_file",
        "record_index",
        "epoch_tt2000",
        "epoch_utc",
        "event_time",
        "test_id",
        "test_start_time",
        "test_start_utc",
        "injection_index",
        "injection_level",
        "l2a_inferred_injection_level",
        "pulser_like_l2a_charge",
        "injection_level_source",
        "injection_repeat_index",
        "test_event_count",
        "quicklook_classification",
        "selection_source",
        "l2a_source_file",
        "l2a_record_index",
        "l2a_target_low_impact_charge",
        "l2a_target_high_impact_charge",
        "l2a_ion_grid_impact_charge",
        "l2a_target_low_chi_squared",
        "l2a_target_high_chi_squared",
        "l2a_ion_grid_chi_squared",
        "l2a_target_low_reduced_chi_squared",
        "l2a_target_high_reduced_chi_squared",
        "l2a_ion_grid_reduced_chi_squared",
        "l2a_tof_snr",
    ]
    for source_name, source_events in selected_events.groupby("source_file", sort=True):
        cdf = cdflib.CDF(str(source_paths[str(source_name)]))
        high_time = np.asarray(cdf.varget("time_high_sample_rate"), dtype=float)
        low_time = np.asarray(cdf.varget("time_low_sample_rate"), dtype=float)
        waveforms = {var_name: np.asarray(cdf.varget(var_name)) for _label, var_name in CHANNELS}
        for event in source_events.itertuples(index=False):
            event_dict = event._asdict()
            record_index = int(event_dict["record_index"])
            for channel_label, var_name in CHANNELS:
                time_axis = high_time[record_index] if channel_label.startswith("TOF") else low_time[record_index]
                metrics = waveform_metrics(waveforms[var_name][record_index], time_axis, baseline_samples)
                if channel_label == "Ion Grid":
                    response_peak_amplitude = -metrics["negative_peak_amplitude"]
                    response_peak_polarity = "negative"
                    response_peak_time_us = metrics["absolute_peak_time_us"]
                else:
                    response_peak_amplitude = metrics["positive_peak_amplitude"]
                    response_peak_polarity = "positive"
                    response_peak_time_us = metrics["positive_peak_time_us"]
                row = {column: event_dict.get(column) for column in event_columns}
                row.update(
                    {
                        "channel": channel_label,
                        "cdf_variable": var_name,
                        "time_axis": "time_high_sample_rate" if channel_label.startswith("TOF") else "time_low_sample_rate",
                        "response_peak_amplitude": response_peak_amplitude,
                        "response_peak_polarity": response_peak_polarity,
                        "response_peak_time_us": response_peak_time_us,
                        **metrics,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def collect_fit_residual_metrics(
    selected_events: pd.DataFrame,
    l1b_sources: list[Path],
    l2a_sources: list[Path],
) -> pd.DataFrame:
    l1b_paths = {source.name: source for source in l1b_sources}
    l2a_paths = {source.name: source for source in l2a_sources}
    rows: list[dict[str, object]] = []
    event_columns = [
        "source_file",
        "record_index",
        "epoch_tt2000",
        "epoch_utc",
        "event_time",
        "test_id",
        "test_start_time",
        "test_start_utc",
        "injection_index",
        "injection_level",
        "l2a_inferred_injection_level",
        "pulser_like_l2a_charge",
        "injection_level_source",
        "injection_repeat_index",
        "test_event_count",
        "quicklook_classification",
        "selection_source",
        "l2a_source_file",
        "l2a_record_index",
        "l2a_target_low_impact_charge",
        "l2a_target_high_impact_charge",
        "l2a_ion_grid_impact_charge",
        "l2a_target_low_chi_squared",
        "l2a_target_high_chi_squared",
        "l2a_ion_grid_chi_squared",
        "l2a_target_low_reduced_chi_squared",
        "l2a_target_high_reduced_chi_squared",
        "l2a_ion_grid_reduced_chi_squared",
    ]
    for (l1b_source_name, l2a_source_name), source_events in selected_events.groupby(
        ["source_file", "l2a_source_file"], sort=True
    ):
        if pd.isna(l2a_source_name):
            continue
        l1b_cdf = cdflib.CDF(str(l1b_paths[str(l1b_source_name)]))
        l2a_cdf = cdflib.CDF(str(l2a_paths[str(l2a_source_name)]))
        l1b_waveforms = {var_name: np.asarray(l1b_cdf.varget(var_name), dtype=float) for _label, var_name, _fit in FIT_CHANNELS}
        l2a_fits = {fit_var: np.asarray(l2a_cdf.varget(fit_var), dtype=float) for _label, _var, fit_var in FIT_CHANNELS}
        for event in source_events.itertuples(index=False):
            event_dict = event._asdict()
            l1b_index = int(event_dict["record_index"])
            l2a_index = int(event_dict["l2a_record_index"])
            for channel_label, waveform_var, fit_var in FIT_CHANNELS:
                waveform = clean_waveform(l1b_waveforms[waveform_var][l1b_index])
                fit = clean_waveform(l2a_fits[fit_var][l2a_index])
                sample_count = min(waveform.size, fit.size)
                if sample_count == 0:
                    continue
                residual = waveform[:sample_count] - fit[:sample_count]
                finite = residual[np.isfinite(residual)]
                if finite.size == 0:
                    rms = math.nan
                    mad = math.nan
                    max_abs = math.nan
                    residual_median = math.nan
                else:
                    rms = float(np.sqrt(np.nanmean(finite**2)))
                    residual_median = float(np.nanmedian(finite))
                    mad = float(1.4826 * np.nanmedian(np.abs(finite - residual_median)))
                    max_abs = float(np.nanmax(np.abs(finite)))
                row = {column: event_dict.get(column) for column in event_columns}
                row.update(
                    {
                        "channel": channel_label,
                        "waveform_variable": waveform_var,
                        "fit_variable": fit_var,
                        "fit_residual_rms": rms,
                        "fit_residual_robust_sigma": mad,
                        "fit_residual_median": residual_median,
                        "fit_residual_max_abs": max_abs,
                        "fit_residual_sample_count": int(finite.size),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_tests(selected_events: pd.DataFrame) -> pd.DataFrame:
    return (
        selected_events.groupby("test_id", as_index=False)
        .agg(
            test_start_utc=("test_start_utc", "first"),
            test_end_utc=("epoch_utc", "last"),
            event_count=("epoch_tt2000", "size"),
            first_epoch_tt2000=("epoch_tt2000", "first"),
            last_epoch_tt2000=("epoch_tt2000", "last"),
            min_injection_level=("injection_level", "min"),
            max_injection_level=("injection_level", "max"),
            unique_injection_levels=("injection_level", "nunique"),
            event_list_selected=("selection_source", lambda values: int(any("event_list" in str(v) for v in values))),
            quicklook_selected=("selection_source", lambda values: int(any("quicklook" in str(v) for v in values))),
        )
        .sort_values("test_id")
    )


def median_channel_level(metrics: pd.DataFrame, value_column: str) -> pd.DataFrame:
    data = metrics.copy()
    data["test_start_time"] = pd.to_datetime(data["test_start_time"], errors="coerce")
    return (
        data.groupby(["channel", "test_id", "test_start_time", "test_start_utc", "injection_level"], as_index=False)
        .agg(
            response=(value_column, "median"),
            response_std=(value_column, "std"),
            event_count=("epoch_tt2000", "nunique"),
        )
        .sort_values(["channel", "injection_level", "test_start_time"])
    )


def setup_date_axis(ax) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_channel_response(metrics: pd.DataFrame, output: Path, value_column: str) -> None:
    data = median_channel_level(metrics, value_column)
    levels = sorted(int(level) for level in data["injection_level"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(levels), 1))
    colors = {level: cmap(idx / max(len(levels) - 1, 1)) for idx, level in enumerate(levels)}

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 10.5), sharex=True)
    for ax, (channel_label, _var_name) in zip(axes.ravel(), CHANNELS):
        channel_data = data[data["channel"] == channel_label]
        for level in levels:
            level_data = channel_data[channel_data["injection_level"] == level]
            if level_data.empty:
                continue
            ax.plot(
                level_data["test_start_time"],
                level_data["response"],
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                alpha=0.82,
                color=colors[level],
                label=f"Level {level}",
            )
        ax.set_title(channel_label)
        ax.set_ylabel("Median response [pC]")
        ax.grid(True, alpha=0.35)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    for ax in axes[-1, :]:
        ax.set_xlabel("Pulser test time")
        setup_date_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("L1B Pulser Response by Injection Level", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(len(levels), 5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_normalized_channel_response(metrics: pd.DataFrame, output: Path, value_column: str) -> None:
    data = median_channel_level(metrics, value_column)
    data["first_response"] = data.groupby(["channel", "injection_level"])["response"].transform("first")
    data["normalized_response"] = data["response"] / data["first_response"]
    data = data[np.isfinite(data["normalized_response"])]
    levels = sorted(int(level) for level in data["injection_level"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(levels), 1))
    colors = {level: cmap(idx / max(len(levels) - 1, 1)) for idx, level in enumerate(levels)}

    fig, axes = plt.subplots(3, 2, figsize=(15.5, 10.5), sharex=True)
    for ax, (channel_label, _var_name) in zip(axes.ravel(), CHANNELS):
        channel_data = data[data["channel"] == channel_label]
        for level in levels:
            level_data = channel_data[channel_data["injection_level"] == level]
            if level_data.empty:
                continue
            ax.plot(
                level_data["test_start_time"],
                level_data["normalized_response"],
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                alpha=0.82,
                color=colors[level],
                label=f"Level {level}",
            )
        ax.axhline(1.0, color="#666666", linewidth=0.9, alpha=0.55)
        ax.set_title(channel_label)
        ax.set_ylabel("Response / first test")
        ax.grid(True, alpha=0.35)
    for ax in axes[-1, :]:
        ax.set_xlabel("Pulser test time")
        setup_date_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Normalized L1B Pulser Response by Injection Level", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=min(len(levels), 5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_l2a_charges(selected_events: pd.DataFrame, output: Path) -> None:
    charge_columns = (
        ("Target H", "l2a_target_high_impact_charge"),
        ("Target L", "l2a_target_low_impact_charge"),
        ("Ion Grid", "l2a_ion_grid_impact_charge"),
    )
    data = selected_events.copy()
    data["test_start_time"] = pd.to_datetime(data["test_start_time"], errors="coerce")
    levels = sorted(int(level) for level in data["injection_level"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(levels), 1))
    colors = {level: cmap(idx / max(len(levels) - 1, 1)) for idx, level in enumerate(levels)}

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.0), sharex=True)
    for ax, (label, column) in zip(axes, charge_columns):
        medians = (
            data.groupby(["test_id", "test_start_time", "injection_level"], as_index=False)[column]
            .median()
            .sort_values(["injection_level", "test_start_time"])
        )
        for level in levels:
            level_data = medians[medians["injection_level"] == level]
            if level_data.empty:
                continue
            ax.plot(
                level_data["test_start_time"],
                level_data[column],
                marker="o",
                markersize=4.0,
                linewidth=1.2,
                color=colors[level],
                label=f"Level {level}",
            )
        ax.set_title(label)
        ax.set_ylabel("Charge [pC]")
        ax.grid(True, alpha=0.35)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axes[-1].set_xlabel("Pulser test time")
    setup_date_axis(axes[-1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("L2A Pulser Charge Products by Injection Level", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=min(len(levels), 5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_fit_residuals(fit_metrics: pd.DataFrame, output: Path) -> None:
    data = fit_metrics.copy()
    data["test_start_time"] = pd.to_datetime(data["test_start_time"], errors="coerce")
    medians = (
        data.groupby(["channel", "test_id", "test_start_time", "injection_level"], as_index=False)
        .agg(
            residual_rms=("fit_residual_rms", "median"),
            residual_robust_sigma=("fit_residual_robust_sigma", "median"),
            event_count=("epoch_tt2000", "nunique"),
        )
        .sort_values(["channel", "injection_level", "test_start_time"])
    )
    levels = sorted(int(level) for level in medians["injection_level"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(levels), 1))
    colors = {level: cmap(idx / max(len(levels) - 1, 1)) for idx, level in enumerate(levels)}

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.0), sharex=True)
    for ax, (channel_label, _waveform, _fit) in zip(axes, FIT_CHANNELS):
        channel_data = medians[medians["channel"] == channel_label]
        for level in levels:
            level_data = channel_data[channel_data["injection_level"] == level]
            if level_data.empty:
                continue
            ax.plot(
                level_data["test_start_time"],
                level_data["residual_rms"],
                marker="o",
                markersize=4.0,
                linewidth=1.2,
                color=colors[level],
                label=f"Level {level}",
            )
        ax.set_title(channel_label)
        ax.set_ylabel("Residual RMS [pC]")
        ax.grid(True, alpha=0.35)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axes[-1].set_xlabel("Pulser test time")
    setup_date_axis(axes[-1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("L2A Fit Residuals by Injection Level", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=min(len(levels), 5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_reduced_chi_squared(selected_events: pd.DataFrame, output: Path) -> None:
    chi_columns = (
        ("Target H", "l2a_target_high_reduced_chi_squared"),
        ("Target L", "l2a_target_low_reduced_chi_squared"),
        ("Ion Grid", "l2a_ion_grid_reduced_chi_squared"),
    )
    rows: list[dict[str, object]] = []
    for channel_label, column in chi_columns:
        for event in selected_events.itertuples(index=False):
            event_dict = event._asdict()
            rows.append(
                {
                    "channel": channel_label,
                    "test_id": event_dict["test_id"],
                    "test_start_time": event_dict["test_start_time"],
                    "injection_level": int(event_dict["injection_level"]),
                    "reduced_chi_squared": event_dict.get(column, math.nan),
                    "epoch_tt2000": event_dict["epoch_tt2000"],
                }
            )
    data = pd.DataFrame(rows)
    data["test_start_time"] = pd.to_datetime(data["test_start_time"], errors="coerce")
    medians = (
        data.groupby(["channel", "test_id", "test_start_time", "injection_level"], as_index=False)
        .agg(reduced_chi_squared=("reduced_chi_squared", "median"), event_count=("epoch_tt2000", "nunique"))
        .sort_values(["channel", "injection_level", "test_start_time"])
    )
    levels = sorted(int(level) for level in medians["injection_level"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(levels), 1))
    colors = {level: cmap(idx / max(len(levels) - 1, 1)) for idx, level in enumerate(levels)}

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.0), sharex=True)
    for ax, (channel_label, _column) in zip(axes, chi_columns):
        channel_data = medians[medians["channel"] == channel_label]
        for level in levels:
            if channel_label == "Target H" and level == 5:
                continue
            level_data = channel_data[channel_data["injection_level"] == level]
            if level_data.empty:
                continue
            ax.plot(
                level_data["test_start_time"],
                level_data["reduced_chi_squared"],
                marker="o",
                markersize=4.0,
                linewidth=1.2,
                color=colors[level],
                label=f"Level {level}",
            )
        ax.set_title(channel_label)
        ax.set_ylabel("Reduced χ²")
        ax.grid(True, alpha=0.35)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    axes[-1].set_xlabel("Pulser test time")
    setup_date_axis(axes[-1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("L2A Reduced χ² by Injection Level", y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=min(len(levels), 5), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def grouped_record_indices(events: pd.DataFrame, source_column: str, index_column: str) -> dict[str, list[int]]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for row in events.itertuples(index=False):
        source = getattr(row, source_column)
        if pd.isna(source):
            continue
        by_source[str(source)].append(int(getattr(row, index_column)))
    return by_source


def write_variables_from_sources(
    writer: cdfwrite.CDF,
    *,
    template: cdflib.CDF,
    source_paths: dict[str, Path],
    by_source: dict[str, list[int]],
    variable_names: tuple[str, ...],
    preserve_compression: bool = False,
) -> None:
    for varname in variable_names:
        varinq = template.varinq(varname)
        attrs = template.varattsget(varname)
        rec_vary = varinq["Rec_Vary"] if isinstance(varinq, dict) else varinq.Rec_Vary
        if rec_vary:
            chunks = []
            for source_name in sorted(by_source):
                cdf = cdflib.CDF(str(source_paths[source_name]))
                chunks.append(np.asarray(cdf.varget(varname))[by_source[source_name]])
            data = concat_records(chunks)
        else:
            data = template.varget(varname)
        writer.write_var(var_spec(varinq, preserve_compression=preserve_compression), var_attrs=attrs, var_data=data)


def write_comprehensive_pulser_cdf(
    *,
    output: Path,
    selected_events: pd.DataFrame,
    l1b_dir: Path,
    l2a_dir: Path,
) -> None:
    l1b_paths = {path.name: path for path in l1b_dir.glob("imap_idex_l1b_sci-10days_*.cdf")}
    l2a_paths = {path.name: path for path in l2a_dir.glob("imap_idex_l2a_sci-10days_*.cdf")}
    l1b_by_source = grouped_record_indices(selected_events, "source_file", "record_index")
    l2a_by_source = grouped_record_indices(selected_events, "l2a_source_file", "l2a_record_index")

    missing_l1b = sorted(set(l1b_by_source) - set(l1b_paths))
    missing_l2a = sorted(set(l2a_by_source) - set(l2a_paths))
    if missing_l1b:
        raise FileNotFoundError(f"Missing L1B source CDF files: {', '.join(missing_l1b)}")
    if missing_l2a:
        raise FileNotFoundError(f"Missing L2A source CDF files: {', '.join(missing_l2a)}")

    l1b_template = cdflib.CDF(str(l1b_paths[sorted(l1b_by_source)[0]]))
    l2a_template = cdflib.CDF(str(l2a_paths[sorted(l2a_by_source)[0]]))
    info = l1b_template.cdf_info()
    cdf_spec = {
        "Majority": cdf_info_value(info, "Majority"),
        "Encoding": cdf_info_value(info, "Encoding"),
        "Checksum": cdf_info_value(info, "Checksum"),
        "Compressed": cdf_info_value(info, "Compressed"),
        "rDim_sizes": list(cdf_info_value(info, "rDim_sizes")),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    used_l1b_sources = [l1b_paths[name] for name in sorted(l1b_by_source)]
    writer = cdfwrite.CDF(str(output), cdf_spec=cdf_spec, delete=True)
    try:
        attrs = global_attrs_for_output(l1b_template, output, used_l1b_sources, selected_events)
        attrs["Pulser_l2a_fit_variables"] = {
            index: value for index, value in enumerate(PULSER_L2A_CDF_VARIABLES)
        }
        writer.write_globalattrs(attrs)
        write_variables_from_sources(
            writer,
            template=l1b_template,
            source_paths=l1b_paths,
            by_source=l1b_by_source,
            variable_names=PULSER_CDF_VARIABLES,
            preserve_compression=False,
        )
        write_variables_from_sources(
            writer,
            template=l2a_template,
            source_paths=l2a_paths,
            by_source=l2a_by_source,
            variable_names=PULSER_L2A_CDF_VARIABLES,
            preserve_compression=False,
        )
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1b-dir", type=Path, default=DEFAULT_L1B_DIR)
    parser.add_argument("--l2a-dir", type=Path, default=DEFAULT_L2A_DIR)
    parser.add_argument("--event-list", type=Path, default=DEFAULT_EVENT_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-list-tolerance-seconds", type=float, default=2.0)
    parser.add_argument("--test-gap-minutes", type=float, default=30.0)
    parser.add_argument("--baseline-samples", type=int, default=200)
    parser.add_argument(
        "--exclude-epoch-utc",
        action="append",
        default=[],
        help="Exact UTC epoch string to remove from the pulser trend products. May be supplied multiple times.",
    )
    parser.add_argument(
        "--min-quicklook-only-ion-charge",
        type=float,
        default=0.05,
        help="Keep quicklook-only pulser records only if L2A ion-grid charge is at least this value. "
        "Use a negative value to disable this filter.",
    )
    parser.add_argument(
        "--quicklook-only",
        action="store_true",
        help="Ignore Event List pulser labels and use only the quicklook-style L1B classifier.",
    )
    args = parser.parse_args()

    if args.baseline_samples <= 0:
        raise SystemExit("--baseline-samples must be positive")
    if args.test_gap_minutes <= 0:
        raise SystemExit("--test-gap-minutes must be positive")
    exclude_epoch_utc = set(args.exclude_epoch_utc)

    apply_style()
    l1b_sources = cdf_sources(args.l1b_dir, "l1b")
    l2a_sources = cdf_sources(args.l2a_dir, "l2a")
    l1b_events = load_l1b_event_index(l1b_sources)
    l2a_table = load_l2a_table(l2a_sources)

    quicklook_epochs = set(
        int(epoch)
        for epoch in l1b_events.loc[l1b_events["quicklook_classification"].eq("Pulser"), "epoch_tt2000"].to_numpy()
    )
    event_list_epochs: set[int] = set()
    if not args.quicklook_only:
        event_list_epochs = load_event_list_pulser_epochs(
            args.event_list,
            l1b_events,
            args.event_list_tolerance_seconds,
        )
    selected_epochs = quicklook_epochs | event_list_epochs
    if not selected_epochs:
        raise SystemExit("No pulser events found.")

    selected_events = l1b_events[l1b_events["epoch_tt2000"].isin(selected_epochs)].copy()
    selected_events["selection_source"] = selected_events["epoch_tt2000"].map(
        lambda epoch: "+".join(
            source
            for source, source_epochs in (
                ("quicklook", quicklook_epochs),
                ("event_list", event_list_epochs),
            )
            if int(epoch) in source_epochs
        )
    )
    selected_events = selected_events.merge(l2a_table, on="epoch_tt2000", how="left")
    quicklook_only = selected_events["selection_source"].eq("quicklook")
    if args.min_quicklook_only_ion_charge >= 0.0:
        quicklook_only_low_charge = quicklook_only & (
            selected_events["l2a_ion_grid_impact_charge"].isna()
            | (selected_events["l2a_ion_grid_impact_charge"] < args.min_quicklook_only_ion_charge)
        )
        filtered_quicklook_only_count = int(quicklook_only_low_charge.sum())
        selected_events = selected_events.loc[~quicklook_only_low_charge].copy()
    else:
        filtered_quicklook_only_count = 0
    excluded_epoch_count = 0
    if exclude_epoch_utc:
        excluded_epoch_count = int(selected_events["epoch_utc"].isin(exclude_epoch_utc).sum())
        selected_events = selected_events.loc[~selected_events["epoch_utc"].isin(exclude_epoch_utc)].copy()
    if selected_events.empty:
        raise SystemExit("No pulser events remain after filtering.")
    selected_events = add_test_and_level_columns(selected_events, args.test_gap_minutes)
    metrics = collect_channel_metrics(selected_events, l1b_sources, args.baseline_samples)
    fit_metrics = collect_fit_residual_metrics(selected_events, l1b_sources, l2a_sources)
    test_summary = summarize_tests(selected_events)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "pulser_event_summary.csv"
    metrics_path = args.output_dir / "pulser_channel_response_metrics.csv"
    fit_metrics_path = args.output_dir / "pulser_fit_residual_metrics.csv"
    tests_path = args.output_dir / "pulser_test_summary.csv"
    response_path = args.output_dir / "pulser_l1b_response_by_injection_level_time.png"
    normalized_path = args.output_dir / "pulser_l1b_response_normalized_by_injection_level_time.png"
    l2a_path = args.output_dir / "pulser_l2a_charge_by_injection_level_time.png"
    residual_path = args.output_dir / "pulser_l2a_fit_residuals_by_injection_level_time.png"
    reduced_chi_path = args.output_dir / "pulser_l2a_reduced_chi_squared_by_injection_level_time.png"
    pulser_cdf_path = args.output_dir / "imap_idex_l1b_pulser_events_v001.cdf"

    selected_events.drop(columns=["event_time", "test_start_time"]).to_csv(events_path, index=False)
    metrics.drop(columns=["event_time", "test_start_time"]).to_csv(metrics_path, index=False)
    fit_metrics.drop(columns=["event_time", "test_start_time"]).to_csv(fit_metrics_path, index=False)
    test_summary.to_csv(tests_path, index=False)
    plot_channel_response(metrics, response_path, "response_peak_amplitude")
    plot_normalized_channel_response(metrics, normalized_path, "response_peak_amplitude")
    plot_l2a_charges(selected_events, l2a_path)
    plot_fit_residuals(fit_metrics, residual_path)
    plot_reduced_chi_squared(selected_events, reduced_chi_path)
    cdf_events = selected_events[["epoch_tt2000", "epoch_utc", "source_file", "record_index"]].copy()
    write_comprehensive_pulser_cdf(
        output=pulser_cdf_path,
        selected_events=selected_events,
        l1b_dir=args.l1b_dir,
        l2a_dir=args.l2a_dir,
    )
    verify_subset(pulser_cdf_path, cdf_events)

    print(f"Selected {selected_events['epoch_tt2000'].nunique()} pulser events across {test_summary.shape[0]} tests.")
    print(f"Quicklook pulser epochs: {len(quicklook_epochs)}")
    if not args.quicklook_only:
        print(f"Event List pulser epochs matched to current L1B CDFs: {len(event_list_epochs)}")
    if filtered_quicklook_only_count:
        print(
            "Filtered "
            f"{filtered_quicklook_only_count} quicklook-only records below "
            f"{args.min_quicklook_only_ion_charge:g} pC L2A ion-grid charge."
        )
    if excluded_epoch_count:
        print(f"Excluded {excluded_epoch_count} explicitly requested pulser epoch(s).")
    print(f"Wrote {events_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {fit_metrics_path}")
    print(f"Wrote {tests_path}")
    print(f"Wrote {response_path}")
    print(f"Wrote {normalized_path}")
    print(f"Wrote {l2a_path}")
    print(f"Wrote {residual_path}")
    print(f"Wrote {reduced_chi_path}")
    print(f"Wrote {pulser_cdf_path}")


if __name__ == "__main__":
    main()
