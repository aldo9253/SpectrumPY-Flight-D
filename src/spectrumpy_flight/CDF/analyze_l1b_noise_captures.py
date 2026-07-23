#!/usr/bin/env python3
"""Analyze L1A/L1B noise-capture waveform baselines and noise widths.

The plotted ``robust_sigma`` is a Gaussian-equivalent noise width estimated as
``1.4826 * median(abs(samples - median(samples)))``. This median absolute
deviation estimator is less sensitive than ordinary standard deviation to
isolated spikes, non-Gaussian tails, or signal leakage in nominal noise captures.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cdflib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "l1b"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "noise_capture_analysis"
DEFAULT_PEAKS = SCRIPT_DIR / "l1b_peak_analysis" / "l1b_tof_peak_peaks_default.csv"
FILL_LIMIT = -1.0e30
PULSER_HG_TRIGGER_LEVEL = 0.289
PULSER_HG_TRIGGER_LEVEL_TOLERANCE = 1.0e-6
RAW_TRIGGER_LEVEL_SCALES = {
    "TOF H": 2.89e-4,
    "TOF M": 1.13e-2,
    "TOF L": 5.14e-4,
}

CHANNELS: tuple[tuple[str, str], ...] = (
    ("TOF H", "TOF_High"),
    ("TOF M", "TOF_Mid"),
    ("TOF L", "TOF_Low"),
    ("Target H", "Target_High"),
    ("Target L", "Target_Low"),
    ("Ion Grid", "Ion_Grid"),
)

TIME_SERIES_CHANNELS: tuple[tuple[str, str], ...] = (
    ("TOF H", "TOF_High"),
    ("Target H", "Target_High"),
    ("TOF M", "TOF_Mid"),
    ("Target L", "Target_Low"),
    ("TOF L", "TOF_Low"),
    ("Ion Grid", "Ion_Grid"),
)

CHANNEL_COLORS = {
    "TOF H": "#0c5da5",
    "TOF M": "#ff8100",
    "TOF L": "#8fb339",
    "Target H": "#7a68a6",
    "Target L": "#c62828",
    "Ion Grid": "#0096c7",
}


def apply_style() -> None:
    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    try:
        from plot_style import apply_plot_style
    except Exception:
        plt.style.use("default")
    else:
        apply_plot_style("light")


def cdf_sources(input_dir: Path) -> list[Path]:
    sources = sorted(input_dir.glob("imap_idex_l1*_sci-10days_*.cdf"))
    if not sources:
        raise FileNotFoundError(f"No L1A/L1B 10-day CDF files found in {input_dir}")
    return sources


def infer_unit_label(input_dir: Path, explicit_unit: str | None) -> str:
    if explicit_unit:
        return explicit_unit
    return "DN" if input_dir.name.lower() == "l1a" else "pC"


def infer_default_prefix(input_dir: Path, explicit_prefix: str | None) -> str:
    if explicit_prefix:
        return explicit_prefix
    level = input_dir.name.lower()
    if level in {"l1a", "l1b"}:
        return f"{level}_noise_capture"
    return "l1b_noise_capture"


def infer_product_label(input_dir: Path) -> str:
    level = input_dir.name.upper()
    return level if level in {"L1A", "L1B"} else "L1B"


def clean_waveform(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel().copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= FILL_LIMIT] = np.nan
    return arr


def waveform_stats(values: np.ndarray, *, baseline_method: str) -> tuple[float, float, float, float, int]:
    arr = clean_waveform(values)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan, math.nan, math.nan, math.nan, 0
    if baseline_method == "mean":
        baseline = float(np.nanmean(finite))
    else:
        baseline = float(np.nanmedian(finite))
    deviations = finite - baseline
    mad = float(np.nanmedian(np.abs(deviations)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(finite))
    return baseline, sigma, float(np.nanmean(finite)), float(np.nanstd(finite)), int(finite.size)


def residual_sample(values: np.ndarray, baseline: float, max_samples: int) -> np.ndarray:
    arr = clean_waveform(values)
    residuals = arr[np.isfinite(arr)] - float(baseline)
    if residuals.size <= max_samples:
        return residuals
    indices = np.linspace(0, residuals.size - 1, max_samples, dtype=int)
    return residuals[indices]


def text_value(values: np.ndarray, index: int) -> str:
    try:
        value = np.asarray(values[index]).reshape(-1)[0]
    except Exception:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def numeric_value(values: np.ndarray, index: int) -> float | None:
    try:
        value = float(np.asarray(values[index]).reshape(-1)[0])
    except Exception:
        return None
    if not np.isfinite(value) or value <= FILL_LIMIT:
        return None
    return value


def trigger_mode_is_active(trigger_mode: str | None) -> bool:
    if not trigger_mode:
        return False
    token = re.sub(r"[^a-z0-9]", "", str(trigger_mode).strip().lower())
    if not token:
        return False
    return token not in {"0", "dis", "disabled", "off", "none", "nan"}


def trigger_channels_from_origins(origins: list[str]) -> list[str]:
    origin_map = {
        "HS ADC0I trigger": ["TOF H"],
        "HS ADC0I trigger (TOF HG)": ["TOF H"],
        "HS ADC0Q trigger": ["TOF L"],
        "HS ADC0Q trigger (TOF LG)": ["TOF L"],
        "HS ADC1Q trigger": ["TOF M"],
        "HS ADC1Q trigger (TOF MG)": ["TOF M"],
        "LS ADC1 trigger": ["Target H"],
        "LS ADC1 trigger (Target HG / low range)": ["Target H"],
    }
    channels: list[str] = []
    for origin in origins:
        channels.extend(origin_map.get(origin.strip(), []))
    return channels


def decode_trigger_origins(trigger_id: int) -> list[str]:
    labels: list[str] = []
    u10 = int(trigger_id) & 0x3FF
    if (u10 >> 0) & 1:
        labels.append("HS ADC0I trigger (TOF HG)")
    if (u10 >> 1) & 1:
        labels.append("HS ADC0Q trigger (TOF LG)")
    if (u10 >> 2) & 1:
        labels.append("HS ADC1Q trigger (TOF MG)")
    if (u10 >> 3) & 1:
        labels.append("LS ADC1 trigger (Target HG / low range)")
    if (u10 >> 4) & 1:
        labels.append("SW trigger")
    if (u10 >> 5) & 1:
        labels.append("external trigger")
    return labels


def raw_trigger_level(raw_value: float | None, channel: str) -> float | None:
    if raw_value is None:
        return None
    scale = RAW_TRIGGER_LEVEL_SCALES.get(channel)
    if scale is None:
        return None
    try:
        trigger_counts = (int(raw_value) >> 22) & 0x3FF
    except Exception:
        return None
    return float(scale * trigger_counts)


def raw_trigger_mode(raw_value: object, channel: str) -> str:
    text = str(raw_value).strip()
    if text and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text):
        return text
    try:
        mode_value = int(float(np.asarray(raw_value).reshape(-1)[0]))
    except Exception:
        return ""
    if mode_value <= 0:
        return ""
    channel_prefix = {"TOF H": "HG", "TOF M": "MG", "TOF L": "LG"}.get(channel, "TRIG")
    mode_label = {1: "Threshold", 2: "SinglePulse"}.get(mode_value, "DoublePulse")
    return f"{channel_prefix}{mode_label}"


def classify_event(cdf_vars: dict[str, np.ndarray], index: int) -> str:
    origins: list[str] = []
    if "trigger_origin" in cdf_vars:
        origin_text = text_value(cdf_vars["trigger_origin"], index)
        origins = [item.strip() for item in origin_text.split(",") if item.strip()]
    elif "idx__txhdrtrigid" in cdf_vars:
        trigger_id = numeric_value(cdf_vars["idx__txhdrtrigid"], index)
        if trigger_id is not None:
            origins = decode_trigger_origins(int(trigger_id))
    origin_tokens = [origin.lower() for origin in origins]

    configured_channels: list[str] = []
    configured_channels.extend(trigger_channels_from_origins(origins))

    if "trigger_mode_hg" in cdf_vars:
        mode_by_channel = {
            "TOF H": text_value(cdf_vars["trigger_mode_hg"], index),
            "TOF M": text_value(cdf_vars["trigger_mode_mg"], index),
            "TOF L": text_value(cdf_vars["trigger_mode_lg"], index),
        }
        level_by_channel = {
            "TOF H": numeric_value(cdf_vars["trigger_level_hg"], index),
            "TOF M": numeric_value(cdf_vars["trigger_level_mg"], index),
            "TOF L": numeric_value(cdf_vars["trigger_level_lg"], index),
        }
    else:
        raw_mode_fields = {
            "TOF H": "idx__txhdrhgtrigmode",
            "TOF M": "idx__txhdrmgtrigmode",
            "TOF L": "idx__txhdrlgtrigmode",
        }
        raw_level_fields = {
            "TOF H": "idx__txhdrhgtrigctrl1",
            "TOF M": "idx__txhdrmgtrigctrl1",
            "TOF L": "idx__txhdrlgtrigctrl1",
        }
        mode_by_channel = {
            channel: raw_trigger_mode(cdf_vars[field][index], channel)
            for channel, field in raw_mode_fields.items()
            if field in cdf_vars
        }
        level_by_channel = {
            channel: raw_trigger_level(numeric_value(cdf_vars[field], index), channel)
            for channel, field in raw_level_fields.items()
            if field in cdf_vars
        }
    for channel, level in level_by_channel.items():
        if level is None:
            continue
        if trigger_mode_is_active(mode_by_channel.get(channel, "")):
            configured_channels.append(channel)

    configured_set = set(configured_channels)
    has_software_trigger = any("sw trigger" in token or "external trigger" in token for token in origin_tokens)

    if not configured_channels:
        return "Noise"
    if has_software_trigger and configured_set <= {"TOF H"}:
        return "Noise"
    if (
        configured_set == {"TOF H"}
        and mode_by_channel.get("TOF H", "").strip() in {"HGThreshold", "HGThreshold"}
        and level_by_channel["TOF H"] is not None
        and abs(float(level_by_channel["TOF H"]) - PULSER_HG_TRIGGER_LEVEL) <= PULSER_HG_TRIGGER_LEVEL_TOLERANCE
    ):
        return "Pulser"
    if len(configured_set) >= 2 or any(channel != "TOF H" for channel in configured_set):
        return "Science"
    return "Science"


def epoch_to_datetime(epoch_tt2000: int) -> pd.Timestamp:
    encoded = cdflib.cdfepoch.encode_tt2000(int(epoch_tt2000))
    if isinstance(encoded, list):
        encoded = encoded[0]
    return pd.to_datetime(str(encoded))


def select_evenly(rows: pd.DataFrame, max_events: int | None) -> pd.DataFrame:
    if max_events is None or max_events <= 0 or len(rows) <= max_events:
        return rows
    indices = np.linspace(0, len(rows) - 1, max_events, dtype=int)
    return rows.iloc[indices].reset_index(drop=True)


def load_peak_event_epochs(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Peak table not found: {path}")
    peaks = pd.read_csv(path, usecols=["epoch_tt2000"])
    return set(int(epoch) for epoch in peaks["epoch_tt2000"].dropna().astype(np.int64).unique())


def collect_noise_metrics(
    sources: list[Path],
    *,
    max_noise_events: int | None,
    max_residual_samples_per_event: int,
    baseline_method: str,
    exclude_peak_epochs: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[np.ndarray]], pd.DataFrame]:
    event_rows: list[dict[str, object]] = []
    source_paths = {source.name: source for source in sources}
    classification_counts: dict[str, int] = {"Noise": 0, "Pulser": 0, "Science": 0}

    trigger_candidates = (
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

    for source in sources:
        cdf = cdflib.CDF(str(source))
        cdf_vars = {"epoch": np.asarray(cdf.varget("epoch"), dtype=np.int64)}
        for varname in trigger_candidates:
            try:
                cdf_vars[varname] = np.asarray(cdf.varget(varname))
            except Exception:
                continue
        for record_index, epoch in enumerate(cdf_vars["epoch"]):
            classification = classify_event(cdf_vars, record_index)
            classification_counts[classification] = classification_counts.get(classification, 0) + 1
            if classification != "Noise":
                continue
            if exclude_peak_epochs is not None and int(epoch) in exclude_peak_epochs:
                continue
            event_rows.append(
                {
                    "source_file": source.name,
                    "record_index": int(record_index),
                    "epoch_tt2000": int(epoch),
                    "epoch_utc": str(cdflib.cdfepoch.encode_tt2000(int(epoch))),
                }
            )

    events = pd.DataFrame(event_rows).sort_values("epoch_tt2000").reset_index(drop=True)
    selected_events = select_evenly(events, max_noise_events)
    metrics_rows: list[dict[str, object]] = []
    residuals_by_channel: dict[str, list[np.ndarray]] = {label: [] for label, _var in CHANNELS}

    for source_name, source_events in selected_events.groupby("source_file", sort=True):
        source = source_paths[str(source_name)]
        cdf = cdflib.CDF(str(source))
        source_waveforms = {var_name: np.asarray(cdf.varget(var_name)) for _label, var_name in CHANNELS}
        for event in source_events.itertuples(index=False):
            for channel_label, var_name in CHANNELS:
                waveform = source_waveforms[var_name][int(event.record_index)]
                baseline, sigma, mean, std, sample_count = waveform_stats(waveform, baseline_method=baseline_method)
                metrics_rows.append(
                    {
                        "source_file": event.source_file,
                        "record_index": int(event.record_index),
                        "epoch_tt2000": int(event.epoch_tt2000),
                        "epoch_utc": event.epoch_utc,
                        "channel": channel_label,
                        "cdf_variable": var_name,
                        "baseline_method": baseline_method,
                        "baseline": baseline,
                        "robust_sigma": sigma,
                        "mean": mean,
                        "std": std,
                        "sample_count": sample_count,
                    }
                )
                if np.isfinite(baseline):
                    residuals_by_channel[channel_label].append(
                        residual_sample(waveform, baseline, max_residual_samples_per_event)
                    )

    count_rows = [{"classification": key, "event_count": value} for key, value in sorted(classification_counts.items())]
    return pd.DataFrame(metrics_rows), residuals_by_channel, pd.DataFrame(count_rows)


def plot_context(metrics: pd.DataFrame) -> str:
    events = metrics[["epoch_utc", "epoch_tt2000"]].drop_duplicates().copy()
    event_count = len(events)
    times = pd.to_datetime(events["epoch_utc"], errors="coerce").dropna()
    if times.empty:
        return f"{event_count} noise captures"
    start = times.min().strftime("%Y-%m-%d %H:%M:%S")
    end = times.max().strftime("%Y-%m-%d %H:%M:%S")
    return f"{event_count} noise captures, {start} to {end} UTC"


def plot_residual_histograms(
    metrics: pd.DataFrame,
    residuals_by_channel: dict[str, list[np.ndarray]],
    output: Path,
    *,
    context: str,
    sigma_column: str,
    sigma_label: str,
    unit_label: str,
    product_label: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    for ax, (channel_label, _var_name) in zip(axes.ravel(), CHANNELS):
        chunks = residuals_by_channel[channel_label]
        residuals = np.concatenate(chunks) if chunks else np.asarray([], dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        channel_metrics = metrics[metrics["channel"] == channel_label]
        sigma = float(np.nanmedian(channel_metrics[sigma_column])) if not channel_metrics.empty else math.nan
        if residuals.size:
            lo, hi = np.nanpercentile(residuals, [0.5, 99.5])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                plot_values = residuals[(residuals >= lo) & (residuals <= hi)]
            else:
                plot_values = residuals
            counts, bin_edges, _patches = ax.hist(
                plot_values,
                bins=90,
                density=True,
                color=CHANNEL_COLORS[channel_label],
                alpha=0.72,
            )
            if np.isfinite(sigma) and sigma > 0.0:
                xs = np.linspace(float(np.nanmin(plot_values)), float(np.nanmax(plot_values)), 400)
                gaussian = np.exp(-0.5 * (xs / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
                max_count = float(np.nanmax(counts)) if counts.size else math.nan
                max_gaussian = float(np.nanmax(gaussian)) if gaussian.size else math.nan
                if np.isfinite(max_count) and np.isfinite(max_gaussian) and max_count > 0.0 and max_gaussian > 0.0:
                    gaussian = gaussian * (max_count / max_gaussian)
                ax.plot(xs, gaussian, color="#121212", linewidth=1.6, label=f"Gaussian {sigma_label}={sigma:.3e}")
                ax.legend(loc="upper right")
        ax.set_title(f"{channel_label} residuals")
        ax.set_xlabel(f"Sample - Baseline [{unit_label}]")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.35)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    fig.suptitle(f"{product_label} Noise Capture Residual Distributions ({sigma_label})\n{context}")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def use_scientific_yaxis(ax) -> None:
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _position: f"{value:.2e}"))


def plot_time_series(
    metrics: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output: Path,
    *,
    context: str,
) -> None:
    data = metrics.copy()
    data["event_time"] = pd.to_datetime(data["epoch_utc"], errors="coerce")
    fig, axes = plt.subplots(3, 2, figsize=(14.0, 10.0), sharex=True)
    for ax, (channel_label, _var_name) in zip(axes.ravel(), TIME_SERIES_CHANNELS):
        channel_data = data[data["channel"] == channel_label].sort_values("event_time")
        ax.scatter(
            channel_data["event_time"],
            channel_data[value_column],
            s=12,
            alpha=0.72,
            color=CHANNEL_COLORS[channel_label],
        )
        ax.set_title(channel_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.35)
        if value_column == "robust_sigma":
            use_scientific_yaxis(ax)
        else:
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
    for ax in axes[-1, :]:
        ax.set_xlabel("Event time")
    axes[-1, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1, 0].xaxis.get_major_locator()))
    fig.suptitle(f"{title}\n{context}")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--exclude-events-with-peaks", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--unit-label", default=None)
    parser.add_argument(
        "--sigma-method",
        choices=("robust", "std"),
        default="robust",
        help="Noise-width metric to use for Gaussian overlays and sigma-vs-time plot.",
    )
    parser.add_argument(
        "--baseline-method",
        choices=("median", "mean"),
        default="median",
        help="Baseline estimator for each noise-capture waveform.",
    )
    parser.add_argument("--max-noise-events", type=int, default=None, help="Evenly sample at most this many noise events.")
    parser.add_argument("--max-residual-samples-per-event", type=int, default=2000)
    args = parser.parse_args()

    if args.max_residual_samples_per_event <= 0:
        raise SystemExit("--max-residual-samples-per-event must be positive")

    apply_style()
    sources = cdf_sources(args.input_dir)
    unit_label = infer_unit_label(args.input_dir, args.unit_label)
    product_label = infer_product_label(args.input_dir)
    exclude_peak_epochs = load_peak_event_epochs(args.peaks) if args.exclude_events_with_peaks else None
    metrics, residuals_by_channel, classification_counts = collect_noise_metrics(
        sources,
        max_noise_events=args.max_noise_events,
        max_residual_samples_per_event=args.max_residual_samples_per_event,
        baseline_method=args.baseline_method,
        exclude_peak_epochs=exclude_peak_epochs,
    )
    if metrics.empty:
        raise SystemExit("No noise-capture events found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = infer_default_prefix(args.input_dir, args.output_prefix)
    metrics_path = args.output_dir / f"{prefix}_channel_metrics.csv"
    counts_path = args.output_dir / f"{prefix}_classification_counts.csv"
    metrics.to_csv(metrics_path, index=False)
    classification_counts.to_csv(counts_path, index=False)
    context = plot_context(metrics)
    sigma_column = "std" if args.sigma_method == "std" else "robust_sigma"
    sigma_label = "std" if args.sigma_method == "std" else "robust sigma"
    baseline_label = "Mean baseline" if args.baseline_method == "mean" else "Median baseline"

    residual_path = args.output_dir / f"{prefix}_residual_histograms.png"
    baseline_path = args.output_dir / f"{prefix}_baseline_time.png"
    sigma_path = args.output_dir / f"{prefix}_sigma_time.png"
    plot_residual_histograms(
        metrics,
        residuals_by_channel,
        residual_path,
        context=context,
        sigma_column=sigma_column,
        sigma_label=sigma_label,
        unit_label=unit_label,
        product_label=product_label,
    )
    plot_time_series(
        metrics,
        "baseline",
        baseline_label,
        f"{product_label} Noise Capture {baseline_label} vs Time",
        baseline_path,
        context=context,
    )
    plot_time_series(
        metrics,
        sigma_column,
        "Standard deviation" if args.sigma_method == "std" else "Robust sigma",
        f"{product_label} Noise Capture Standard Deviation vs Time"
        if args.sigma_method == "std"
        else f"{product_label} Noise Capture Sigma vs Time",
        sigma_path,
        context=context,
    )

    noise_events = metrics[["source_file", "record_index", "epoch_tt2000"]].drop_duplicates().shape[0]
    print(f"Analyzed {noise_events} noise-capture events across {len(sources)} CDF files.")
    if exclude_peak_epochs is not None:
        print(f"Excluded noise captures whose epoch appears in {args.peaks}.")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {counts_path}")
    print(f"Wrote {residual_path}")
    print(f"Wrote {baseline_path}")
    print(f"Wrote {sigma_path}")


if __name__ == "__main__":
    main()
