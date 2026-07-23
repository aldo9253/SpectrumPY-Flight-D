#!/usr/bin/env python3
"""Plot TOF peak-width histograms from the L1B TOF peak analysis product."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = SCRIPT_DIR / "l1b_peak_analysis" / "l1b_tof_peak_peaks_default.csv"
DEFAULT_EVENT_LIST = SCRIPT_DIR / "l1b_peak_analysis" / "IDEX_Event_List.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "l1b_peak_analysis" / "l1b_tof_peak_width_histograms_default.png"
CHANNEL_ORDER = ("tof_high", "tof_mid", "tof_low")
CHANNEL_TITLES = {
    "tof_high": "TOF High",
    "tof_mid": "TOF Mid",
    "tof_low": "TOF Low",
}
CHANNEL_COLORS = {
    "tof_high": "#0c5da5",
    "tof_mid": "#ff8100",
    "tof_low": "#8fb339",
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


def load_peak_widths(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Peak table not found: {path}")
    peaks = pd.read_csv(path)
    required = {"channel", "width_us"}
    missing = required.difference(peaks.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    peaks = peaks.copy()
    peaks["width_us"] = pd.to_numeric(peaks["width_us"], errors="coerce")
    peaks = peaks[np.isfinite(peaks["width_us"]) & (peaks["width_us"] > 0.0)]
    if peaks.empty:
        raise ValueError(f"No positive finite peak widths found in {path}")
    peaks["event_time"] = pd.to_datetime(peaks["epoch_utc"], errors="coerce")
    return peaks


def load_triggered_dust_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event list not found: {path}")
    events = pd.read_csv(path, header=1)
    events.columns = [str(column).strip() for column in events.columns]
    required = {"Time", "Event type", "Dust category"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    for column in required:
        events[column] = events[column].astype(str).str.strip()
    triggered_dust = events[
        events["Event type"].str.casefold().eq("triggered")
        & events["Dust category"].str.casefold().eq("dust")
    ].copy()
    triggered_dust["event_time"] = pd.to_datetime(
        triggered_dust["Time"],
        format="%d-%b-%Y %H:%M:%S",
        errors="coerce",
    )
    triggered_dust = triggered_dust.dropna(subset=["event_time"])
    if triggered_dust.empty:
        raise ValueError(f"No Triggered/Dust events with parseable times found in {path}")
    return triggered_dust


def triggered_dust_peak_widths(
    peaks: pd.DataFrame,
    events: pd.DataFrame,
    *,
    tolerance_seconds: float,
) -> tuple[pd.DataFrame, int]:
    event_times = peaks[["epoch_utc", "event_time"]].drop_duplicates().dropna(subset=["event_time"])
    matched_epoch_utcs: set[str] = set()
    tolerance = pd.Timedelta(seconds=float(tolerance_seconds))

    for event_time in events["event_time"]:
        deltas = (event_times["event_time"] - event_time).abs()
        if deltas.empty:
            continue
        nearest_index = deltas.idxmin()
        if deltas.loc[nearest_index] <= tolerance:
            matched_epoch_utcs.add(str(event_times.loc[nearest_index, "epoch_utc"]))

    overlay = peaks[peaks["epoch_utc"].astype(str).isin(matched_epoch_utcs)].copy()
    return overlay, len(matched_epoch_utcs)


def widest_peak_per_event(peaks: pd.DataFrame, *, per_channel: bool = False) -> pd.DataFrame:
    required = {"epoch_tt2000", "width_us", "channel"}
    missing = required.difference(peaks.columns)
    if missing:
        raise ValueError(f"Peak table is missing required columns: {', '.join(sorted(missing))}")
    group_columns = ["epoch_tt2000", "channel"] if per_channel else ["epoch_tt2000"]
    idx = peaks.groupby(group_columns)["width_us"].idxmax()
    return peaks.loc[idx].sort_values("epoch_tt2000").reset_index(drop=True)


def width_bins(widths: pd.Series, bins: int, max_width_us: float | None) -> np.ndarray:
    values = widths.to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if max_width_us is not None:
        values = values[values <= max_width_us]
        if values.size:
            upper = float(max_width_us)
        else:
            upper = float(max_width_us)
    else:
        upper = float(np.nanpercentile(values, 99.0))
    if not np.isfinite(upper) or upper <= 0.0:
        upper = 1.0
    return np.linspace(0.0, upper, int(bins) + 1)


def log_width_bins(widths: pd.Series, bins: int, max_width_us: float | None) -> np.ndarray:
    values = widths.to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if max_width_us is not None:
        values = values[values <= max_width_us]
        upper = float(max_width_us)
    else:
        upper = float(np.nanpercentile(values, 99.0))
    if values.size == 0:
        raise ValueError("No positive finite peak widths are available for log-scaled bins")
    lower = float(np.nanmin(values))
    if not np.isfinite(lower) or lower <= 0.0:
        lower = 1.0e-6
    if not np.isfinite(upper) or upper <= lower:
        upper = float(np.nanmax(values))
    if not np.isfinite(upper) or upper <= lower:
        upper = lower * 10.0
    return np.logspace(np.log10(lower), np.log10(upper), int(bins) + 1)


def plot_histograms(
    peaks: pd.DataFrame,
    output: Path,
    *,
    bins: int,
    max_width_us: float | None,
    log_y: bool,
    log_x: bool,
    overlay_peaks: pd.DataFrame | None,
    overlay_label: str,
    overlay_matched_events: int,
    event_level: bool,
) -> None:
    plot_peaks = peaks
    plot_overlay = overlay_peaks
    if max_width_us is not None:
        plot_peaks = peaks[peaks["width_us"] <= max_width_us]
        if plot_overlay is not None:
            plot_overlay = plot_overlay[plot_overlay["width_us"] <= max_width_us]
        if plot_peaks.empty:
            raise ValueError(f"No peak widths are <= --max-width-us {max_width_us}")

    if log_x:
        edges = log_width_bins(plot_peaks["width_us"], bins, max_width_us)
    else:
        edges = width_bins(plot_peaks["width_us"], bins, max_width_us)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)

    for ax, channel in zip(axes, CHANNEL_ORDER):
        channel_widths = plot_peaks.loc[plot_peaks["channel"] == channel, "width_us"].to_numpy(dtype=float)
        ax.hist(
            channel_widths,
            bins=edges,
            color=CHANNEL_COLORS[channel],
            alpha=0.82,
            label="All events with peaks" if event_level else "All detected peaks",
            edgecolor="white",
            linewidth=0.35,
        )
        title = f"{CHANNEL_TITLES[channel]} peak widths (n={channel_widths.size})"
        if plot_overlay is not None:
            overlay_widths = plot_overlay.loc[plot_overlay["channel"] == channel, "width_us"].to_numpy(dtype=float)
            ax.hist(
                overlay_widths,
                bins=edges,
                histtype="step",
                color="#c62828",
                linewidth=2.0,
                label=f"{overlay_label} events with peaks" if event_level else f"{overlay_label} peaks",
            )
            title += f"; overlay n={overlay_widths.size}"
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.45)
        ax.legend(loc="upper right")
        if log_y and (channel_widths.size > 0 or (plot_overlay is not None and overlay_widths.size > 0)):
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")

    axes[-1].set_xlabel("Peak width (microseconds)")
    title = "L1B TOF Widest Positive Peak Per Event" if event_level else "L1B TOF Positive Peak Widths"
    if plot_overlay is not None:
        title += f" with Triggered/Dust Overlay ({overlay_matched_events} events)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Peak-level CSV from analyze_l1b_tof_peaks.py.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-list", type=Path, default=DEFAULT_EVENT_LIST)
    parser.add_argument("--overlay-triggered-dust", action="store_true")
    parser.add_argument("--event-match-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max-width-us", type=float, default=None, help="Optional upper width cutoff for display.")
    parser.add_argument("--linear-y", action="store_true", help="Use a linear count axis instead of log scale.")
    parser.add_argument("--log-x", action="store_true", help="Use logarithmic peak-width bins and x-axis.")
    parser.add_argument(
        "--widest-peak-per-event",
        action="store_true",
        help="Collapse each event to its single widest TOF peak before plotting.",
    )
    parser.add_argument(
        "--widest-peak-per-event-channel",
        action="store_true",
        help="Collapse each event/channel to its widest peak before plotting.",
    )
    args = parser.parse_args()

    if args.bins <= 0:
        raise SystemExit("--bins must be positive")

    apply_style()
    peaks = load_peak_widths(args.input)
    overlay_peaks = None
    overlay_matched_events = 0
    if args.overlay_triggered_dust:
        triggered_dust = load_triggered_dust_events(args.event_list)
        overlay_peaks, overlay_matched_events = triggered_dust_peak_widths(
            peaks,
            triggered_dust,
            tolerance_seconds=args.event_match_tolerance_seconds,
        )
        print(
            f"Matched {overlay_matched_events} / {len(triggered_dust)} Triggered/Dust events "
            f"within {args.event_match_tolerance_seconds:g} second(s)."
        )
    event_level = args.widest_peak_per_event or args.widest_peak_per_event_channel
    if event_level:
        peaks = widest_peak_per_event(peaks, per_channel=args.widest_peak_per_event_channel)
        if overlay_peaks is not None:
            overlay_peaks = widest_peak_per_event(overlay_peaks, per_channel=args.widest_peak_per_event_channel)
        if args.widest_peak_per_event_channel:
            print(f"Collapsed to {len(peaks)} event/channel entries with at least one detected peak.")
        else:
            print(f"Collapsed to {len(peaks)} events with at least one detected peak.")
    plot_histograms(
        peaks,
        args.output,
        bins=args.bins,
        max_width_us=args.max_width_us,
        log_y=not args.linear_y,
        log_x=args.log_x,
        overlay_peaks=overlay_peaks,
        overlay_label="Triggered/Dust",
        overlay_matched_events=overlay_matched_events,
        event_level=event_level,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
