#!/usr/bin/env python3
"""Analyze positive-going TOF peaks in IDEX L1B 10-day CDF products."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import cdflib
import numpy as np
from scipy.signal import find_peaks, peak_widths


CHANNELS: tuple[tuple[str, str], ...] = (
    ("tof_high", "TOF_High"),
    ("tof_mid", "TOF_Mid"),
    ("tof_low", "TOF_Low"),
)

DEFAULT_INPUT_DIR = Path(__file__).resolve().parent / "l1b"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "l1b_peak_analysis"
FILL_LIMIT = -1.0e30


def cdf_sources(input_dir: Path) -> list[Path]:
    sources = sorted(input_dir.glob("imap_idex_l1b_sci-10days_*.cdf"))
    if not sources:
        raise FileNotFoundError(f"No L1B 10-day CDF files found in {input_dir}")
    return sources


def epoch_to_iso(epoch_tt2000: int) -> str:
    try:
        encoded = cdflib.cdfepoch.encode_tt2000(int(epoch_tt2000))
    except Exception:
        return str(epoch_tt2000)
    if isinstance(encoded, list):
        return str(encoded[0]) if encoded else str(epoch_tt2000)
    return str(encoded)


def finite_waveform(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr.copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= FILL_LIMIT] = np.nan
    return arr


def baseline_mask(times: np.ndarray, baseline_us: float) -> np.ndarray:
    time_arr = np.asarray(times, dtype=float)
    mask = np.zeros(time_arr.size, dtype=bool)
    finite = np.isfinite(time_arr)
    if not np.any(finite):
        mask[: min(50, time_arr.size)] = True
        return mask

    finite_times = time_arr[finite]
    start = float(finite_times[0])
    window = abs(float(baseline_us))
    mask = finite & (time_arr >= start) & ((time_arr - start) <= window)
    if not np.any(mask):
        mask[: min(50, time_arr.size)] = True
    return mask


def robust_baseline(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    samples = values[mask & np.isfinite(values)]
    if samples.size == 0:
        samples = values[np.isfinite(values)]
    if samples.size == 0:
        return math.nan, math.nan, 0

    baseline = float(np.nanmedian(samples))
    deviations = samples - baseline
    mad = float(np.nanmedian(np.abs(deviations)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(samples))
    if not np.isfinite(sigma):
        sigma = math.nan
    return baseline, sigma, int(samples.size)


def interpolated_crossing_time(
    values: np.ndarray,
    times: np.ndarray,
    low_index: int,
    high_index: int,
    target: float,
) -> float:
    y0 = float(values[low_index])
    y1 = float(values[high_index])
    t0 = float(times[low_index])
    t1 = float(times[high_index])
    if not all(np.isfinite(item) for item in (y0, y1, t0, t1, target)):
        return math.nan
    if y1 == y0:
        return t0
    fraction = (target - y0) / (y1 - y0)
    return t0 + fraction * (t1 - t0)


def fwhm_width(
    corrected: np.ndarray,
    times: np.ndarray,
    peak_sample: int,
) -> tuple[float, float, float, float]:
    peak_height = float(corrected[peak_sample])
    if not np.isfinite(peak_height) or peak_height <= 0.0:
        return math.nan, math.nan, math.nan, math.nan
    half_height = 0.5 * peak_height

    left = int(peak_sample)
    while left > 0 and np.isfinite(corrected[left]) and corrected[left] >= half_height:
        left -= 1
    if left == int(peak_sample) or not np.isfinite(corrected[left]):
        left_time = math.nan
    elif corrected[left] < half_height:
        left_time = interpolated_crossing_time(corrected, times, left, left + 1, half_height)
    else:
        left_time = float(times[left])

    right = int(peak_sample)
    last_index = corrected.size - 1
    while right < last_index and np.isfinite(corrected[right]) and corrected[right] >= half_height:
        right += 1
    if right == int(peak_sample) or not np.isfinite(corrected[right]):
        right_time = math.nan
    elif corrected[right] < half_height:
        right_time = interpolated_crossing_time(corrected, times, right - 1, right, half_height)
    else:
        right_time = float(times[right])

    if not np.isfinite(left_time) or not np.isfinite(right_time):
        return math.nan, math.nan, left_time, right_time
    width_us = abs(float(right_time) - float(left_time))
    dt_us = float(np.nanmedian(np.abs(np.diff(times[np.isfinite(times)])))) if np.isfinite(times).sum() >= 2 else math.nan
    width_samples = width_us / dt_us if np.isfinite(dt_us) and dt_us > 0.0 else math.nan
    return width_samples, width_us, left_time, right_time


def positive_peak_metrics(
    values: np.ndarray,
    times: np.ndarray,
    *,
    baseline_us: float,
    threshold_sigma: float,
    prominence_sigma: float,
    min_distance_us: float,
    width_method: str,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    waveform = finite_waveform(values)
    time_arr = np.asarray(times, dtype=float)
    length = min(waveform.size, time_arr.size)
    waveform = waveform[:length]
    time_arr = time_arr[:length]

    mask = baseline_mask(time_arr, baseline_us)
    baseline, sigma, baseline_samples = robust_baseline(waveform, mask[:length])
    corrected = waveform - baseline

    finite = np.isfinite(corrected) & np.isfinite(time_arr)
    if not np.any(finite) or not np.isfinite(sigma) or sigma <= 0.0:
        return {
            "baseline": baseline,
            "baseline_sigma": sigma,
            "baseline_samples": baseline_samples,
            "peak_count": 0,
        }, []

    search = corrected.copy()
    search[~finite] = -np.inf

    finite_times = time_arr[finite]
    if finite_times.size >= 2:
        dt_us = float(np.nanmedian(np.abs(np.diff(finite_times))))
    else:
        dt_us = math.nan
    min_distance_samples = 1
    if np.isfinite(dt_us) and dt_us > 0.0 and min_distance_us > 0.0:
        min_distance_samples = max(1, int(round(min_distance_us / dt_us)))

    height = float(threshold_sigma) * sigma
    prominence = float(prominence_sigma) * sigma
    peaks, properties = find_peaks(
        search,
        height=height,
        prominence=prominence,
        distance=min_distance_samples,
    )

    summary: dict[str, float | int] = {
        "baseline": baseline,
        "baseline_sigma": sigma,
        "baseline_samples": baseline_samples,
        "peak_count": int(peaks.size),
    }
    if peaks.size == 0:
        return summary, []

    if width_method == "half_prominence":
        widths = peak_widths(search, peaks, rel_height=0.5)
        left_ips = np.asarray(widths[2], dtype=float)
        right_ips = np.asarray(widths[3], dtype=float)
        sample_index = np.arange(length, dtype=float)
        left_times = np.interp(left_ips, sample_index, time_arr)
        right_times = np.interp(right_ips, sample_index, time_arr)
        width_samples = np.asarray(widths[0], dtype=float)
        width_us = np.abs(right_times - left_times)
    else:
        width_values = [fwhm_width(search, time_arr, int(peak_sample)) for peak_sample in peaks]
        width_samples = np.asarray([item[0] for item in width_values], dtype=float)
        width_us = np.asarray([item[1] for item in width_values], dtype=float)
        left_times = np.asarray([item[2] for item in width_values], dtype=float)
        right_times = np.asarray([item[3] for item in width_values], dtype=float)

    peak_rows: list[dict[str, float | int]] = []
    peak_heights = np.asarray(properties.get("peak_heights", []), dtype=float)
    prominences = np.asarray(properties.get("prominences", []), dtype=float)
    for idx, peak_sample in enumerate(peaks):
        left_time = float(left_times[idx])
        right_time = float(right_times[idx])
        peak_rows.append(
            {
                "peak_index": idx,
                "sample_index": int(peak_sample),
                "peak_time_us": float(time_arr[peak_sample]),
                "peak_height": float(peak_heights[idx]) if idx < peak_heights.size else float(search[peak_sample]),
                "peak_prominence": float(prominences[idx]) if idx < prominences.size else math.nan,
                "width_method": width_method,
                "width_samples": float(width_samples[idx]),
                "width_us": float(width_us[idx]),
                "left_time_us": left_time,
                "right_time_us": right_time,
            }
        )
    return summary, peak_rows


def write_products(
    sources: Iterable[Path],
    output_dir: Path,
    *,
    baseline_us: float,
    threshold_sigma: float,
    prominence_sigma: float,
    min_distance_us: float,
    width_method: str,
    output_prefix: str,
    max_events: int | None,
) -> tuple[Path, Path, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_summary_path = output_dir / f"{output_prefix}_event_summary.csv"
    peaks_path = output_dir / f"{output_prefix}_peaks.csv"

    summary_fields = [
        "source_file",
        "record_index",
        "epoch_tt2000",
        "epoch_utc",
        "channel",
        "cdf_variable",
        "baseline",
        "baseline_sigma",
        "baseline_samples",
        "peak_count",
    ]
    peak_fields = [
        "source_file",
        "record_index",
        "epoch_tt2000",
        "epoch_utc",
        "channel",
        "cdf_variable",
        "peak_index",
        "sample_index",
        "peak_time_us",
        "peak_height",
        "peak_prominence",
        "width_method",
        "width_samples",
        "width_us",
        "left_time_us",
        "right_time_us",
    ]

    event_count = 0
    peak_count = 0
    with event_summary_path.open("w", newline="", encoding="utf-8") as summary_file, peaks_path.open(
        "w", newline="", encoding="utf-8"
    ) as peaks_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        peaks_writer = csv.DictWriter(peaks_file, fieldnames=peak_fields)
        summary_writer.writeheader()
        peaks_writer.writeheader()

        for source in sources:
            cdf = cdflib.CDF(str(source))
            epochs = np.asarray(cdf.varget("epoch"), dtype=np.int64)
            times = np.asarray(cdf.varget("time_high_sample_rate"), dtype=float)
            channel_data = {var_name: np.asarray(cdf.varget(var_name), dtype=float) for _label, var_name in CHANNELS}

            for record_index, epoch in enumerate(epochs):
                if max_events is not None and event_count >= max_events:
                    return event_summary_path, peaks_path, event_count, peak_count
                event_count += 1
                event_times = times[record_index]
                epoch_text = epoch_to_iso(int(epoch))

                for channel_label, var_name in CHANNELS:
                    summary, peak_rows = positive_peak_metrics(
                        channel_data[var_name][record_index],
                        event_times,
                        baseline_us=baseline_us,
                        threshold_sigma=threshold_sigma,
                        prominence_sigma=prominence_sigma,
                        min_distance_us=min_distance_us,
                        width_method=width_method,
                    )
                    common = {
                        "source_file": source.name,
                        "record_index": record_index,
                        "epoch_tt2000": int(epoch),
                        "epoch_utc": epoch_text,
                        "channel": channel_label,
                        "cdf_variable": var_name,
                    }
                    summary_writer.writerow({**common, **summary})
                    for peak_row in peak_rows:
                        peaks_writer.writerow({**common, **peak_row})
                    peak_count += len(peak_rows)

    return event_summary_path, peaks_path, event_count, peak_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-us", type=float, default=3.0)
    parser.add_argument("--threshold-sigma", type=float, default=5.0)
    parser.add_argument("--prominence-sigma", type=float, default=5.0)
    parser.add_argument("--min-distance-us", type=float, default=0.03)
    parser.add_argument(
        "--width-method",
        choices=("half_prominence", "fwhm"),
        default="half_prominence",
        help="Peak width definition. half_prominence matches scipy.signal.peak_widths(rel_height=0.5); "
        "fwhm uses baseline-subtracted half-maximum crossings.",
    )
    parser.add_argument("--output-prefix", default="l1b_tof_peak")
    parser.add_argument("--max-events", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()

    sources = cdf_sources(args.input_dir)
    summary_path, peaks_path, event_count, peak_count = write_products(
        sources,
        args.output_dir,
        baseline_us=args.baseline_us,
        threshold_sigma=args.threshold_sigma,
        prominence_sigma=args.prominence_sigma,
        min_distance_us=args.min_distance_us,
        width_method=args.width_method,
        output_prefix=args.output_prefix,
        max_events=args.max_events,
    )

    print(f"Analyzed {event_count} L1B events across {len(sources)} CDF files.")
    print(f"Detected {peak_count} TOF peaks.")
    print(f"Wrote event summary: {summary_path}")
    print(f"Wrote peak table: {peaks_path}")


if __name__ == "__main__":
    main()
