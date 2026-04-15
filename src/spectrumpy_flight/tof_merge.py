from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

GAIN_HIGH = 1600.0
GAIN_MEDIUM = 40.0
GAIN_LOW = 1.0

GAIN_MAP: Dict[str, float] = {
    "TOF H": GAIN_HIGH,
    "TOF M": GAIN_MEDIUM,
    "TOF L": GAIN_LOW,
}


def _contiguous_mask(condition: np.ndarray, min_samples: int) -> np.ndarray:
    if condition.size == 0:
        return np.zeros(0, dtype=bool)
    padded = np.concatenate(([False], condition, [False])).astype(int)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    mask = np.zeros_like(condition, dtype=bool)
    for start, end in zip(starts, ends):
        if end - start >= max(1, min_samples):
            mask[start:end] = True
    return mask


def _median_baseline_subtract(values: np.ndarray, *, samples: int = 200) -> Tuple[np.ndarray, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr, 0.0

    npre = min(int(samples), arr.size)
    if npre <= 0:
        return arr, 0.0

    baseline = float(np.nanmedian(arr[:npre]))
    return arr - baseline, baseline


def detect_saturation(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)

    magnitude = np.nanmax(np.abs(arr))
    if not np.isfinite(magnitude) or magnitude == 0.0:
        return np.zeros_like(arr, dtype=bool)

    grad = np.abs(np.gradient(arr))
    derivative_threshold = 0.0025 * magnitude
    plateau = grad < derivative_threshold

    repeated = np.zeros_like(arr, dtype=bool)
    if arr.size >= 2:
        diffs = np.abs(np.diff(arr))
        repeat_tol = max(1.0e-9, 1.0e-4 * magnitude)
        repeats = diffs <= repeat_tol
        if repeats.any():
            repeated[1:] |= repeats
            repeated[:-1] |= repeats

    amplitude_threshold = np.nanpercentile(np.abs(arr), 99.7)
    high_amp = np.abs(arr) >= amplitude_threshold
    plateau_mask = (plateau | repeated) & high_amp

    extreme_mask = np.zeros_like(arr, dtype=bool)
    tolerance = 0.003 * magnitude + 1.0e-9
    max_val = float(np.nanmax(arr))
    min_val = float(np.nanmin(arr))
    if np.isfinite(max_val) and max_val > 0.0:
        extreme_mask |= (max_val - arr) <= tolerance
    if np.isfinite(min_val) and min_val < 0.0:
        extreme_mask |= (arr - min_val) <= tolerance
    plateau_mask |= extreme_mask & high_amp

    if plateau_mask.size < 2:
        return plateau_mask

    # Short clipped plateaus are common in these TOF traces; requiring a
    # microsecond-scale duration over-filters legitimate saturation runs on the
    # high-rate channels. A small contiguous-run requirement matches the
    # instrument's quantised clipping behavior much better across HDF5, L1A,
    # and L1B products.
    return _contiguous_mask(plateau_mask, 3)


def _first_microsecond_mean(values: Optional[np.ndarray], times: Optional[np.ndarray]) -> float:
    if values is None:
        return 0.0

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0

    if times is None:
        sample_count = min(arr.size, 50)
        if sample_count == 0:
            return 0.0
        return float(np.nanmean(arr[:sample_count]))

    time_arr = np.asarray(times, dtype=float)
    if time_arr.size == 0:
        sample_count = min(arr.size, 50)
        if sample_count == 0:
            return 0.0
        return float(np.nanmean(arr[:sample_count]))

    length = min(arr.size, time_arr.size)
    if length == 0:
        return 0.0

    arr = arr[:length]
    time_arr = time_arr[:length]

    if length >= 2:
        diffs = np.diff(time_arr)
        finite_diffs = diffs[np.isfinite(diffs) & (diffs != 0.0)]
        dt = float(np.nanmedian(np.abs(finite_diffs))) if finite_diffs.size else 0.0
        direction = 0.0
        for diff in diffs:
            if np.isfinite(diff) and diff != 0.0:
                direction = math.copysign(1.0, diff)
                break
    else:
        dt = 0.0
        direction = 0.0

    if np.isfinite(dt) and dt > 0.0 and dt < 1.0e-6:
        window = 1.0e-6
    else:
        window = 1.0

    start = float(time_arr[0])
    if direction > 0.0:
        mask = (time_arr >= start) & ((time_arr - start) <= window)
    elif direction < 0.0:
        mask = (time_arr <= start) & ((start - time_arr) <= window)
    else:
        mask = np.abs(time_arr - start) <= window

    mask[0] = True

    if not np.any(mask):
        if np.isfinite(dt) and dt > 0.0:
            samples = int(math.ceil(window / dt))
        else:
            samples = 50
        samples = min(max(samples, 1), length)
        mask = np.zeros(length, dtype=bool)
        mask[:samples] = True

    return float(np.nanmean(arr[mask]))


def combine_mid_low_waveforms(
    time_axis: np.ndarray,
    medium: Optional[np.ndarray],
    low: Optional[np.ndarray],
    *,
    gain_map: Optional[Dict[str, float]] = None,
) -> Optional[np.ndarray]:
    if medium is None or low is None or time_axis is None:
        return None

    times = np.asarray(time_axis, dtype=float)
    mid_arr = np.asarray(medium, dtype=float)
    low_arr = np.asarray(low, dtype=float)

    length = min(times.size, mid_arr.size, low_arr.size)
    if length == 0:
        return None

    mid_arr = mid_arr[:length]
    low_arr = low_arr[:length]

    mid_baseline_sub, mid_baseline = _median_baseline_subtract(mid_arr)
    low_baseline_sub, _ = _median_baseline_subtract(low_arr)

    gain_map = gain_map or GAIN_MAP
    mid_gain = float(gain_map.get("TOF M", GAIN_MEDIUM))
    low_gain = float(gain_map.get("TOF L", GAIN_LOW))
    scale = mid_gain / low_gain if low_gain else 1.0
    low_scaled = low_baseline_sub * scale

    if not mid_baseline_sub.size:
        return mid_baseline_sub

    peak = float(np.nanmax(mid_baseline_sub))
    if not np.isfinite(peak):
        return mid_baseline_sub

    saturation_threshold = 0.90 * peak
    saturation_mask = mid_baseline_sub >= saturation_threshold

    combined = mid_baseline_sub.copy()
    replace_mask = saturation_mask & np.isfinite(low_scaled)
    replace_mask |= (~np.isfinite(combined)) & np.isfinite(low_scaled)
    combined[replace_mask] = low_scaled[replace_mask]

    return combined + mid_baseline


def combine_waveform_channels(
    time_axis: np.ndarray,
    high: Optional[np.ndarray],
    medium: Optional[np.ndarray],
    low: Optional[np.ndarray],
    gain_map: Optional[Dict[str, float]] = None,
    enabled_channels: Optional[Iterable[str]] = None,
) -> Optional[np.ndarray]:
    if time_axis is None:
        return None

    times = np.asarray(time_axis, dtype=float)
    if times.size == 0:
        return None

    gain_map = gain_map or GAIN_MAP

    valid_order = ("TOF H", "TOF M", "TOF L")
    selected_set: Optional[Set[str]] = None
    selected: Optional[List[str]] = None
    if enabled_channels is not None:
        selected_set = {str(name) for name in enabled_channels}
        selected_list = [name for name in valid_order if name in selected_set]
        if selected_list:
            selected = selected_list

    channel_map: Dict[str, Optional[np.ndarray]] = {
        "TOF H": high,
        "TOF M": medium,
        "TOF L": low,
    }

    arrays = [
        channel_map[name]
        for name in valid_order
        if (selected is None or name in selected)
        and channel_map[name] is not None
        and getattr(channel_map[name], "size", 0)
    ]
    if not arrays:
        return None

    if selected_set == {"TOF M", "TOF L"}:
        return combine_mid_low_waveforms(time_axis, medium, low, gain_map=gain_map)

    lengths = [times.size]
    lengths.extend(arr.size for arr in arrays)
    length = min(lengths)
    if length <= 0:
        return None

    times = times[:length]
    channel_entries: List[Tuple[str, np.ndarray]] = []
    for name in valid_order:
        if selected is not None and name not in selected:
            continue
        arr = channel_map[name]
        if arr is None or not getattr(arr, "size", 0):
            continue
        channel_entries.append((name, np.asarray(arr[:length], dtype=float)))

    if not channel_entries:
        return None

    target_name = channel_entries[0][0]
    target_gain = float(gain_map.get(target_name, 1.0))

    corrected_channels: List[Dict[str, Any]] = []
    for name, arr in channel_entries:
        baseline = _first_microsecond_mean(arr, times)
        corrected = arr - baseline
        gain = float(gain_map.get(name, 1.0))
        scale = target_gain / gain if gain else 1.0
        scaled = corrected * scale
        saturation = detect_saturation(arr, times)
        corrected_channels.append(
            {
                "name": name,
                "baseline": baseline,
                "scaled": scaled,
                "saturation": saturation,
            }
        )

    primary = corrected_channels[0]
    primary_scaled = np.asarray(primary["scaled"], dtype=float)
    combined_corrected = primary_scaled.copy()
    primary_saturation = np.asarray(primary["saturation"], dtype=bool)
    remaining_mask = primary_saturation.copy()
    if combined_corrected.size:
        with np.errstate(invalid="ignore"):
            remaining_mask |= ~np.isfinite(combined_corrected)

    for entry in corrected_channels[1:]:
        candidate = np.asarray(entry["scaled"], dtype=float)
        candidate_saturation = np.asarray(entry["saturation"], dtype=bool)
        finite_candidate = np.isfinite(candidate)

        replace_mask = remaining_mask & ~candidate_saturation & finite_candidate

        if combined_corrected.size:
            with np.errstate(invalid="ignore"):
                replace_mask |= (~np.isfinite(combined_corrected)) & ~candidate_saturation & finite_candidate

        combined_corrected[replace_mask] = candidate[replace_mask]
        remaining_mask &= candidate_saturation
        remaining_mask &= ~replace_mask

    primary_baseline = float(primary.get("baseline", 0.0))
    if np.isfinite(primary_baseline):
        return combined_corrected + primary_baseline
    return combined_corrected
