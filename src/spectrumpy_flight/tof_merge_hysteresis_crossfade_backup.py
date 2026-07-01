from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

GAIN_HIGH = 2.89e-4
GAIN_MEDIUM = 1.13e-2
GAIN_LOW = 5.14e-1

GAIN_MAP: Dict[str, float] = {
    "TOF H": GAIN_HIGH,
    "TOF M": GAIN_MEDIUM,
    "TOF L": GAIN_LOW,
}

NOISE_BASELINE_WINDOW_US = 3.0
NOISE_THRESHOLD_SIGMA    = 5.0
RAIL_HIGH   = 0.148
RAIL_MEDIUM = 5.786
RAIL_LOW    = 263.7
CROSSFADE_SAMPLES     = 20
RETURN_STABLE_SAMPLES = 20
SWITCH_RAIL_FRACTION  = 0.85
RETURN_RAIL_FRACTION  = 0.85

RAIL_MAP: Dict[str, float] = {
    "TOF H": RAIL_HIGH,
    "TOF M": RAIL_MEDIUM,
    "TOF L": RAIL_LOW,
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


def _baseline_window_mask(
    values: np.ndarray,
    times: Optional[np.ndarray],
    *,
    window_us: float = NOISE_BASELINE_WINDOW_US,
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(0, dtype=bool)

    if times is None:
        sample_count = min(arr.size, 50)
        mask = np.zeros(arr.size, dtype=bool)
        mask[:sample_count] = True
        return mask

    time_arr = np.asarray(times, dtype=float)
    length = min(arr.size, time_arr.size)
    mask = np.zeros(arr.size, dtype=bool)
    if length == 0:
        sample_count = min(arr.size, 50)
        mask[:sample_count] = True
        return mask

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
        window = float(window_us) * 1.0e-6
    else:
        window = float(window_us)

    start = float(time_arr[0])
    if direction > 0.0:
        local_mask = (time_arr >= start) & ((time_arr - start) <= window)
    elif direction < 0.0:
        local_mask = (time_arr <= start) & ((start - time_arr) <= window)
    else:
        local_mask = np.abs(time_arr - start) <= window

    local_mask[0] = True

    if not np.any(local_mask):
        if np.isfinite(dt) and dt > 0.0:
            samples = int(math.ceil(window / dt))
        else:
            samples = 50
        samples = min(max(samples, 1), length)
        local_mask = np.zeros(length, dtype=bool)
        local_mask[:samples] = True

    mask[:length] = local_mask
    return mask


def _baseline_noise_stats(
    values: Optional[np.ndarray],
    times: Optional[np.ndarray],
    *,
    window_us: float = NOISE_BASELINE_WINDOW_US,
) -> Tuple[float, float]:
    if values is None:
        return 0.0, 0.0

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0

    mask = _baseline_window_mask(arr, times, window_us=window_us)
    samples = arr[mask & np.isfinite(arr)]
    if samples.size == 0:
        samples = arr[np.isfinite(arr)]
    if samples.size == 0:
        return 0.0, 0.0

    baseline = float(np.nanmedian(samples))
    deviations = samples - baseline
    mad = float(np.nanmedian(np.abs(deviations)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.nanstd(samples))
    if not np.isfinite(sigma):
        sigma = 0.0
    return baseline, sigma


def _first_microsecond_mean(values: Optional[np.ndarray], times: Optional[np.ndarray]) -> float:
    baseline, _ = _baseline_noise_stats(values, times, window_us=1.0)
    return baseline


def _raised_cosine_blend(
    output: np.ndarray,
    from_values: np.ndarray,
    to_values: np.ndarray,
    start: int,
    end: int,
) -> None:
    start = max(0, int(start))
    end = min(int(end), output.size)
    if end <= start:
        return
    count = end - start
    if count == 1:
        weights = np.array([1.0], dtype=float)
    else:
        x = np.linspace(0.0, 1.0, count)
        weights = 0.5 - 0.5 * np.cos(np.pi * x)
    source = np.asarray(from_values[start:end], dtype=float)
    target = np.asarray(to_values[start:end], dtype=float)
    blended = (1.0 - weights) * source + weights * target
    finite_source = np.isfinite(source)
    finite_target = np.isfinite(target)
    blended = np.where(finite_source & finite_target, blended, np.where(finite_target, target, source))
    output[start:end] = blended


def _combine_hysteresis_crossfade(
    times: np.ndarray,
    channel_entries: List[Tuple[str, np.ndarray]],
    gain_map: Dict[str, float],
    *,
    scale_to_gain: bool,
    length: int,
) -> Optional[np.ndarray]:
    if not channel_entries or length <= 0:
        return None

    names: List[str] = []
    corrected_values: List[np.ndarray] = []
    rails: List[float] = []
    for name, arr in channel_entries:
        baseline, _ = _baseline_noise_stats(arr, times)
        corrected = np.asarray(arr[:length], dtype=float) - baseline
        if scale_to_gain:
            corrected = corrected * float(gain_map.get(name, 1.0))
        names.append(name)
        corrected_values.append(corrected)
        rails.append(float(RAIL_MAP.get(name, np.inf)))

    if not corrected_values:
        return None

    if len(corrected_values) == 1:
        return corrected_values[0].copy()

    output = np.full(length, np.nan, dtype=float)
    current = 0
    return_count = 0

    for i in range(length):
        current_values = corrected_values[current]
        current_value = current_values[i]
        current_rail = rails[current]

        if current < len(corrected_values) - 1:
            at_switch_limit = (
                np.isfinite(current_value)
                and np.isfinite(current_rail)
                and abs(current_value) >= SWITCH_RAIL_FRACTION * current_rail
            )
            if at_switch_limit or not np.isfinite(current_value):
                lower = current + 1
                blend_start = max(0, i - CROSSFADE_SAMPLES)
                _raised_cosine_blend(
                    output,
                    corrected_values[current],
                    corrected_values[lower],
                    blend_start,
                    i,
                )
                current = lower
                return_count = 0
                current_values = corrected_values[current]
                current_value = current_values[i]

        if current > 0:
            higher = current - 1
            higher_value = corrected_values[higher][i]
            higher_rail = rails[higher]
            below_return_limit = (
                np.isfinite(higher_value)
                and np.isfinite(higher_rail)
                and abs(higher_value) <= RETURN_RAIL_FRACTION * higher_rail
            )
            if below_return_limit:
                return_count += 1
            else:
                return_count = 0

            if return_count >= RETURN_STABLE_SAMPLES:
                blend_start = max(0, i - CROSSFADE_SAMPLES + 1)
                _raised_cosine_blend(
                    output,
                    corrected_values[current],
                    corrected_values[higher],
                    blend_start,
                    i + 1,
                )
                current = higher
                return_count = 0
                current_values = corrected_values[current]
                current_value = current_values[i]

        output[i] = current_value

    return output


def combine_mid_low_waveforms(
    time_axis: np.ndarray,
    medium: Optional[np.ndarray],
    low: Optional[np.ndarray],
    *,
    gain_map: Optional[Dict[str, float]] = None,
    scale_to_gain: bool = False,
) -> Optional[np.ndarray]:
    return combine_waveform_channels(
        time_axis,
        None,
        medium,
        low,
        gain_map=gain_map,
        enabled_channels=("TOF M", "TOF L"),
        scale_to_gain=scale_to_gain,
    )


def combine_waveform_channels(
    time_axis: np.ndarray,
    high: Optional[np.ndarray],
    medium: Optional[np.ndarray],
    low: Optional[np.ndarray],
    gain_map: Optional[Dict[str, float]] = None,
    enabled_channels: Optional[Iterable[str]] = None,
    scale_to_gain: bool = False,
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

    return _combine_hysteresis_crossfade(
        times,
        channel_entries,
        gain_map,
        scale_to_gain=scale_to_gain,
        length=length,
    )
