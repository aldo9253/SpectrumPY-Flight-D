from __future__ import annotations

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

# The TOF ADC is 10-bit. Treat values at or above this DN ceiling as saturated
# and fall back to the next lower gain stage.
DN_SATURATION_LIMIT = 1020.0

# The packet/HDF5 writers convert raw DN into engineering units by multiplying
# by the channel-specific conversion factor. Use the same factors here so the
# saturation threshold is evaluated in the same units as the stored waveform.
TOF_CONVERSION_FACTORS: Dict[str, float] = {
    "TOF H": 2.89e-4,
    "TOF M": 1.13e-2,
    "TOF L": 5.14e-4,
}

# Treat midscale as the zero point for TOF combine.
DN_MIDPOINT = 512.0
MAX_SATURATION_GAP_US = 0.4
# Keep a little hysteresis below the hard saturation limit so ringing traces
# do not drop back to a higher-gain channel too early.
SATURATION_RELEASE_FRACTION = 0.89


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


def _bridge_short_false_gaps(
    mask: np.ndarray,
    times: np.ndarray,
    *,
    max_gap_us: float,
) -> np.ndarray:
    if mask.size == 0 or max_gap_us <= 0:
        return np.asarray(mask, dtype=bool)

    result = np.asarray(mask, dtype=bool).copy()
    time_arr = np.asarray(times, dtype=float)
    if time_arr.size != result.size or result.size == 0:
        return result

    padded = np.concatenate(([False], result, [False])).astype(int)
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size < 2:
        return result

    for left_end, right_start in zip(ends[:-1], starts[1:]):
        if left_end == 0 or right_start >= time_arr.size:
            continue
        gap = abs(float(time_arr[right_start]) - float(time_arr[left_end - 1]))
        if gap <= max_gap_us:
            result[left_end:right_start] = True
    return result


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


def _to_midpoint_corrected(values: np.ndarray, channel: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    offset = DN_MIDPOINT * TOF_CONVERSION_FACTORS.get(channel, 1.0)
    return arr - offset


def _to_reference_gain_scale(
    values: np.ndarray,
    channel: str,
    *,
    gain_map: Optional[Dict[str, float]],
    reference_channel: str = "TOF M",
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    active_gain_map = gain_map or GAIN_MAP
    reference_gain = float(active_gain_map.get(reference_channel, GAIN_MEDIUM))
    if not np.isfinite(reference_gain) or reference_gain <= 0.0:
        reference_gain = GAIN_MEDIUM
    channel_gain = float(active_gain_map.get(channel, 1.0))
    if not np.isfinite(channel_gain) or channel_gain <= 0.0:
        channel_gain = 1.0
    return arr * (channel_gain / reference_gain)


def combine_mid_low_waveforms(
    time_axis: np.ndarray,
    medium: Optional[np.ndarray],
    low: Optional[np.ndarray],
    *,
    gain_map: Optional[Dict[str, float]] = None,
    max_saturation_gap_us: float = MAX_SATURATION_GAP_US,
    saturation_release_fraction: float = SATURATION_RELEASE_FRACTION,
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

    gain_map = gain_map or GAIN_MAP
    mid_corrected = _to_midpoint_corrected(mid_arr, "TOF M")
    mid_corrected = _to_reference_gain_scale(mid_corrected, "TOF M", gain_map=gain_map)
    low_corrected = _to_midpoint_corrected(low_arr, "TOF L")
    low_corrected = _to_reference_gain_scale(low_corrected, "TOF L", gain_map=gain_map)

    mid_threshold = DN_SATURATION_LIMIT

    if not mid_corrected.size:
        return mid_corrected

    saturation_mask = (mid_arr / TOF_CONVERSION_FACTORS.get("TOF M", 1.0)) >= mid_threshold
    saturation_mask |= (mid_arr / TOF_CONVERSION_FACTORS.get("TOF M", 1.0)) >= (
        DN_SATURATION_LIMIT * saturation_release_fraction
    )
    saturation_mask = _bridge_short_false_gaps(
        saturation_mask,
        times,
        max_gap_us=max_saturation_gap_us,
    )

    combined = mid_corrected.copy()
    replace_mask = saturation_mask & np.isfinite(low_corrected)
    replace_mask |= (~np.isfinite(combined)) & np.isfinite(low_corrected)
    combined[replace_mask] = low_corrected[replace_mask]

    return combined


def combine_waveform_channels(
    time_axis: np.ndarray,
    high: Optional[np.ndarray],
    medium: Optional[np.ndarray],
    low: Optional[np.ndarray],
    gain_map: Optional[Dict[str, float]] = None,
    enabled_channels: Optional[Iterable[str]] = None,
    max_saturation_gap_us: float = MAX_SATURATION_GAP_US,
    saturation_release_fraction: float = SATURATION_RELEASE_FRACTION,
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
        return combine_mid_low_waveforms(
            time_axis,
            medium,
            low,
            gain_map=gain_map,
            max_saturation_gap_us=max_saturation_gap_us,
            saturation_release_fraction=saturation_release_fraction,
        )

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

    corrected_channels: List[Dict[str, Any]] = []
    for name, arr in channel_entries:
        corrected = _to_midpoint_corrected(arr, name)
        conversion = TOF_CONVERSION_FACTORS.get(name, 1.0)
        gain = float(gain_map.get(name, 1.0))
        dn_values = np.asarray(arr, dtype=float) / conversion
        saturation = dn_values >= DN_SATURATION_LIMIT
        saturation |= dn_values >= (DN_SATURATION_LIMIT * saturation_release_fraction)
        # Catch clipped ring-downs that dip below the hard ceiling but still
        # retain the plateau / repeat signature of saturation.
        saturation |= detect_saturation(dn_values, times)
        corrected = _to_reference_gain_scale(
            corrected,
            name,
            gain_map=gain_map,
            reference_channel="TOF M",
        )
        saturation = _bridge_short_false_gaps(
            saturation,
            times,
            max_gap_us=max_saturation_gap_us,
        )
        corrected_channels.append(
            {
                "name": name,
                "corrected": corrected,
                "gain": gain,
                "saturation": saturation,
            }
        )

    corrected_stack = np.vstack([np.asarray(entry["corrected"], dtype=float) for entry in corrected_channels])
    saturation_stack = np.vstack([np.asarray(entry["saturation"], dtype=bool) for entry in corrected_channels])
    gains = np.asarray([float(entry["gain"]) for entry in corrected_channels], dtype=float)
    finite_stack = np.isfinite(corrected_stack)
    unsaturated_stack = finite_stack & ~saturation_stack

    combined = np.full(length, np.nan, dtype=float)

    has_unsaturated = np.any(unsaturated_stack, axis=0)
    if np.any(has_unsaturated):
        best_unsaturated = np.argmax(np.where(unsaturated_stack, gains[:, None], -np.inf), axis=0)
        col_idx = np.nonzero(has_unsaturated)[0]
        combined[col_idx] = corrected_stack[best_unsaturated[col_idx], col_idx]

    remaining = ~has_unsaturated
    if np.any(remaining):
        has_saturated = np.any(finite_stack, axis=0)
        fallback = remaining & has_saturated
        if np.any(fallback):
            best_saturated = np.argmin(np.where(finite_stack, gains[:, None], np.inf), axis=0)
            col_idx = np.nonzero(fallback)[0]
            combined[col_idx] = corrected_stack[best_saturated[col_idx], col_idx]

    return combined
