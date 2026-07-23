import sys
import types

import pytest


def _install_qt_stubs():
    if "PySide6" in sys.modules or "PyQt6" in sys.modules:
        return

    qt_module = types.ModuleType("PySide6")
    qt_module.__version__ = "0.0"
    qt_core = types.ModuleType("PySide6.QtCore")
    qt_gui = types.ModuleType("PySide6.QtGui")
    qt_widgets = types.ModuleType("PySide6.QtWidgets")

    Qt = types.SimpleNamespace(
        KeyboardModifier=types.SimpleNamespace(ControlModifier=0x01000000),
        ItemDataRole=types.SimpleNamespace(UserRole=32, FontRole=6, ForegroundRole=9),
        AlignmentFlag=types.SimpleNamespace(AlignCenter=0x0004, AlignRight=0x0002, AlignVCenter=0x0020),
        TextFormat=types.SimpleNamespace(RichText=1),
        ScrollBarPolicy=types.SimpleNamespace(ScrollBarAsNeeded=0, ScrollBarAlwaysOff=1),
        Orientation=types.SimpleNamespace(Horizontal=0),
        ItemFlag=types.SimpleNamespace(ItemIsEditable=0x0002),
    )

    palette_cls = type("QPalette", (), {"ColorRole": types.SimpleNamespace(Mid=0)})

    qt_core.Qt = Qt
    qt_gui.QPalette = palette_cls

    widget_names = [
        "QAbstractItemView",
        "QCheckBox",
        "QComboBox",
        "QDialog",
        "QDialogButtonBox",
        "QDoubleSpinBox",
        "QFormLayout",
        "QGridLayout",
        "QGroupBox",
        "QHBoxLayout",
        "QLabel",
        "QLineEdit",
        "QMainWindow",
        "QMessageBox",
        "QPushButton",
        "QScrollArea",
        "QSizePolicy",
        "QSplitter",
        "QStatusBar",
        "QTableWidget",
        "QTableWidgetItem",
        "QVBoxLayout",
        "QWidget",
        "QToolButton",
    ]

    for name in widget_names:
        setattr(qt_widgets, name, type(name, (), {}))

    qt_widgets.QSizePolicy.Policy = types.SimpleNamespace(Expanding=0)

    qt_module.QtCore = qt_core
    qt_module.QtGui = qt_gui
    qt_module.QtWidgets = qt_widgets

    sys.modules["PySide6"] = qt_module
    sys.modules["PySide6.QtCore"] = qt_core
    sys.modules["PySide6.QtGui"] = qt_gui
    sys.modules["PySide6.QtWidgets"] = qt_widgets


_install_qt_stubs()

if "matplotlib.backends.backend_qtagg" not in sys.modules:
    backend_stub = types.ModuleType("matplotlib.backends.backend_qtagg")

    class _Canvas:
        def __init__(self, *args, **kwargs):
            pass

    class _Toolbar:
        def __init__(self, *args, **kwargs):
            pass

    backend_stub.FigureCanvasQTAgg = _Canvas
    backend_stub.NavigationToolbar2QT = _Toolbar

    import importlib

    backends = importlib.import_module("matplotlib.backends")
    setattr(backends, "backend_qtagg", backend_stub)
    sys.modules["matplotlib.backends.backend_qtagg"] = backend_stub

np = pytest.importorskip("numpy")

from spectrumpy_flight.dust_composition import (
    GAIN_HIGH,
    GAIN_LOW,
    GAIN_MEDIUM,
    combine_waveform_channels,
    detect_saturation,
)
from spectrumpy_flight.tof_merge import DN_MIDPOINT, DN_SATURATION_LIMIT, SATURATION_RELEASE_FRACTION, TOF_CONVERSION_FACTORS

SATURATION_EFFECTIVE_LIMIT = DN_SATURATION_LIMIT * SATURATION_RELEASE_FRACTION
HIGH_GAIN_SCALE = GAIN_HIGH / GAIN_MEDIUM
MID_GAIN_SCALE = 1.0
LOW_GAIN_SCALE = GAIN_LOW / GAIN_MEDIUM


def _normalize_expected(values: np.ndarray, channel: str) -> np.ndarray:
    if channel == "TOF H":
        scale = HIGH_GAIN_SCALE
        offset = DN_MIDPOINT * TOF_CONVERSION_FACTORS["TOF H"]
    elif channel == "TOF L":
        scale = LOW_GAIN_SCALE
        offset = DN_MIDPOINT * TOF_CONVERSION_FACTORS["TOF L"]
    else:
        scale = MID_GAIN_SCALE
        offset = DN_MIDPOINT * TOF_CONVERSION_FACTORS["TOF M"]
    return (np.asarray(values, dtype=float) - offset) * scale


def test_detect_saturation_flags_clipped_segments_with_jitter():
    times = np.linspace(0.0, 31.5, 4096)
    base = 0.24 * np.exp(-0.5 * ((times - 6.0) / 0.8) ** 2)
    clip_level = 0.19
    signal = base.copy()
    clipped = base >= clip_level

    signal[clipped] = clip_level

    mask = detect_saturation(signal, times)

    saturated_window = (times >= 5.2) & (times <= 6.5)
    assert saturated_window.any()
    core_window = (times >= 5.4) & (times <= 6.3)
    assert core_window.any()
    assert mask[core_window].mean() > 0.8

    unsaturated_window = (times <= 4.5) | (times >= 7.0)
    assert unsaturated_window.any()
    assert not np.any(mask[unsaturated_window])


def test_combine_waveform_channels_replaces_saturated_high_with_medium():
    times = np.linspace(0.0, 31.5, 4096)
    physical_signal = 0.5 * np.exp(-0.5 * ((times - 6.0) / 0.4) ** 2)

    high_baseline = 900.0
    medium_baseline = 75.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM
    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]

    saturation_mask = high_raw >= SATURATION_EFFECTIVE_LIMIT

    combined = combine_waveform_channels(times, high, medium, None)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    expected[saturation_mask] = _normalize_expected(medium, "TOF M")[saturation_mask]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_uses_low_when_high_and_medium_saturate():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 30.0 * np.exp(-((times - 7.5) / 0.35) ** 2)

    high_baseline = 980.0
    medium_baseline = 63.0
    low_baseline = -4.5

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM
    low_raw = low_baseline + physical_signal * GAIN_LOW

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]
    low = low_raw * TOF_CONVERSION_FACTORS["TOF L"]

    high_saturation_mask = high_raw >= SATURATION_EFFECTIVE_LIMIT
    medium_saturation_mask = medium_raw >= SATURATION_EFFECTIVE_LIMIT

    combined = combine_waveform_channels(times, high, medium, low)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    expected[high_saturation_mask] = _normalize_expected(medium, "TOF M")[high_saturation_mask]
    expected[medium_saturation_mask] = _normalize_expected(low, "TOF L")[medium_saturation_mask]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_uses_low_when_only_medium_and_low_selected():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 30.0 * np.exp(-((times - 7.5) / 0.35) ** 2)

    medium_baseline = 63.0
    low_baseline = -4.5

    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM
    low_raw = low_baseline + physical_signal * GAIN_LOW

    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]
    low = low_raw * TOF_CONVERSION_FACTORS["TOF L"]

    medium_saturation_mask = medium_raw >= SATURATION_EFFECTIVE_LIMIT

    combined = combine_waveform_channels(
        times,
        None,
        medium,
        low,
        enabled_channels=("TOF M", "TOF L"),
    )
    assert combined is not None

    expected = _normalize_expected(medium, "TOF M")
    expected[medium_saturation_mask] = _normalize_expected(low, "TOF L")[medium_saturation_mask]
    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_bridges_short_high_gain_saturation_dips():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 0.45 * np.exp(-0.5 * ((times - 4.0) / 0.18) ** 2)

    high_baseline = 900.0
    medium_baseline = 72.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM

    saturation_region = (times >= 3.85) & (times <= 4.25)
    dip_region = (times >= 4.00) & (times <= 4.28)

    high_raw[saturation_region] = 1020.0
    high_raw[dip_region] = 917.0

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]

    combined = combine_waveform_channels(times, high, medium, None)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    high_saturation_mask = high_raw >= SATURATION_EFFECTIVE_LIMIT
    expected[high_saturation_mask] = _normalize_expected(medium, "TOF M")[high_saturation_mask]
    expected[dip_region] = _normalize_expected(medium, "TOF M")[dip_region]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_respects_gap_setting():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 0.45 * np.exp(-0.5 * ((times - 4.0) / 0.18) ** 2)

    high_baseline = 900.0
    medium_baseline = 72.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM

    saturation_region = (times >= 3.85) & (times <= 4.25)
    dip_region = (times >= 4.00) & (times <= 4.28)

    high_raw[saturation_region] = 1020.0
    high_raw[dip_region] = 917.0

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]

    combined = combine_waveform_channels(times, high, medium, None, max_saturation_gap_us=0.0)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    high_saturation_mask = high_raw >= SATURATION_EFFECTIVE_LIMIT
    expected[high_saturation_mask] = _normalize_expected(medium, "TOF M")[high_saturation_mask]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_keeps_low_when_mid_gain_rings_below_hard_limit():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 0.05 * np.exp(-0.5 * ((times - 4.0) / 0.20) ** 2)

    high_baseline = 900.0
    medium_baseline = 610.0
    low_baseline = 48.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM
    low_raw = low_baseline + physical_signal * GAIN_LOW

    ring_region = (times >= 3.90) & (times <= 4.30)
    dip_region = (times >= 4.02) & (times <= 4.28)

    high_raw[ring_region] = 1020.0
    medium_raw[ring_region] = 1000.0
    medium_raw[dip_region] = 917.0

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]
    low = low_raw * TOF_CONVERSION_FACTORS["TOF L"]

    combined = combine_waveform_channels(times, high, medium, low)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    high_saturation_mask = high_raw >= SATURATION_EFFECTIVE_LIMIT
    medium_saturation_mask = medium_raw >= SATURATION_EFFECTIVE_LIMIT
    expected[high_saturation_mask] = _normalize_expected(medium, "TOF M")[high_saturation_mask]
    expected[medium_saturation_mask] = _normalize_expected(low, "TOF L")[medium_saturation_mask]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_respects_release_fraction_setting():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 0.05 * np.exp(-0.5 * ((times - 4.0) / 0.20) ** 2)

    high_baseline = 900.0
    medium_baseline = 610.0
    low_baseline = 48.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM
    low_raw = low_baseline + physical_signal * GAIN_LOW

    ring_region = (times >= 3.90) & (times <= 4.30)
    high_raw[ring_region] = 1020.0
    medium_raw[ring_region] = np.linspace(800.0, 990.0, int(np.count_nonzero(ring_region)))

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]
    low = low_raw * TOF_CONVERSION_FACTORS["TOF L"]

    combined = combine_waveform_channels(
        times,
        high,
        medium,
        low,
        saturation_release_fraction=0.98,
    )
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    expected[ring_region] = _normalize_expected(medium, "TOF M")[ring_region]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_demotes_ringy_mid_gain_below_ceiling():
    times = np.linspace(0.0, 31.5, 2048)
    physical_signal = 0.10 * np.exp(-0.5 * ((times - 4.0) / 0.20) ** 2)

    high_baseline = 900.0
    medium_baseline = 74.0
    low_baseline = 48.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + physical_signal * GAIN_MEDIUM
    low_raw = low_baseline + physical_signal * GAIN_LOW

    ring_region = (times >= 3.90) & (times <= 4.30)
    high_raw[ring_region] = 1020.0
    medium_raw[ring_region] = 900.0

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]
    low = low_raw * TOF_CONVERSION_FACTORS["TOF L"]

    combined = combine_waveform_channels(
        times,
        high,
        medium,
        low,
        saturation_release_fraction=1.0,
    )
    assert combined is not None

    expected = _normalize_expected(low, "TOF L")

    assert np.allclose(combined[ring_region], expected[ring_region], rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_prefers_high_when_unsaturated():
    times = np.linspace(0.0, 31.5, 4096)
    physical_signal = 0.2 * np.exp(-0.5 * ((times - 6.0) / 0.55) ** 2)

    high_baseline = 620.0
    medium_baseline = 74.0

    high_raw = high_baseline + physical_signal * GAIN_HIGH
    medium_raw = medium_baseline + 1.12 * physical_signal * GAIN_MEDIUM

    high = high_raw * TOF_CONVERSION_FACTORS["TOF H"]
    medium = medium_raw * TOF_CONVERSION_FACTORS["TOF M"]

    combined = combine_waveform_channels(times, high, medium, None)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    high_saturation_mask = high_raw >= SATURATION_EFFECTIVE_LIMIT
    expected[high_saturation_mask] = _normalize_expected(medium, "TOF M")[high_saturation_mask]

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_respects_manual_selection():
    times = np.linspace(0.0, 31.5, 1024)
    physical_signal = 0.38 * np.exp(-0.5 * ((times - 5.2) / 0.6) ** 2)

    medium_baseline = 66.0
    medium = medium_baseline + physical_signal * GAIN_MEDIUM

    combined = combine_waveform_channels(
        times,
        None,
        medium,
        None,
        enabled_channels=("TOF M",),
    )

    assert combined is not None

    expected = _normalize_expected(medium, "TOF M")

    assert np.allclose(combined, expected, rtol=1e-6, atol=1e-6)


def test_combine_waveform_channels_baseline_with_descending_time_axis():
    times = np.linspace(10.0, -5.0, 800)
    baseline_level = 123.0
    slope = 0.02

    high = baseline_level + slope * np.arange(times.size)

    combined = combine_waveform_channels(times, high, None, None)
    assert combined is not None

    expected = _normalize_expected(high, "TOF H")
    np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-6)
