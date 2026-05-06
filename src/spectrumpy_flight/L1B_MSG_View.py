"""Qt viewer for browsing directory collections of ``imap_idex_l1b_msg`` CDF files."""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cdflib  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "L1B MSG viewer requires the optional 'cdflib' dependency."
    ) from exc

_QT_API = None
try:  # pragma: no cover - import guard
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    _QT_API = "PySide6"
except Exception:  # pragma: no cover - import guard
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    _QT_API = "PyQt6"

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter, DateFormatter
from matplotlib.figure import Figure


FILE_GLOB = "imap_idex_l1b_msg_*.cdf"
DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "CDF" / "l1b_msg"
SCIENCE_FILE_GLOB = "imap_idex_l1b_sci-1week_*.cdf"
DEFAULT_SCIENCE_DIRECTORY = Path(__file__).resolve().parent / "CDF" / "l1b"


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _stringify_value(value.item())
        return np.array2string(value, threshold=12)
    if isinstance(value, np.generic):
        return _stringify_value(value.item())
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        preview = ", ".join(_stringify_value(item) for item in value[:6])
        if len(value) > 6:
            preview += ", ..."
        return f"[{preview}]"
    if isinstance(value, dict):
        items = list(value.items())
        preview = ", ".join(f"{key}={_stringify_value(item)}" for key, item in items[:6])
        if len(items) > 6:
            preview += ", ..."
        return "{" + preview + "}"
    return str(value)


def _shape_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return "Scalar"
    if arr.size == 0:
        return "Empty"
    return "x".join(str(dim) for dim in arr.shape)


def _normalize_attr_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): item for key, item in dict(value).items()}
    except Exception:
        return {}


def _epoch_to_iso(epoch_value: Any) -> str:
    seconds = _epoch_to_seconds(epoch_value)
    if seconds is not None:
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except Exception:
            pass
    arr = np.asarray(epoch_value)
    if arr.size == 0:
        return ""
    return _stringify_value(arr.reshape(-1)[0])


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, np.datetime64):
        try:
            microseconds = int(value.astype("datetime64[us]").astype(np.int64))
            return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=microseconds)
        except Exception:
            return None
    if hasattr(value, "to_pydatetime"):
        try:
            converted = value.to_pydatetime()
            return _coerce_datetime(converted)
        except Exception:
            return None
    if isinstance(value, str):
        cleaned = value.replace(" UTC", "").replace("Z", "+00:00")
        try:
            converted = datetime.fromisoformat(cleaned)
        except Exception:
            return None
        if converted.tzinfo is None:
            return converted.replace(tzinfo=timezone.utc)
        return converted.astimezone(timezone.utc)
    return None


def _epoch_to_datetime(epoch_value: Any) -> Optional[datetime]:
    seconds = _epoch_to_seconds(epoch_value)
    if seconds is not None:
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except Exception:
            pass
    arr = np.asarray(epoch_value)
    if arr.size == 0:
        return None
    return _coerce_datetime(arr.reshape(-1)[0])


def _epoch_to_seconds(epoch_value: Any) -> Optional[float]:
    arr = np.asarray(epoch_value)
    if arr.size == 0:
        return None
    try:
        converted = cdflib.cdfepoch.to_datetime(arr.reshape(-1)[:1])
    except Exception:
        converted = None
    if converted:
        try:
            dt = converted[0]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return float(datetime.timestamp(dt))
        except Exception:
            pass
    return _fallback_epoch_seconds(arr.reshape(-1)[0])


def _epoch_vector_to_seconds(epoch_values: np.ndarray) -> List[Optional[float]]:
    arr = np.asarray(epoch_values).reshape(-1)
    if arr.size == 0:
        return []
    try:
        converted = cdflib.cdfepoch.to_datetime(arr)
    except Exception:
        converted = None
    if converted:
        seconds: List[Optional[float]] = []
        try:
            for dt in converted:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                seconds.append(float(datetime.timestamp(dt)))
            return seconds
        except Exception:
            pass
    return [_epoch_to_seconds(value) for value in arr]


def _epoch_vector_to_iso(epoch_values: np.ndarray) -> List[str]:
    seconds = _epoch_vector_to_seconds(epoch_values)
    if seconds:
        texts: List[str] = []
        for value in seconds:
            if value is None:
                texts.append("")
                continue
            try:
                texts.append(datetime.fromtimestamp(value, tz=timezone.utc).isoformat())
            except Exception:
                texts.append(str(value))
        return texts
    arr = np.asarray(epoch_values).reshape(-1)
    return [_stringify_value(value) for value in arr]


def _read_scalar(cdf: Any, varname: str) -> Any:
    try:
        data = np.asarray(cdf.varget(varname))
    except Exception:
        return None
    if data.size == 0:
        return None
    flat = data.reshape(-1)
    if flat.size == 1:
        return flat[0]
    return data


def _read_vector(cdf: Any, varname: str) -> np.ndarray:
    try:
        data = np.asarray(cdf.varget(varname))
    except Exception:
        return np.asarray([])
    return data.reshape(-1)


def _fallback_epoch_seconds(value: Any) -> Optional[float]:
    dt = _coerce_datetime(value)
    if dt is not None:
        return dt.timestamp()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return float(numeric)


@dataclass(frozen=True)
class L1BMsgTableRow:
    epoch: str
    epoch_dt: Optional[datetime]
    pulser_on: str
    science_on: str
    filename: str
    path: Path
    row_index: int


@dataclass(frozen=True)
class L1BMsgTranslatedRow:
    epoch: str
    epoch_dt: Optional[datetime]
    event: str
    state: str
    filename: str
    path: Path
    row_index: int


@dataclass(frozen=True)
class L1BMsgFlagRow:
    epoch: str
    epoch_dt: Optional[datetime]
    category: str
    condition: str
    filename: str
    path: Path
    row_index: int


@dataclass(frozen=True)
class L1BMsgIntervalRow:
    start: str
    start_dt: Optional[datetime]
    end: str
    end_dt: Optional[datetime]
    category: str
    filename: str
    path: Path
    row_index: int


@dataclass(frozen=True)
class L1BScienceEventRow:
    epoch: str
    epoch_dt: Optional[datetime]
    filename: str
    path: Path
    row_index: int


@dataclass
class L1BMsgRecord:
    path: Path
    filename: str
    epoch_raw: Any
    epoch_utc: str
    pulser_on_raw: Any
    science_on_raw: Any
    epoch_values: np.ndarray
    epoch_texts: List[str]
    pulser_values: np.ndarray
    science_values: np.ndarray
    global_attributes: Dict[str, Any]
    variable_rows: List[Tuple[str, Any, str]]
    variable_attributes: List[Tuple[str, str, Any]]

    def summary_lines(self) -> List[str]:
        lines = [
            f"File: {self.filename}",
            f"Path: {self.path}",
            f"Rows: {len(self.epoch_values)}",
            f"Epoch shape: {_shape_text(self.epoch_values)}",
            f"Pulser shape: {_shape_text(self.pulser_values)}",
            f"Science shape: {_shape_text(self.science_values)}",
        ]
        for key in (
            "Logical_file_id",
            "Logical_source",
            "Data_version",
            "Generation_date",
            "Generated_by",
            "Start_date",
        ):
            value = self.global_attributes.get(key)
            if value is not None:
                lines.append(f"{key}: {_stringify_value(value)}")
        return lines


def _load_l1b_msg_record(path: Path) -> L1BMsgRecord:
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    names: List[str] = []
    for key in ("zVariables", "rVariables"):
        values = info.get(key, []) if isinstance(info, dict) else getattr(info, key, [])
        for name in values or []:
            names.append(str(name))

    global_attributes = _normalize_attr_mapping(cdf.globalattsget())
    epoch_raw = _read_scalar(cdf, "epoch")
    pulser_on_raw = _read_scalar(cdf, "pulser_on")
    science_on_raw = _read_scalar(cdf, "science_on")
    epoch_values = _read_vector(cdf, "epoch")
    epoch_texts = _epoch_vector_to_iso(epoch_values)
    pulser_values = _read_vector(cdf, "pulser_on")
    science_values = _read_vector(cdf, "science_on")

    variable_rows: List[Tuple[str, Any, str]] = []
    variable_attributes: List[Tuple[str, str, Any]] = []
    for name in sorted(set(names)):
        try:
            value = cdf.varget(name)
        except Exception:
            continue
        variable_rows.append((name, value, _shape_text(value)))
        attrs = _normalize_attr_mapping(cdf.varattsget(name))
        for attr_name, attr_value in sorted(attrs.items()):
            variable_attributes.append((name, attr_name, attr_value))

    return L1BMsgRecord(
        path=path,
        filename=path.name,
        epoch_raw=epoch_raw,
        epoch_utc=_epoch_to_iso(epoch_raw),
        pulser_on_raw=pulser_on_raw,
        science_on_raw=science_on_raw,
        epoch_values=epoch_values,
        epoch_texts=epoch_texts,
        pulser_values=pulser_values,
        science_values=science_values,
        global_attributes=global_attributes,
        variable_rows=variable_rows,
        variable_attributes=variable_attributes,
    )


class L1BMsgViewWindow(QMainWindow):
    """Interactive browser for collections of ``imap_idex_l1b_msg`` CDF files."""

    RAW_HEADERS = (
        "Epoch",
        "pulsar_on",
        "science_on",
    )

    SERIES_HEADERS = (
        "Epoch",
        "Event",
        "On/Off",
    )

    FLAG_HEADERS = (
        "Epoch",
        "Category",
        "Condition",
    )

    SCIENCE_MAX_ON_DURATION = timedelta(hours=48)
    PULSAR_GROUP_MAX_GAP = timedelta(minutes=5)
    PULSAR_EXPECTED_CYCLES = 5
    PULSAR_GROUP_MAX_SPACING = timedelta(days=9)

    @staticmethod
    def _time_sort_key(epoch_dt: Optional[datetime], epoch_text: str, filename: str, row_index: int) -> Tuple[Any, ...]:
        return (
            epoch_dt is None,
            epoch_dt or datetime.max.replace(tzinfo=timezone.utc),
            epoch_text,
            filename,
            row_index,
        )

    def __init__(self, directory: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._directory = Path(directory).expanduser() if directory else DEFAULT_DIRECTORY
        self._records: List[L1BMsgRecord] = []
        self._table_rows: List[Tuple[L1BMsgTableRow, L1BMsgRecord]] = []
        self._translated_rows: List[Tuple[L1BMsgTranslatedRow, L1BMsgRecord]] = []
        self._flag_rows: List[L1BMsgFlagRow] = []
        self._science_intervals: List[L1BMsgIntervalRow] = []
        self._pulsar_intervals: List[L1BMsgIntervalRow] = []
        self._science_event_rows: List[L1BScienceEventRow] = []

        self.setWindowTitle("IDEX L1B MSG Viewer")
        self.resize(1280, 820)

        self._directory_edit: QLineEdit
        self._left_tabs: QTabWidget
        self._raw_table: QTableWidget
        self._series_table: QTableWidget
        self._science_table: QTableWidget
        self._pulsar_table: QTableWidget
        self._flags_table: QTableWidget
        self._summary_text: QPlainTextEdit
        self._timeline_figure: Figure
        self._timeline_canvas: FigureCanvas
        self._global_attrs_table: QTableWidget
        self._variable_table: QTableWidget
        self._variable_attrs_table: QTableWidget

        self._build_ui()
        self.reload_directory()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        toolbar = QToolBar("L1B MSG Tools", self)
        toolbar.setMovable(False)
        browse_action = QAction("Browse Directory…", self)
        browse_action.triggered.connect(self._choose_directory)
        toolbar.addAction(browse_action)
        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self.reload_directory)
        toolbar.addAction(reload_action)
        export_action = QAction("Export Time Series CSV…", self)
        export_action.triggered.connect(self._export_series_csv)
        toolbar.addAction(export_action)
        self.addToolBar(toolbar)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:", central))
        self._directory_edit = QLineEdit(str(self._directory), central)
        self._directory_edit.returnPressed.connect(self.reload_directory)
        dir_row.addWidget(self._directory_edit, 1)
        browse_button = QPushButton("Browse…", central)
        browse_button.clicked.connect(self._choose_directory)
        dir_row.addWidget(browse_button)
        reload_button = QPushButton("Reload", central)
        reload_button.clicked.connect(self.reload_directory)
        dir_row.addWidget(reload_button)
        layout.addLayout(dir_row)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        self._left_tabs = QTabWidget(splitter)

        self._series_table = QTableWidget(self._left_tabs)
        self._series_table.setColumnCount(len(self.SERIES_HEADERS))
        self._series_table.setHorizontalHeaderLabels(list(self.SERIES_HEADERS))
        self._series_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._series_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._series_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._series_table.setAlternatingRowColors(True)
        self._series_table.verticalHeader().setVisible(False)
        self._series_table.horizontalHeader().setStretchLastSection(False)
        self._series_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._series_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._series_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._series_table.itemSelectionChanged.connect(self._on_translated_selection_changed)
        self._left_tabs.addTab(self._series_table, "Translated")

        self._science_table = QTableWidget(self._left_tabs)
        self._science_table.setColumnCount(len(self.SERIES_HEADERS))
        self._science_table.setHorizontalHeaderLabels(list(self.SERIES_HEADERS))
        self._science_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._science_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._science_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._science_table.setAlternatingRowColors(True)
        self._science_table.verticalHeader().setVisible(False)
        self._science_table.horizontalHeader().setStretchLastSection(False)
        self._science_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._science_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._science_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._science_table.itemSelectionChanged.connect(self._on_science_selection_changed)
        self._left_tabs.addTab(self._science_table, "Science")

        self._pulsar_table = QTableWidget(self._left_tabs)
        self._pulsar_table.setColumnCount(len(self.SERIES_HEADERS))
        self._pulsar_table.setHorizontalHeaderLabels(list(self.SERIES_HEADERS))
        self._pulsar_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._pulsar_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._pulsar_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._pulsar_table.setAlternatingRowColors(True)
        self._pulsar_table.verticalHeader().setVisible(False)
        self._pulsar_table.horizontalHeader().setStretchLastSection(False)
        self._pulsar_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._pulsar_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._pulsar_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._pulsar_table.itemSelectionChanged.connect(self._on_pulsar_selection_changed)
        self._left_tabs.addTab(self._pulsar_table, "Pulsar")

        self._flags_table = QTableWidget(self._left_tabs)
        self._flags_table.setColumnCount(len(self.FLAG_HEADERS))
        self._flags_table.setHorizontalHeaderLabels(list(self.FLAG_HEADERS))
        self._flags_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._flags_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._flags_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._flags_table.setAlternatingRowColors(True)
        self._flags_table.verticalHeader().setVisible(False)
        self._flags_table.horizontalHeader().setStretchLastSection(False)
        self._flags_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._flags_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._flags_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._left_tabs.addTab(self._flags_table, "Flags")

        self._raw_table = QTableWidget(self._left_tabs)
        self._raw_table.setColumnCount(len(self.RAW_HEADERS))
        self._raw_table.setHorizontalHeaderLabels(list(self.RAW_HEADERS))
        self._raw_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._raw_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._raw_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._raw_table.setAlternatingRowColors(True)
        self._raw_table.verticalHeader().setVisible(False)
        self._raw_table.horizontalHeader().setStretchLastSection(False)
        self._raw_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._raw_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._raw_table.itemSelectionChanged.connect(self._on_raw_selection_changed)
        self._left_tabs.addTab(self._raw_table, "Raw")

        tabs = QTabWidget(splitter)

        self._summary_text = QPlainTextEdit(tabs)
        self._summary_text.setReadOnly(True)
        tabs.addTab(self._summary_text, "Summary")

        timeline_tab = QWidget(tabs)
        timeline_layout = QVBoxLayout(timeline_tab)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(0)
        self._timeline_figure = Figure(figsize=(8, 3.5), constrained_layout=True)
        self._timeline_canvas = FigureCanvas(self._timeline_figure)
        timeline_toolbar = NavigationToolbar(self._timeline_canvas, timeline_tab)
        timeline_layout.addWidget(timeline_toolbar)
        timeline_layout.addWidget(self._timeline_canvas, 1)
        tabs.addTab(timeline_tab, "Timeline")

        self._global_attrs_table = QTableWidget(tabs)
        self._global_attrs_table.setColumnCount(2)
        self._global_attrs_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self._global_attrs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._global_attrs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._global_attrs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        tabs.addTab(self._global_attrs_table, "Global Attributes")

        self._variable_table = QTableWidget(tabs)
        self._variable_table.setColumnCount(3)
        self._variable_table.setHorizontalHeaderLabels(["Variable", "Value", "Shape"])
        self._variable_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._variable_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._variable_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._variable_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        tabs.addTab(self._variable_table, "Variables")

        self._variable_attrs_table = QTableWidget(tabs)
        self._variable_attrs_table.setColumnCount(3)
        self._variable_attrs_table.setHorizontalHeaderLabels(["Variable", "Attribute", "Value"])
        self._variable_attrs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._variable_attrs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._variable_attrs_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._variable_attrs_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        tabs.addTab(self._variable_attrs_table, "Variable Attributes")

        if hasattr(splitter, "setCollapsible"):
            splitter.setCollapsible(0, False)
            splitter.setCollapsible(1, False)
        splitter.setSizes([560, 720])

        self.setCentralWidget(central)

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with imap_idex_l1b_msg CDF Files",
            str(self._directory),
        )
        if not selected:
            return
        self._directory = Path(selected)
        self._directory_edit.setText(str(self._directory))
        self.reload_directory()

    def reload_directory(self) -> None:
        self._directory = Path(self._directory_edit.text().strip() or self._directory).expanduser()
        self._directory_edit.setText(str(self._directory))
        if not self._directory.exists():
            QMessageBox.warning(self, "Directory Not Found", f"Directory not found:\n{self._directory}")
            self._records = []
            self._table_rows = []
            self._translated_rows = []
            self._flag_rows = []
            self._science_intervals = []
            self._pulsar_intervals = []
            self._science_event_rows = []
            self._populate_series_table()
            self._populate_science_table()
            self._populate_pulsar_table()
            self._populate_flags_table()
            self._populate_raw_table()
            self._update_timeline_plot()
            self._clear_details()
            return

        records: List[L1BMsgRecord] = []
        failures: List[str] = []
        for path in sorted(self._directory.glob(FILE_GLOB)):
            try:
                records.append(_load_l1b_msg_record(path))
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")

        self._records = records
        self._rebuild_table_rows()
        self._rebuild_translated_rows()
        self._science_intervals = self._build_intervals("science")
        self._pulsar_intervals = self._build_intervals("pulsar")
        self._science_event_rows = self._load_science_event_rows()
        self._rebuild_flag_rows()
        self._populate_series_table()
        self._populate_science_table()
        self._populate_pulsar_table()
        self._populate_flags_table()
        self._populate_raw_table()
        self._update_timeline_plot()
        self._clear_details()
        if self._translated_rows:
            self._series_table.selectRow(0)
            self._show_translated_record(0)
        elif self._table_rows:
            self._raw_table.selectRow(0)
            self._show_raw_record(0)
        self._update_window_title()

        if failures:
            QMessageBox.warning(
                self,
                "Some Files Could Not Be Read",
                "The following files could not be loaded:\n\n" + "\n".join(failures),
            )

    def _update_window_title(self) -> None:
        file_count = len(self._records)
        event_count = len(self._translated_rows)
        raw_count = len(self._table_rows)
        self.setWindowTitle(
            f"IDEX L1B MSG Viewer — {self._directory} ({event_count} translated rows, {raw_count} raw rows from {file_count} file{'s' if file_count != 1 else ''})"
        )

    def _rebuild_table_rows(self) -> None:
        rows: List[Tuple[L1BMsgTableRow, L1BMsgRecord]] = []
        for record in self._records:
            row_count = min(len(record.epoch_values), len(record.pulser_values), len(record.science_values))
            for row_index in range(row_count):
                rows.append(
                    (
                        L1BMsgTableRow(
                            epoch=record.epoch_texts[row_index] if row_index < len(record.epoch_texts) else "",
                            epoch_dt=_epoch_to_datetime(record.epoch_values[row_index]),
                            pulser_on=_stringify_value(record.pulser_values[row_index]),
                            science_on=_stringify_value(record.science_values[row_index]),
                            filename=record.filename,
                            path=record.path,
                            row_index=row_index,
                        ),
                        record,
                    )
                )
        rows.sort(
            key=lambda item: self._time_sort_key(
                item[0].epoch_dt,
                item[0].epoch,
                item[0].filename,
                item[0].row_index,
            )
        )
        self._table_rows = rows

    def _rebuild_translated_rows(self) -> None:
        rows: List[Tuple[L1BMsgTranslatedRow, L1BMsgRecord]] = []
        for raw_row, record in self._table_rows:
            if raw_row.pulser_on in {"0", "1"}:
                rows.append(
                    (
                        L1BMsgTranslatedRow(
                            epoch=raw_row.epoch,
                            epoch_dt=raw_row.epoch_dt,
                            event="pulsar",
                            state="On" if raw_row.pulser_on == "1" else "Off",
                            filename=raw_row.filename,
                            path=raw_row.path,
                            row_index=raw_row.row_index,
                        ),
                        record,
                    )
                )
            if raw_row.science_on in {"0", "1"}:
                rows.append(
                    (
                        L1BMsgTranslatedRow(
                            epoch=raw_row.epoch,
                            epoch_dt=raw_row.epoch_dt,
                            event="science",
                            state="On" if raw_row.science_on == "1" else "Off",
                            filename=raw_row.filename,
                            path=raw_row.path,
                            row_index=raw_row.row_index,
                        ),
                        record,
                    )
                )
        rows.sort(
            key=lambda item: self._time_sort_key(
                item[0].epoch_dt,
                item[0].epoch,
                item[0].filename,
                item[0].row_index,
            )
            + (item[0].event,)
        )
        self._translated_rows = rows

    def _rebuild_flag_rows(self) -> None:
        flags: List[L1BMsgFlagRow] = []
        flags.extend(self._build_duplicate_state_flags("science"))
        flags.extend(self._build_duplicate_state_flags("pulsar"))
        flags.extend(self._build_science_duration_flags())
        flags.extend(self._build_pulsar_sequence_flags())
        flags.extend(self._build_science_event_outside_science_flags())
        flags.extend(self._build_pulsar_event_count_flags())
        flags.sort(
            key=lambda flag: self._time_sort_key(
                flag.epoch_dt,
                flag.epoch,
                flag.filename,
                flag.row_index,
            )
            + (flag.category,)
        )
        self._flag_rows = flags

    def _translated_rows_for_event(self, event_name: str) -> List[Tuple[L1BMsgTranslatedRow, L1BMsgRecord]]:
        return [(entry, record) for entry, record in self._translated_rows if entry.event == event_name]

    def _build_intervals(self, event_name: str) -> List[L1BMsgIntervalRow]:
        intervals: List[L1BMsgIntervalRow] = []
        current_on: Optional[L1BMsgTranslatedRow] = None
        event_rows = self._translated_rows_for_event(event_name)
        for entry, _record in event_rows:
            if entry.state == "On":
                if current_on is None:
                    current_on = entry
                continue
            if entry.state == "Off" and current_on is not None:
                intervals.append(
                    L1BMsgIntervalRow(
                        start=current_on.epoch,
                        start_dt=current_on.epoch_dt,
                        end=entry.epoch,
                        end_dt=entry.epoch_dt,
                        category=event_name,
                        filename=current_on.filename,
                        path=current_on.path,
                        row_index=current_on.row_index,
                    )
                )
                current_on = None
        if current_on is not None and event_rows:
            last_entry = event_rows[-1][0]
            intervals.append(
                L1BMsgIntervalRow(
                    start=current_on.epoch,
                    start_dt=current_on.epoch_dt,
                    end=last_entry.epoch,
                    end_dt=last_entry.epoch_dt,
                    category=event_name,
                    filename=current_on.filename,
                    path=current_on.path,
                    row_index=current_on.row_index,
                )
            )
        return intervals

    def _load_science_event_rows(self) -> List[L1BScienceEventRow]:
        rows: List[L1BScienceEventRow] = []
        for path in sorted(DEFAULT_SCIENCE_DIRECTORY.glob(SCIENCE_FILE_GLOB)):
            try:
                cdf = cdflib.CDF(str(path))
                epoch_values = _read_vector(cdf, "epoch")
                epoch_texts = _epoch_vector_to_iso(epoch_values)
            except Exception:
                continue
            for row_index, epoch_value in enumerate(epoch_values):
                rows.append(
                    L1BScienceEventRow(
                        epoch=epoch_texts[row_index] if row_index < len(epoch_texts) else _stringify_value(epoch_value),
                        epoch_dt=_epoch_to_datetime(epoch_value),
                        filename=path.name,
                        path=path,
                        row_index=row_index,
                    )
                )
        rows.sort(
            key=lambda entry: self._time_sort_key(
                entry.epoch_dt,
                entry.epoch,
                entry.filename,
                entry.row_index,
            )
        )
        return rows

    def _build_duplicate_state_flags(self, event_name: str) -> List[L1BMsgFlagRow]:
        flags: List[L1BMsgFlagRow] = []
        event_rows = self._translated_rows_for_event(event_name)
        previous: Optional[L1BMsgTranslatedRow] = None
        for entry, _record in event_rows:
            if previous is not None and previous.state == entry.state:
                flags.append(
                    L1BMsgFlagRow(
                        epoch=entry.epoch,
                        epoch_dt=entry.epoch_dt,
                        category=event_name,
                        condition=(
                            f"Back-to-back {event_name} {entry.state.lower()} states "
                            f"after {previous.epoch or 'an earlier row'}."
                        ),
                        filename=entry.filename,
                        path=entry.path,
                        row_index=entry.row_index,
                    )
                )
            previous = entry
        return flags

    def _build_science_duration_flags(self) -> List[L1BMsgFlagRow]:
        flags: List[L1BMsgFlagRow] = []
        current_on: Optional[L1BMsgTranslatedRow] = None
        science_rows = self._translated_rows_for_event("science")
        for entry, _record in science_rows:
            if entry.state == "On":
                if current_on is None:
                    current_on = entry
                continue
            if entry.state == "Off" and current_on is not None and current_on.epoch_dt and entry.epoch_dt:
                duration = entry.epoch_dt - current_on.epoch_dt
                if duration > self.SCIENCE_MAX_ON_DURATION:
                    flags.append(
                        L1BMsgFlagRow(
                            epoch=current_on.epoch,
                            epoch_dt=current_on.epoch_dt,
                            category="science",
                            condition=(
                                "Science stayed on for "
                                f"{self._format_duration(duration)} before turning off at {entry.epoch}."
                            ),
                            filename=current_on.filename,
                            path=current_on.path,
                            row_index=current_on.row_index,
                        )
                    )
                current_on = None
        if current_on is not None and current_on.epoch_dt and science_rows:
            last_entry = science_rows[-1][0]
            if last_entry.epoch_dt:
                duration = last_entry.epoch_dt - current_on.epoch_dt
                if duration > self.SCIENCE_MAX_ON_DURATION:
                    flags.append(
                        L1BMsgFlagRow(
                            epoch=current_on.epoch,
                            epoch_dt=current_on.epoch_dt,
                            category="science",
                            condition=(
                                "Science stayed on for "
                                f"{self._format_duration(duration)} through the end of available data."
                            ),
                            filename=current_on.filename,
                            path=current_on.path,
                            row_index=current_on.row_index,
                        )
                    )
        return flags

    def _build_pulsar_sequence_flags(self) -> List[L1BMsgFlagRow]:
        flags: List[L1BMsgFlagRow] = []
        pulsar_rows = self._translated_rows_for_event("pulsar")
        groups: List[List[L1BMsgTranslatedRow]] = []
        current_group: List[L1BMsgTranslatedRow] = []
        for entry, _record in pulsar_rows:
            if not current_group:
                current_group = [entry]
                continue
            previous = current_group[-1]
            if previous.epoch_dt and entry.epoch_dt and entry.epoch_dt - previous.epoch_dt <= self.PULSAR_GROUP_MAX_GAP:
                current_group.append(entry)
            else:
                groups.append(current_group)
                current_group = [entry]
        if current_group:
            groups.append(current_group)

        for group_index, group in enumerate(groups):
            cycle_count = self._count_pulsar_cycles(group)
            if cycle_count != self.PULSAR_EXPECTED_CYCLES:
                first = group[0]
                flags.append(
                    L1BMsgFlagRow(
                        epoch=first.epoch,
                        epoch_dt=first.epoch_dt,
                        category="pulsar",
                        condition=(
                            f"Pulsar sequence contains {cycle_count} on/off cycles; "
                            f"expected {self.PULSAR_EXPECTED_CYCLES}."
                        ),
                        filename=first.filename,
                        path=first.path,
                        row_index=first.row_index,
                    )
                )
            if group_index == 0:
                continue
            previous_group = groups[group_index - 1]
            previous_end = previous_group[-1]
            current_start = group[0]
            if previous_end.epoch_dt and current_start.epoch_dt:
                spacing = current_start.epoch_dt - previous_end.epoch_dt
                if spacing > self.PULSAR_GROUP_MAX_SPACING:
                    flags.append(
                        L1BMsgFlagRow(
                            epoch=current_start.epoch,
                            epoch_dt=current_start.epoch_dt,
                            category="pulsar",
                            condition=(
                                "Pulsar sequence gap was "
                                f"{self._format_duration(spacing)}; expected roughly 8 days and no more than 9 days."
                            ),
                            filename=current_start.filename,
                            path=current_start.path,
                            row_index=current_start.row_index,
                        )
                    )
        return flags

    def _build_science_event_outside_science_flags(self) -> List[L1BMsgFlagRow]:
        flags: List[L1BMsgFlagRow] = []
        for event_row in self._science_event_rows:
            if event_row.epoch_dt is None:
                continue
            if self._time_in_intervals(event_row.epoch_dt, self._science_intervals):
                continue
            flags.append(
                L1BMsgFlagRow(
                    epoch=event_row.epoch,
                    epoch_dt=event_row.epoch_dt,
                    category="science",
                    condition="Science event occurred while science was off.",
                    filename=event_row.filename,
                    path=event_row.path,
                    row_index=event_row.row_index,
                )
            )
        return flags

    def _build_pulsar_event_count_flags(self) -> List[L1BMsgFlagRow]:
        flags: List[L1BMsgFlagRow] = []
        for interval in self._pulsar_intervals:
            if interval.start_dt is None or interval.end_dt is None or interval.end_dt < interval.start_dt:
                continue
            event_count = 0
            for event_row in self._science_event_rows:
                if event_row.epoch_dt is None:
                    continue
                if interval.start_dt <= event_row.epoch_dt <= interval.end_dt:
                    event_count += 1
            if event_count == 1:
                continue
            flags.append(
                L1BMsgFlagRow(
                    epoch=interval.start,
                    epoch_dt=interval.start_dt,
                    category="pulsar",
                    condition=(
                        f"Pulsar on period from {interval.start} to {interval.end} "
                        f"contained {event_count} science events; expected exactly 1."
                    ),
                    filename=interval.filename,
                    path=interval.path,
                    row_index=interval.row_index,
                )
            )
        return flags

    @staticmethod
    def _time_in_intervals(moment: datetime, intervals: Sequence[L1BMsgIntervalRow]) -> bool:
        for interval in intervals:
            if interval.start_dt is None or interval.end_dt is None:
                continue
            if interval.start_dt <= moment <= interval.end_dt:
                return True
        return False

    @staticmethod
    def _count_pulsar_cycles(group: Sequence[L1BMsgTranslatedRow]) -> int:
        cycle_count = 0
        for current, following in zip(group, group[1:]):
            if current.state == "On" and following.state == "Off":
                cycle_count += 1
        return cycle_count

    @staticmethod
    def _format_duration(duration: timedelta) -> str:
        total_seconds = int(duration.total_seconds())
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts: List[str] = []
        if days:
            parts.append(f"{days}d")
        if hours or days:
            parts.append(f"{hours}h")
        if minutes or hours or days:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")
        return " ".join(parts)

    def _populate_series_table(self) -> None:
        self._series_table.setRowCount(len(self._translated_rows))
        for row_index, (entry, _record) in enumerate(self._translated_rows):
            values = [
                entry.epoch,
                entry.event,
                entry.state,
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index != 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._series_table.setItem(row_index, col_index, item)
        self._series_table.resizeRowsToContents()

    def _populate_raw_table(self) -> None:
        self._raw_table.setRowCount(len(self._table_rows))
        for row_index, (entry, _record) in enumerate(self._table_rows):
            values = [
                entry.epoch,
                entry.pulser_on,
                entry.science_on,
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index != 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._raw_table.setItem(row_index, col_index, item)
        self._raw_table.resizeRowsToContents()

    def _populate_filtered_table(self, table: QTableWidget, event_name: str) -> None:
        filtered = [(entry, record) for entry, record in self._translated_rows if entry.event == event_name]
        table.setRowCount(len(filtered))
        for row_index, (entry, _record) in enumerate(filtered):
            values = [entry.epoch, entry.event, entry.state]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index != 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, col_index, item)
                if col_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, entry.row_index)
                    item.setData(Qt.ItemDataRole.UserRole + 1, entry.filename)
        table.resizeRowsToContents()

    def _populate_science_table(self) -> None:
        self._populate_filtered_table(self._science_table, "science")

    def _populate_pulsar_table(self) -> None:
        self._populate_filtered_table(self._pulsar_table, "pulsar")

    def _populate_flags_table(self) -> None:
        self._flags_table.setRowCount(len(self._flag_rows))
        for row_index, entry in enumerate(self._flag_rows):
            values = [entry.epoch, entry.category, entry.condition]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index == 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._flags_table.setItem(row_index, col_index, item)
        self._flags_table.resizeRowsToContents()

    def _update_timeline_plot(self) -> None:
        self._timeline_figure.clear()
        ax = self._timeline_figure.add_subplot(111)

        if not self._science_intervals and not self._pulsar_intervals and not self._science_event_rows:
            ax.text(
                0.5,
                0.5,
                "No timeline data available.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            self._timeline_canvas.draw_idle()
            return

        plotted_labels: set[str] = set()
        for interval in self._science_intervals:
            if interval.start_dt is None or interval.end_dt is None or interval.end_dt < interval.start_dt:
                continue
            label = "Science On"
            ax.axvspan(
                interval.start_dt,
                interval.end_dt,
                ymin=0.0,
                ymax=1.0,
                facecolor="#d0d0d0",
                edgecolor="none",
                alpha=0.75,
                zorder=1,
                label=label if label not in plotted_labels else None,
            )
            plotted_labels.add(label)

        for interval in self._pulsar_intervals:
            if interval.start_dt is None or interval.end_dt is None or interval.end_dt < interval.start_dt:
                continue
            label = "Pulsar On"
            ax.axvspan(
                interval.start_dt,
                interval.end_dt,
                ymin=0.0,
                ymax=1.0,
                facecolor="#007bff",
                edgecolor="#0057b8",
                linewidth=0.8,
                alpha=0.9,
                zorder=3,
                label=label if label not in plotted_labels else None,
            )
            plotted_labels.add(label)

        for event_row in self._science_event_rows:
            if event_row.epoch_dt is None:
                continue
            label = "Science Events"
            ax.axvline(
                event_row.epoch_dt,
                ymin=0.0,
                ymax=1.0,
                color="#1f1f1f",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                zorder=5,
                label=label if label not in plotted_labels else None,
            )
            plotted_labels.add(label)

        all_times = [
            interval.start_dt
            for interval in self._science_intervals + self._pulsar_intervals
            if interval.start_dt is not None
        ]
        all_times.extend(
            interval.end_dt
            for interval in self._science_intervals + self._pulsar_intervals
            if interval.end_dt is not None
        )
        all_times.extend(event.epoch_dt for event in self._science_event_rows if event.epoch_dt is not None)

        ax.set_title("Spacecraft State and L1B Science Event Timeline")
        ax.set_yticks([])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Time (UTC)")
        ax.set_axisbelow(False)

        if all_times:
            locator = AutoDateLocator()
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
            ax.fmt_xdata = DateFormatter("%Y-%m-%d %H:%M:%S.%f")
            ax.set_xlim(min(all_times), max(all_times))

        if plotted_labels:
            ax.legend(loc="upper left")

        self._timeline_canvas.draw_idle()

    def _clear_details(self) -> None:
        self._summary_text.clear()
        self._global_attrs_table.setRowCount(0)
        self._variable_table.setRowCount(0)
        self._variable_attrs_table.setRowCount(0)

    def _on_translated_selection_changed(self) -> None:
        ranges = self._series_table.selectedRanges()
        if not ranges:
            return
        self._show_translated_record(ranges[0].topRow())

    def _on_raw_selection_changed(self) -> None:
        ranges = self._raw_table.selectedRanges()
        if not ranges:
            return
        self._show_raw_record(ranges[0].topRow())

    def _on_science_selection_changed(self) -> None:
        self._show_filtered_selection(self._science_table, "science")

    def _on_pulsar_selection_changed(self) -> None:
        self._show_filtered_selection(self._pulsar_table, "pulsar")

    def _show_filtered_selection(self, table: QTableWidget, event_name: str) -> None:
        ranges = table.selectedRanges()
        if not ranges:
            return
        row = ranges[0].topRow()
        filtered_indices = [
            index for index, (entry, _record) in enumerate(self._translated_rows) if entry.event == event_name
        ]
        if row < 0 or row >= len(filtered_indices):
            self._clear_details()
            return
        self._show_translated_record(filtered_indices[row])

    def _show_translated_record(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._translated_rows):
            self._clear_details()
            return
        entry, record = self._translated_rows[row_index]
        lines = [
            f"Selected Epoch: {entry.epoch or 'Unavailable'}",
            f"Selected Event: {entry.event}",
            f"Selected State: {entry.state}",
            f"Selected Source Row: {entry.row_index}",
            "",
        ]
        lines.extend(record.summary_lines())
        self._set_detail_tables(record)
        self._summary_text.setPlainText("\n".join(lines))

    def _show_raw_record(self, row_index: int) -> None:
        if row_index < 0 or row_index >= len(self._table_rows):
            self._clear_details()
            return
        entry, record = self._table_rows[row_index]
        lines = [
            f"Selected Epoch: {entry.epoch or 'Unavailable'}",
            f"Selected pulsar_on: {entry.pulser_on}",
            f"Selected science_on: {entry.science_on}",
            f"Selected Source Row: {entry.row_index}",
            "",
        ]
        lines.extend(record.summary_lines())
        self._set_detail_tables(record)
        self._summary_text.setPlainText("\n".join(lines))

    def _set_detail_tables(self, record: L1BMsgRecord) -> None:
        self._populate_table(
            self._global_attrs_table,
            [(key, _stringify_value(value)) for key, value in sorted(record.global_attributes.items())],
        )
        self._populate_table(
            self._variable_table,
            [(name, _stringify_value(value), shape) for name, value, shape in record.variable_rows],
        )
        self._populate_table(
            self._variable_attrs_table,
            [
                (varname, attr_name, _stringify_value(attr_value))
                for varname, attr_name, attr_value in record.variable_attributes
            ],
        )

    @staticmethod
    def _populate_table(table: QTableWidget, rows: Iterable[Sequence[str]]) -> None:
        rows_list = list(rows)
        table.setRowCount(len(rows_list))
        for row_index, row_values in enumerate(rows_list):
            for col_index, value in enumerate(row_values):
                table.setItem(row_index, col_index, QTableWidgetItem(str(value)))
        table.resizeRowsToContents()

    def _export_series_csv(self) -> None:
        if not self._translated_rows:
            QMessageBox.information(self, "No Data", "No translated L1B MSG rows are currently loaded.")
            return
        default_path = self._directory / "imap_idex_l1b_msg_timeseries.csv"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Time Series CSV",
            str(default_path),
            "CSV Files (*.csv);;All Files (*)",
        )
        if not filename:
            return
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "epoch",
                    "event",
                    "on_off",
                ]
            )
            for entry, _record in self._translated_rows:
                writer.writerow(
                    [
                        entry.epoch,
                        entry.event,
                        entry.state,
                    ]
                )


def launch_l1b_msg_viewer(
    directory: Optional[str] = None,
    parent: Optional[QWidget] = None,
) -> L1BMsgViewWindow:
    window = L1BMsgViewWindow(directory=directory, parent=parent)
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(argv or sys.argv)
    directory = argv[1] if len(argv) > 1 else str(DEFAULT_DIRECTORY)
    app = QApplication.instance() or QApplication(argv)
    try:
        window = launch_l1b_msg_viewer(directory=directory)
    except Exception as exc:
        QMessageBox.critical(None, "Unable to Launch L1B MSG Viewer", str(exc))
        return 1
    if _QT_API == "PySide6":
        return app.exec()
    _ = window
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual GUI entry point
    raise SystemExit(main())
