"""Qt viewer for comparing matched IDEX CDF events across two product libraries."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Support execution both as part of the ``spectrumpy_flight`` package and when
# launched directly from the source tree.
if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    parent_dir = str(package_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from spectrumpy_flight import package_path

try:  # pragma: no cover - optional dependency
    import cdflib  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "IDEX CDF event compare viewer requires the optional 'cdflib' dependency."
    ) from exc

_QT_API = None
try:  # pragma: no cover - import guard
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction, QColor, QBrush
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QStatusBar,
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
    from PyQt6.QtGui import QAction, QColor, QBrush
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    _QT_API = "PyQt6"


LEVELS = ("l1a", "l1b", "l2a")
FLIGHT_PATTERNS = {
    "l1a": "imap_idex_l1a_sci-1week_*.cdf",
    "l1b": "imap_idex_l1b_sci-1week_*.cdf",
    "l2a": "imap_idex_l2a_sci-1week_*.cdf",
}
TEN_DAY_PATTERNS = {
    "l1a": "imap_idex_l1a_sci-10days_*.cdf",
    "l1b": "imap_idex_l1b_sci-10days_*.cdf",
    "l2a": "imap_idex_l2a_sci-10days_*.cdf",
}
DIFF_BRUSH = QBrush(QColor("#fff0c2"))
MISSING_BRUSH = QBrush(QColor("#ffd7d7"))


def _default_ten_day_root() -> Path:
    package_root = package_path().resolve()
    # <repo>/src/spectrumpy_flight -> <workspace>/spectrumpy
    for parent in package_root.parents:
        candidate = parent / "idex-10day"
        if candidate.exists():
            return candidate
    return package_root.parent / "idex-10day"


def _default_flight_root() -> Path:
    return package_path("CDF")


def _stringify_value(value: Any, *, preview: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        arr = value
        if arr.ndim == 0:
            return _stringify_value(arr.item(), preview=preview)
        threshold = 12 if preview else arr.size + 1
        return np.array2string(arr, threshold=threshold, max_line_width=120)
    if isinstance(value, np.generic):
        return _stringify_value(value.item(), preview=preview)
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        items = list(value.items())
        body = ", ".join(
            f"{key}={_stringify_value(item, preview=True)}" for key, item in items[:8]
        )
        if preview and len(items) > 8:
            body += ", ..."
        return "{" + body + "}"
    if isinstance(value, (list, tuple)):
        subset = value[:8] if preview else value
        body = ", ".join(_stringify_value(item, preview=True) for item in subset)
        if preview and len(value) > 8:
            body += ", ..."
        return "[" + body + "]"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _shape_text(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return "Scalar"
    if arr.size == 0:
        return "Empty"
    return "x".join(str(dim) for dim in arr.shape)


def _cdf_info_values(info: Any, key: str) -> List[str]:
    if isinstance(info, dict):
        values = info.get(key, [])
    else:
        values = getattr(info, key, [])
    return [str(value) for value in (values or [])]


def _cdf_zvariable_names(cdf: Any) -> List[str]:
    return _cdf_info_values(cdf.cdf_info(), "zVariables")


def _normalize_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): item for key, item in dict(value).items()}
    except Exception:
        return {}


def _epoch_to_datetime(epoch_value: Any) -> Optional[datetime]:
    arr = np.asarray(epoch_value).reshape(-1)
    if arr.size == 0:
        return None
    try:
        converted = cdflib.cdfepoch.to_datetime(arr[:1])
    except Exception:
        converted = None
    if converted:
        try:
            dt = converted[0]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return None
    return None


def _epoch_to_iso(epoch_value: Any) -> str:
    dt = _epoch_to_datetime(epoch_value)
    if dt is not None:
        return dt.isoformat()
    return _stringify_value(epoch_value)


@dataclass(frozen=True)
class EventLocation:
    path: Path
    row_index: int


@dataclass(frozen=True)
class EventEntry:
    epoch_tt2000: int
    epoch_iso: str
    flight_l1a_file: str
    ten_day_l1a_file: str


@dataclass
class VariableComparison:
    name: str
    left_value: Any
    right_value: Any
    left_shape: str
    right_shape: str
    diff_summary: str
    different: bool
    left_preview: str
    right_preview: str


@dataclass
class AttributeComparison:
    owner: str
    key: str
    left_value: Any
    right_value: Any
    different: bool


class CDFCache:
    """Small cache around ``cdflib.CDF`` handles and variable names."""

    def __init__(self) -> None:
        self._cdf_by_path: Dict[Path, Any] = {}
        self._event_vars_by_path: Dict[Path, List[str]] = {}

    def open(self, path: Path) -> Any:
        if path not in self._cdf_by_path:
            self._cdf_by_path[path] = cdflib.CDF(str(path))
        return self._cdf_by_path[path]

    def event_variables(self, path: Path) -> List[str]:
        if path in self._event_vars_by_path:
            return self._event_vars_by_path[path]
        cdf = self.open(path)
        epoch_values = np.asarray(cdf.varget("epoch"))
        row_count = len(epoch_values)
        variables: List[str] = []
        for name in _cdf_zvariable_names(cdf):
            try:
                value = np.asarray(cdf.varget(name))
            except Exception:
                continue
            if value.ndim >= 1 and value.shape[0] == row_count:
                variables.append(str(name))
        self._event_vars_by_path[path] = variables
        return variables


class EventLibraryIndex:
    """Index matching event epochs across the two product libraries."""

    def __init__(self, flight_root: Path, ten_day_root: Path) -> None:
        self.flight_root = flight_root
        self.ten_day_root = ten_day_root
        self.cache = CDFCache()
        self.locations: Dict[Tuple[str, str], Dict[int, EventLocation]] = {}
        self.events: List[EventEntry] = []

    def rebuild(self) -> None:
        self.locations = {}
        for level in LEVELS:
            self.locations[("flight", level)] = self._index_level(
                self.flight_root / level,
                FLIGHT_PATTERNS[level],
            )
            self.locations[("10day", level)] = self._index_level(
                self.ten_day_root / level / "sci",
                TEN_DAY_PATTERNS[level],
            )

        common_epochs = set.intersection(
            *(set(mapping.keys()) for mapping in self.locations.values())
        )
        sorted_epochs = sorted(common_epochs)
        events: List[EventEntry] = []
        for epoch in sorted_epochs:
            flight_loc = self.locations[("flight", "l1a")][epoch]
            ten_day_loc = self.locations[("10day", "l1a")][epoch]
            events.append(
                EventEntry(
                    epoch_tt2000=epoch,
                    epoch_iso=_epoch_to_iso(epoch),
                    flight_l1a_file=flight_loc.path.name,
                    ten_day_l1a_file=ten_day_loc.path.name,
                )
            )
        self.events = events

    def _index_level(self, directory: Path, pattern: str) -> Dict[int, EventLocation]:
        index: Dict[int, EventLocation] = {}
        if not directory.exists():
            return index
        for path in sorted(directory.glob(pattern)):
            cdf = self.cache.open(path)
            try:
                epochs = np.asarray(cdf.varget("epoch")).astype(np.int64)
            except Exception:
                continue
            for row_index, epoch in enumerate(epochs.reshape(-1)):
                index[int(epoch)] = EventLocation(path=path, row_index=row_index)
        return index


class LevelCompareWidget(QWidget):
    """Display file, metadata, and event-row diffs for one data level."""

    def __init__(self, level: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.level = level
        self._row_comparisons: List[VariableComparison] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.summary_text = QPlainTextEdit(self)
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumBlockCount(5000)

        tabs = QTabWidget(self)
        layout.addWidget(tabs, 1)

        summary_tab = QWidget(self)
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.addWidget(self.summary_text)
        tabs.addTab(summary_tab, "Summary")

        fields_tab = QWidget(self)
        fields_layout = QVBoxLayout(fields_tab)
        fields_controls = QHBoxLayout()
        self.diffs_only_checkbox = QCheckBox("Show Diffs Only", fields_tab)
        self.diffs_only_checkbox.toggled.connect(self._populate_row_table)
        fields_controls.addWidget(self.diffs_only_checkbox)
        fields_controls.addStretch(1)
        fields_layout.addLayout(fields_controls)

        field_splitter = QSplitter(Qt.Orientation.Vertical, fields_tab)
        self.row_values_table = QTableWidget(field_splitter)
        self.row_values_table.setColumnCount(6)
        self.row_values_table.setHorizontalHeaderLabels(
            ["Variable", "Diff", "Flight Shape", "10-Day Shape", "Flight Preview", "10-Day Preview"]
        )
        self.row_values_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.row_values_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.row_values_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.row_values_table.verticalHeader().setVisible(False)
        self.row_values_table.horizontalHeader().setStretchLastSection(True)
        self.row_values_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.row_values_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.row_values_table.itemSelectionChanged.connect(self._on_row_selected)

        detail_widget = QWidget(field_splitter)
        detail_layout = QHBoxLayout(detail_widget)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        self.left_detail_text = QPlainTextEdit(detail_widget)
        self.left_detail_text.setReadOnly(True)
        self.right_detail_text = QPlainTextEdit(detail_widget)
        self.right_detail_text.setReadOnly(True)
        detail_layout.addWidget(self.left_detail_text, 1)
        detail_layout.addWidget(self.right_detail_text, 1)
        field_splitter.setStretchFactor(0, 3)
        field_splitter.setStretchFactor(1, 2)
        fields_layout.addWidget(field_splitter, 1)
        tabs.addTab(fields_tab, "Event Fields")

        self.global_attrs_table = QTableWidget(self)
        self.global_attrs_table.setColumnCount(3)
        self.global_attrs_table.setHorizontalHeaderLabels(["Attribute", "Flight", "10-Day"])
        self.global_attrs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.global_attrs_table.verticalHeader().setVisible(False)
        self.global_attrs_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.global_attrs_table, "Global Metadata")

        self.variable_attrs_table = QTableWidget(self)
        self.variable_attrs_table.setColumnCount(4)
        self.variable_attrs_table.setHorizontalHeaderLabels(
            ["Variable", "Attribute", "Flight", "10-Day"]
        )
        self.variable_attrs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.variable_attrs_table.verticalHeader().setVisible(False)
        self.variable_attrs_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self.variable_attrs_table, "Variable Metadata")

    def clear(self) -> None:
        self.summary_text.clear()
        self.row_values_table.setRowCount(0)
        self.global_attrs_table.setRowCount(0)
        self.variable_attrs_table.setRowCount(0)
        self.left_detail_text.clear()
        self.right_detail_text.clear()
        self._row_comparisons = []

    def populate(
        self,
        *,
        summary_lines: Sequence[str],
        row_comparisons: Sequence[VariableComparison],
        global_comparisons: Sequence[AttributeComparison],
        variable_attr_comparisons: Sequence[AttributeComparison],
    ) -> None:
        self.summary_text.setPlainText("\n".join(summary_lines))
        self._row_comparisons = list(row_comparisons)
        self._populate_row_table()
        self._populate_attr_table(self.global_attrs_table, global_comparisons, include_owner=False)
        self._populate_attr_table(
            self.variable_attrs_table,
            variable_attr_comparisons,
            include_owner=True,
        )
        if self.row_values_table.rowCount():
            self.row_values_table.selectRow(0)
            self._on_row_selected()
        else:
            self.left_detail_text.clear()
            self.right_detail_text.clear()

    def _populate_row_table(self) -> None:
        rows = (
            [row for row in self._row_comparisons if row.different]
            if self.diffs_only_checkbox.isChecked()
            else self._row_comparisons
        )
        self.row_values_table.setRowCount(len(rows))
        self.row_values_table.setProperty("_visible_rows", rows)
        for row_index, row in enumerate(rows):
            values = [
                row.name,
                row.diff_summary,
                row.left_shape,
                row.right_shape,
                row.left_preview,
                row.right_preview,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if row.different:
                    item.setBackground(DIFF_BRUSH)
                self.row_values_table.setItem(row_index, column, item)
        self.row_values_table.resizeRowsToContents()

    def _populate_attr_table(
        self,
        table: QTableWidget,
        comparisons: Sequence[AttributeComparison],
        *,
        include_owner: bool,
    ) -> None:
        table.setRowCount(len(comparisons))
        for row_index, comparison in enumerate(comparisons):
            values = (
                [comparison.owner, comparison.key, _stringify_value(comparison.left_value, preview=True), _stringify_value(comparison.right_value, preview=True)]
                if include_owner
                else [comparison.key, _stringify_value(comparison.left_value, preview=True), _stringify_value(comparison.right_value, preview=True)]
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if comparison.different:
                    brush = MISSING_BRUSH if comparison.left_value is None or comparison.right_value is None else DIFF_BRUSH
                    item.setBackground(brush)
                table.setItem(row_index, column, item)
        table.resizeRowsToContents()

    def _on_row_selected(self) -> None:
        rows: Sequence[VariableComparison] = self.row_values_table.property("_visible_rows") or []
        current_row = self.row_values_table.currentRow()
        if current_row < 0 or current_row >= len(rows):
            self.left_detail_text.clear()
            self.right_detail_text.clear()
            return
        selected = rows[current_row]
        self.left_detail_text.setPlainText(
            f"Flight value\nVariable: {selected.name}\nShape: {selected.left_shape}\n\n"
            f"{_stringify_value(selected.left_value)}"
        )
        self.right_detail_text.setPlainText(
            f"10-Day value\nVariable: {selected.name}\nShape: {selected.right_shape}\n\n"
            f"{_stringify_value(selected.right_value)}"
        )


class IDEXCDFEventCompareWindow(QMainWindow):
    """Interactive browser for comparing shared CDF events between two libraries."""

    def __init__(
        self,
        *,
        flight_root: Optional[str] = None,
        ten_day_root: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._flight_root = Path(flight_root).expanduser() if flight_root else _default_flight_root()
        self._ten_day_root = Path(ten_day_root).expanduser() if ten_day_root else _default_ten_day_root()
        self._index = EventLibraryIndex(self._flight_root, self._ten_day_root)
        self._event_entries: List[EventEntry] = []
        self._level_widgets: Dict[str, LevelCompareWidget] = {}

        self.setWindowTitle("IDEX CDF Event Compare Viewer")
        self.resize(1600, 960)
        self._build_ui()
        self.reload_libraries()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        toolbar = QToolBar("Compare Tools", self)
        toolbar.setMovable(False)
        browse_flight_action = QAction("Browse Flight Root…", self)
        browse_flight_action.triggered.connect(self._choose_flight_root)
        toolbar.addAction(browse_flight_action)
        browse_ten_day_action = QAction("Browse 10-Day Root…", self)
        browse_ten_day_action.triggered.connect(self._choose_ten_day_root)
        toolbar.addAction(browse_ten_day_action)
        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self.reload_libraries)
        toolbar.addAction(reload_action)
        self.addToolBar(toolbar)
        self.setStatusBar(QStatusBar(self))

        roots_group = QGroupBox("Library Roots", central)
        roots_layout = QGridLayout(roots_group)
        roots_layout.addWidget(QLabel("Flight CDF Root:", roots_group), 0, 0)
        self.flight_root_edit = QLineEdit(str(self._flight_root), roots_group)
        self.flight_root_edit.returnPressed.connect(self.reload_libraries)
        roots_layout.addWidget(self.flight_root_edit, 0, 1)
        flight_browse_button = QPushButton("Browse…", roots_group)
        flight_browse_button.clicked.connect(self._choose_flight_root)
        roots_layout.addWidget(flight_browse_button, 0, 2)

        roots_layout.addWidget(QLabel("10-Day Root:", roots_group), 1, 0)
        self.ten_day_root_edit = QLineEdit(str(self._ten_day_root), roots_group)
        self.ten_day_root_edit.returnPressed.connect(self.reload_libraries)
        roots_layout.addWidget(self.ten_day_root_edit, 1, 1)
        ten_day_browse_button = QPushButton("Browse…", roots_group)
        ten_day_browse_button.clicked.connect(self._choose_ten_day_root)
        roots_layout.addWidget(ten_day_browse_button, 1, 2)
        layout.addWidget(roots_group)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Event Search:", central))
        self.search_edit = QLineEdit(central)
        self.search_edit.setPlaceholderText("Type an ISO timestamp fragment, file date, or TT2000 integer")
        self.search_edit.textChanged.connect(self._populate_event_list)
        search_row.addWidget(self.search_edit, 1)
        reload_button = QPushButton("Reload Index", central)
        reload_button.clicked.connect(self.reload_libraries)
        search_row.addWidget(reload_button)
        layout.addLayout(search_row)

        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)

        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.event_count_label = QLabel(left_panel)
        left_layout.addWidget(self.event_count_label)
        self.event_list = QListWidget(left_panel)
        self.event_list.currentRowChanged.connect(self._on_event_selected)
        left_layout.addWidget(self.event_list, 1)

        tabs = QTabWidget(splitter)
        for level in LEVELS:
            widget = LevelCompareWidget(level, tabs)
            self._level_widgets[level] = widget
            tabs.addTab(widget, level.upper())

        self.setCentralWidget(central)

    def _choose_flight_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select Flight CDF Root", str(self._flight_root)
        )
        if selected:
            self._flight_root = Path(selected)
            self.flight_root_edit.setText(str(self._flight_root))
            self.reload_libraries()

    def _choose_ten_day_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select 10-Day Root", str(self._ten_day_root)
        )
        if selected:
            self._ten_day_root = Path(selected)
            self.ten_day_root_edit.setText(str(self._ten_day_root))
            self.reload_libraries()

    def reload_libraries(self) -> None:
        self._flight_root = Path(self.flight_root_edit.text().strip() or self._flight_root).expanduser()
        self._ten_day_root = Path(self.ten_day_root_edit.text().strip() or self._ten_day_root).expanduser()
        self.flight_root_edit.setText(str(self._flight_root))
        self.ten_day_root_edit.setText(str(self._ten_day_root))
        self._index = EventLibraryIndex(self._flight_root, self._ten_day_root)
        try:
            self._index.rebuild()
        except Exception as exc:
            QMessageBox.critical(self, "Failed To Load Libraries", str(exc))
            self._event_entries = []
            self.event_list.clear()
            for widget in self._level_widgets.values():
                widget.clear()
            return

        self._event_entries = list(self._index.events)
        self._populate_event_list()
        self._update_status()
        if self.event_list.count():
            self.event_list.setCurrentRow(0)
            first_item = self.event_list.item(0)
            if first_item is not None:
                self._show_event(int(first_item.data(Qt.ItemDataRole.UserRole)))
        else:
            for widget in self._level_widgets.values():
                widget.clear()

    def _update_status(self) -> None:
        event_count = len(self._event_entries)
        if event_count:
            start_iso = self._event_entries[0].epoch_iso
            end_iso = self._event_entries[-1].epoch_iso
            text = f"{event_count} common events across L1A/L1B/L2A | {start_iso} to {end_iso}"
        else:
            text = "0 common events across L1A/L1B/L2A"
        self.event_count_label.setText(text)
        self.statusBar().showMessage(text)

    def _populate_event_list(self) -> None:
        query = self.search_edit.text().strip().lower()
        previous_epoch = None
        current_item = self.event_list.currentItem()
        if current_item is not None:
            previous_epoch = current_item.data(Qt.ItemDataRole.UserRole)
        self.event_list.clear()
        match_row = None
        for entry in self._event_entries:
            haystack = " | ".join(
                (
                    entry.epoch_iso,
                    str(entry.epoch_tt2000),
                    entry.flight_l1a_file,
                    entry.ten_day_l1a_file,
                )
            ).lower()
            if query and query not in haystack:
                continue
            label = f"{entry.epoch_iso} | flight:{entry.flight_l1a_file} | 10day:{entry.ten_day_l1a_file}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry.epoch_tt2000)
            self.event_list.addItem(item)
            if previous_epoch is not None and entry.epoch_tt2000 == previous_epoch:
                match_row = self.event_list.count() - 1
        if self.event_list.count():
            target_row = match_row if match_row is not None else 0
            self.event_list.setCurrentRow(target_row)

    def _on_event_selected(self, row_index: int) -> None:
        item = self.event_list.item(row_index)
        if item is None:
            for widget in self._level_widgets.values():
                widget.clear()
            return
        try:
            epoch = int(item.data(Qt.ItemDataRole.UserRole))
            self._show_event(epoch)
        except Exception as exc:
            message = f"Failed to load selected event: {exc}"
            self.statusBar().showMessage(message)
            for widget in self._level_widgets.values():
                widget.clear()
                widget.summary_text.setPlainText(message)

    def _show_event(self, epoch: int) -> None:
        for level in LEVELS:
            widget = self._level_widgets[level]
            try:
                flight_loc = self._index.locations[("flight", level)][epoch]
                ten_day_loc = self._index.locations[("10day", level)][epoch]
            except KeyError:
                widget.clear()
                widget.summary_text.setPlainText(f"No matched {level.upper()} event found for epoch {epoch}.")
                continue
            widget.populate(
                summary_lines=self._build_summary(level, epoch, flight_loc, ten_day_loc),
                row_comparisons=self._compare_event_rows(level, flight_loc, ten_day_loc),
                global_comparisons=self._compare_global_attrs(flight_loc.path, ten_day_loc.path),
                variable_attr_comparisons=self._compare_variable_attrs(flight_loc.path, ten_day_loc.path),
            )

    def _build_summary(
        self,
        level: str,
        epoch: int,
        flight_loc: EventLocation,
        ten_day_loc: EventLocation,
    ) -> List[str]:
        cdf_flight = self._index.cache.open(flight_loc.path)
        cdf_ten_day = self._index.cache.open(ten_day_loc.path)
        return [
            f"Level: {level.upper()}",
            f"Event epoch (TT2000): {epoch}",
            f"Event epoch (ISO): {_epoch_to_iso(epoch)}",
            "",
            f"Flight file: {flight_loc.path.name}",
            f"Flight path: {flight_loc.path}",
            f"Flight row: {flight_loc.row_index}",
            f"Flight data version: {_stringify_value(_normalize_mapping(cdf_flight.globalattsget()).get('Data_version'), preview=True)}",
            "",
            f"10-Day file: {ten_day_loc.path.name}",
            f"10-Day path: {ten_day_loc.path}",
            f"10-Day row: {ten_day_loc.row_index}",
            f"10-Day data version: {_stringify_value(_normalize_mapping(cdf_ten_day.globalattsget()).get('Data_version'), preview=True)}",
        ]

    def _compare_global_attrs(self, flight_path: Path, ten_day_path: Path) -> List[AttributeComparison]:
        cdf_flight = self._index.cache.open(flight_path)
        cdf_ten_day = self._index.cache.open(ten_day_path)
        left = _normalize_mapping(cdf_flight.globalattsget())
        right = _normalize_mapping(cdf_ten_day.globalattsget())
        comparisons: List[AttributeComparison] = []
        for key in sorted(set(left) | set(right)):
            left_value = left.get(key)
            right_value = right.get(key)
            comparisons.append(
                AttributeComparison(
                    owner="",
                    key=key,
                    left_value=left_value,
                    right_value=right_value,
                    different=_stringify_value(left_value) != _stringify_value(right_value),
                )
            )
        return comparisons

    def _compare_variable_attrs(self, flight_path: Path, ten_day_path: Path) -> List[AttributeComparison]:
        cdf_flight = self._index.cache.open(flight_path)
        cdf_ten_day = self._index.cache.open(ten_day_path)
        variables = sorted(
            set(getattr(cdf_flight.cdf_info(), "zVariables", []))
            | set(getattr(cdf_ten_day.cdf_info(), "zVariables", []))
        )
        comparisons: List[AttributeComparison] = []
        for variable in variables:
            flight_vars = set(_cdf_zvariable_names(cdf_flight))
            ten_day_vars = set(_cdf_zvariable_names(cdf_ten_day))
            left_attrs = _normalize_mapping(cdf_flight.varattsget(variable)) if variable in flight_vars else {}
            right_attrs = _normalize_mapping(cdf_ten_day.varattsget(variable)) if variable in ten_day_vars else {}
            for key in sorted(set(left_attrs) | set(right_attrs)):
                left_value = left_attrs.get(key)
                right_value = right_attrs.get(key)
                comparisons.append(
                    AttributeComparison(
                        owner=str(variable),
                        key=key,
                        left_value=left_value,
                        right_value=right_value,
                        different=_stringify_value(left_value) != _stringify_value(right_value),
                    )
                )
        return comparisons

    def _compare_event_rows(
        self,
        level: str,
        flight_loc: EventLocation,
        ten_day_loc: EventLocation,
    ) -> List[VariableComparison]:
        del level
        cdf_flight = self._index.cache.open(flight_loc.path)
        cdf_ten_day = self._index.cache.open(ten_day_loc.path)
        variables = sorted(
            set(self._index.cache.event_variables(flight_loc.path))
            | set(self._index.cache.event_variables(ten_day_loc.path))
        )
        comparisons: List[VariableComparison] = []
        for variable in variables:
            left_missing = variable not in self._index.cache.event_variables(flight_loc.path)
            right_missing = variable not in self._index.cache.event_variables(ten_day_loc.path)
            left_value = None if left_missing else np.asarray(cdf_flight.varget(variable))[flight_loc.row_index]
            right_value = None if right_missing else np.asarray(cdf_ten_day.varget(variable))[ten_day_loc.row_index]
            different, summary = self._diff_summary(left_value, right_value)
            comparisons.append(
                VariableComparison(
                    name=str(variable),
                    left_value=left_value,
                    right_value=right_value,
                    left_shape=_shape_text(left_value),
                    right_shape=_shape_text(right_value),
                    diff_summary=summary,
                    different=different,
                    left_preview=_stringify_value(left_value, preview=True),
                    right_preview=_stringify_value(right_value, preview=True),
                )
            )
        return comparisons

    @staticmethod
    def _diff_summary(left_value: Any, right_value: Any) -> Tuple[bool, str]:
        if left_value is None or right_value is None:
            return True, "Missing"
        left = np.asarray(left_value)
        right = np.asarray(right_value)
        if left.shape != right.shape:
            return True, f"Shape mismatch: {left.shape} vs {right.shape}"
        if left.dtype.kind in "OUS" or right.dtype.kind in "OUS":
            same = np.array_equal(left, right)
            return (not same, "Different" if not same else "Equal")
        if np.array_equal(left, right, equal_nan=True):
            return False, "Equal"
        diff = left.astype(float) - right.astype(float)
        absdiff = np.abs(diff)
        changed = int(np.count_nonzero(~np.isclose(left, right, equal_nan=True)))
        max_abs = float(np.nanmax(absdiff))
        mean_abs = float(np.nanmean(absdiff))
        return True, f"max|Δ|={max_abs:.6g}, mean|Δ|={mean_abs:.6g}, n={changed}"


def launch_idex_cdf_event_compare_viewer(
    *,
    flight_root: Optional[str] = None,
    ten_day_root: Optional[str] = None,
) -> int:
    app = QApplication.instance() or QApplication([])
    window = IDEXCDFEventCompareWindow(
        flight_root=flight_root,
        ten_day_root=ten_day_root,
    )
    window.show()
    return app.exec()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare matched IDEX CDF events between the flight and 10-day libraries."
    )
    parser.add_argument(
        "--flight-root",
        default=str(_default_flight_root()),
        help="Root directory containing the flight CDF l1a/l1b/l2a folders.",
    )
    parser.add_argument(
        "--ten-day-root",
        default=str(_default_ten_day_root()),
        help="Root directory containing the idex-10day l1a/l1b/l2a folders.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return launch_idex_cdf_event_compare_viewer(
        flight_root=args.flight_root,
        ten_day_root=args.ten_day_root,
    )


if __name__ == "__main__":  # pragma: no cover - manual tool entry point
    raise SystemExit(main())
