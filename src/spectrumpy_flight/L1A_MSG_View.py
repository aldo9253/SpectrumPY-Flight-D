"""Qt viewer for browsing directory collections of ``imap_idex_l1a_msg`` CDF files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cdflib  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "L1A MSG viewer requires the optional 'cdflib' dependency."
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
        QPlainTextEdit,
        QPushButton,
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
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
    _QT_API = "PyQt6"


FILE_GLOB = "imap_idex_l1a_msg_*.cdf"
DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "CDF" / "l1a_msg"


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


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, np.datetime64):
        try:
            microseconds = int(value.astype("datetime64[us]").astype(np.int64))
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
            return epoch + timedelta(microseconds=microseconds)
        except Exception:
            return None
    if hasattr(value, "to_pydatetime"):
        try:
            return _coerce_datetime(value.to_pydatetime())
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


def _epoch_to_datetime(epoch_value: Any) -> Optional[datetime]:
    seconds = _epoch_to_seconds(epoch_value)
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except Exception:
        return None


def _epoch_vector_to_iso(epoch_values: np.ndarray) -> List[str]:
    arr = np.asarray(epoch_values).reshape(-1)
    if arr.size == 0:
        return []
    try:
        converted = cdflib.cdfepoch.to_datetime(arr)
    except Exception:
        converted = None
    if converted:
        texts: List[str] = []
        for dt in converted:
            try:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                texts.append(dt.isoformat())
            except Exception:
                texts.append(_stringify_value(dt))
        return texts
    return [_stringify_value(value) for value in arr]


def _read_vector(cdf: Any, varname: str) -> np.ndarray:
    try:
        data = np.asarray(cdf.varget(varname))
    except Exception:
        return np.asarray([])
    return data.reshape(-1)


@dataclass(frozen=True)
class L1AMsgTableRow:
    epoch: str
    epoch_dt: Optional[datetime]
    shcoarse: str
    shfine: str
    message: str
    filename: str
    path: Path
    row_index: int


@dataclass
class L1AMsgRecord:
    path: Path
    filename: str
    epoch_values: np.ndarray
    epoch_texts: List[str]
    shcoarse_values: np.ndarray
    shfine_values: np.ndarray
    messages_values: np.ndarray
    global_attributes: Dict[str, Any]
    variable_rows: List[Tuple[str, Any, str]]
    variable_attributes: List[Tuple[str, str, Any]]

    def summary_lines(self) -> List[str]:
        lines = [
            f"File: {self.filename}",
            f"Path: {self.path}",
            f"Rows: {len(self.epoch_values)}",
            f"Epoch shape: {_shape_text(self.epoch_values)}",
            f"shcoarse shape: {_shape_text(self.shcoarse_values)}",
            f"shfine shape: {_shape_text(self.shfine_values)}",
            f"messages shape: {_shape_text(self.messages_values)}",
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


def _load_l1a_msg_record(path: Path) -> L1AMsgRecord:
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    names: List[str] = []
    for key in ("zVariables", "rVariables"):
        values = info.get(key, []) if isinstance(info, dict) else getattr(info, key, [])
        for name in values or []:
            names.append(str(name))

    global_attributes = _normalize_attr_mapping(cdf.globalattsget())
    epoch_values = _read_vector(cdf, "epoch")
    shcoarse_values = _read_vector(cdf, "shcoarse")
    shfine_values = _read_vector(cdf, "shfine")
    messages_values = _read_vector(cdf, "messages")
    epoch_texts = _epoch_vector_to_iso(epoch_values)

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

    return L1AMsgRecord(
        path=path,
        filename=path.name,
        epoch_values=epoch_values,
        epoch_texts=epoch_texts,
        shcoarse_values=shcoarse_values,
        shfine_values=shfine_values,
        messages_values=messages_values,
        global_attributes=global_attributes,
        variable_rows=variable_rows,
        variable_attributes=variable_attributes,
    )


class L1AMsgViewWindow(QMainWindow):
    """Interactive browser for collections of ``imap_idex_l1a_msg`` CDF files."""

    RAW_HEADERS = (
        "Epoch",
        "shcoarse",
        "shfine",
        "message",
        "file",
    )

    @staticmethod
    def _time_sort_key(
        epoch_dt: Optional[datetime], epoch_text: str, filename: str, row_index: int
    ) -> Tuple[Any, ...]:
        return (
            epoch_dt is None,
            epoch_dt or datetime.max.replace(tzinfo=timezone.utc),
            epoch_text,
            filename,
            row_index,
        )

    def __init__(
        self, directory: Optional[str] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._directory = Path(directory).expanduser() if directory else DEFAULT_DIRECTORY
        self._records: List[L1AMsgRecord] = []
        self._table_rows: List[Tuple[L1AMsgTableRow, L1AMsgRecord]] = []

        self.setWindowTitle("IDEX L1A MSG Viewer")
        self.resize(1280, 820)

        self._directory_edit: QLineEdit
        self._raw_table: QTableWidget
        self._summary_text: QPlainTextEdit
        self._global_attrs_table: QTableWidget
        self._row_values_table: QTableWidget
        self._variable_attrs_table: QTableWidget

        self._build_ui()
        self.reload_directory()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        toolbar = QToolBar("L1A MSG Tools", self)
        toolbar.setMovable(False)
        browse_action = QAction("Browse Directory…", self)
        browse_action.triggered.connect(self._choose_directory)
        toolbar.addAction(browse_action)
        reload_action = QAction("Reload", self)
        reload_action.triggered.connect(self.reload_directory)
        toolbar.addAction(reload_action)
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

        self._raw_table = QTableWidget(splitter)
        self._raw_table.setColumnCount(len(self.RAW_HEADERS))
        self._raw_table.setHorizontalHeaderLabels(list(self.RAW_HEADERS))
        self._raw_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._raw_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._raw_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._raw_table.setAlternatingRowColors(True)
        self._raw_table.verticalHeader().setVisible(False)
        self._raw_table.horizontalHeader().setStretchLastSection(False)
        self._raw_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self._raw_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._raw_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self._raw_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self._raw_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self._raw_table.itemSelectionChanged.connect(self._on_raw_selection_changed)

        tabs = QTabWidget(splitter)

        self._summary_text = QPlainTextEdit(tabs)
        self._summary_text.setReadOnly(True)
        tabs.addTab(self._summary_text, "Summary")

        self._row_values_table = QTableWidget(tabs)
        self._row_values_table.setColumnCount(2)
        self._row_values_table.setHorizontalHeaderLabels(["Variable", "Value"])
        self._row_values_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._row_values_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self._row_values_table, "Row Values")

        self._global_attrs_table = QTableWidget(tabs)
        self._global_attrs_table.setColumnCount(2)
        self._global_attrs_table.setHorizontalHeaderLabels(["Attribute", "Value"])
        self._global_attrs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._global_attrs_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self._global_attrs_table, "Global Attributes")

        self._variable_attrs_table = QTableWidget(tabs)
        self._variable_attrs_table.setColumnCount(3)
        self._variable_attrs_table.setHorizontalHeaderLabels(
            ["Variable", "Attribute", "Value"]
        )
        self._variable_attrs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._variable_attrs_table.horizontalHeader().setStretchLastSection(True)
        tabs.addTab(self._variable_attrs_table, "Variable Attributes")

        self.setCentralWidget(central)

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Select L1A MSG Directory", str(self._directory)
        )
        if not selected:
            return
        self._directory = Path(selected)
        self._directory_edit.setText(str(self._directory))
        self.reload_directory()

    def reload_directory(self) -> None:
        self._directory = Path(
            self._directory_edit.text().strip() or self._directory
        ).expanduser()
        self._directory_edit.setText(str(self._directory))
        if not self._directory.exists():
            QMessageBox.warning(
                self, "Directory Not Found", f"Directory not found:\n{self._directory}"
            )
            self._records = []
            self._table_rows = []
            self._populate_raw_table()
            self._clear_details()
            return

        records: List[L1AMsgRecord] = []
        failures: List[str] = []
        for path in sorted(self._directory.glob(FILE_GLOB)):
            try:
                records.append(_load_l1a_msg_record(path))
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")

        self._records = records
        self._rebuild_table_rows()
        self._populate_raw_table()
        self._clear_details()
        if self._table_rows:
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
        row_count = len(self._table_rows)
        self.setWindowTitle(
            "IDEX L1A MSG Viewer — "
            f"{self._directory} ({row_count} combined rows from {file_count} "
            f"file{'s' if file_count != 1 else ''})"
        )

    def _rebuild_table_rows(self) -> None:
        rows: List[Tuple[L1AMsgTableRow, L1AMsgRecord]] = []
        for record in self._records:
            row_count = min(
                len(record.epoch_values),
                len(record.shcoarse_values),
                len(record.shfine_values),
                len(record.messages_values),
            )
            for row_index in range(row_count):
                rows.append(
                    (
                        L1AMsgTableRow(
                            epoch=record.epoch_texts[row_index]
                            if row_index < len(record.epoch_texts)
                            else "",
                            epoch_dt=_epoch_to_datetime(record.epoch_values[row_index]),
                            shcoarse=_stringify_value(record.shcoarse_values[row_index]),
                            shfine=_stringify_value(record.shfine_values[row_index]),
                            message=_stringify_value(record.messages_values[row_index]),
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

    def _populate_raw_table(self) -> None:
        self._raw_table.setRowCount(len(self._table_rows))
        for row_index, (entry, _record) in enumerate(self._table_rows):
            values = [
                entry.epoch,
                entry.shcoarse,
                entry.shfine,
                entry.message,
                entry.filename,
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_index in (1, 2, 4):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._raw_table.setItem(row_index, col_index, item)
        self._raw_table.resizeRowsToContents()

    def _on_raw_selection_changed(self) -> None:
        indexes = self._raw_table.selectionModel().selectedRows()
        if not indexes:
            self._clear_details()
            return
        self._show_raw_record(indexes[0].row())

    def _show_raw_record(self, table_index: int) -> None:
        if table_index < 0 or table_index >= len(self._table_rows):
            self._clear_details()
            return
        entry, record = self._table_rows[table_index]
        self._summary_text.setPlainText(
            "\n".join(
                record.summary_lines()
                + [
                    "",
                    f"Selected row: {entry.row_index}",
                    f"Selected epoch: {entry.epoch}",
                    f"Selected message: {entry.message}",
                ]
            )
        )

        row_values = [
            ("epoch", entry.epoch),
            ("shcoarse", entry.shcoarse),
            ("shfine", entry.shfine),
            ("messages", entry.message),
        ]
        self._populate_table(self._row_values_table, row_values)
        self._populate_table(
            self._global_attrs_table,
            [
                (key, _stringify_value(value))
                for key, value in sorted(record.global_attributes.items())
            ],
        )
        self._populate_table(
            self._variable_attrs_table,
            [
                (name, attr_name, _stringify_value(attr_value))
                for name, attr_name, attr_value in record.variable_attributes
            ],
        )

    def _clear_details(self) -> None:
        self._summary_text.clear()
        for table in (
            self._row_values_table,
            self._global_attrs_table,
            self._variable_attrs_table,
        ):
            table.clearContents()
            table.setRowCount(0)

    @staticmethod
    def _populate_table(
        table: QTableWidget, rows: Iterable[Sequence[str]]
    ) -> None:
        items = list(rows)
        table.setRowCount(len(items))
        for row_index, values in enumerate(items):
            for col_index, value in enumerate(values):
                table.setItem(row_index, col_index, QTableWidgetItem(value))
        table.resizeRowsToContents()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else [])
    directory = args[0] if args else None

    app = QApplication.instance() or QApplication([])
    window = L1AMsgViewWindow(directory=directory)
    window.show()
    return app.exec()


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
