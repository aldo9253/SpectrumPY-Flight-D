#!/usr/bin/env python3
"""Interactive viewer for IDEX L2C rectangular map CDF products.

Run from the SpectrumPY micromamba environment with, for example::

    python IDEX-L2C-quicklook.py
    python IDEX-L2C-quicklook.py /path/to/imap_idex_l2c_*.cdf

The viewer performs no scientific processing. It reads the map arrays and their
coordinate variables from the CDF and displays one epoch record at a time.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

try:  # Prefer the binding used by the existing SpectrumPY viewers.
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:  # pragma: no cover - depends on the local GUI installation
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

import cdflib


MAP_NAMES = (
    "counts_map",
    "rate_map",
    "counts_by_charge_map",
    "rate_by_charge_map",
    "counts_by_mass_map",
    "rate_by_mass_map",
)
LON_NAMES = ("rectangular_lon_pixel", "longitude", "lon")
LAT_NAMES = ("rectangular_lat_pixel", "latitude", "lat")


def _cdf_names(cdf: cdflib.CDF) -> list[str]:
    info = cdf.cdf_info()
    if isinstance(info, dict):
        return list(info.get("zVariables", []))
    return list(getattr(info, "zVariables", []))


def _first_available(cdf: cdflib.CDF, names: tuple[str, ...]) -> str | None:
    available = set(_cdf_names(cdf))
    return next((name for name in names if name in available), None)


def _array(cdf: cdflib.CDF, name: str) -> np.ndarray:
    return np.asarray(cdf.varget(name))


def _global_text(cdf: cdflib.CDF, name: str) -> str | None:
    value = cdf.globalattsget().get(name)
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0] if len(value) else None
    return str(value) if value is not None else None


def _format_epoch(value: object) -> str:
    """Format a CDF TT2000 value as a UTC timestamp."""
    converted = np.asarray(cdflib.cdfepoch.to_datetime(value)).reshape(-1)[0]
    return str(converted).replace("T", " ") + " UTC"


class L2CMapWindow(QMainWindow):
    """Window containing map selection, epoch navigation, and a Matplotlib map."""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = Path(filename)
        self.cdf = cdflib.CDF(str(self.filename))
        global_attributes = self.cdf.globalattsget()
        self.coordinate_frame = str(
            global_attributes.get("Spice_reference_frame", "unspecified")
        )
        self.start_date = _global_text(self.cdf, "Start_date")
        self.impact_days = (
            _array(self.cdf, "impact_day_of_year")
            if "impact_day_of_year" in _cdf_names(self.cdf)
            else None
        )
        self.epochs = (
            _array(self.cdf, "epoch") if "epoch" in _cdf_names(self.cdf) else None
        )
        self.names = _cdf_names(self.cdf)
        self.map_names = [name for name in MAP_NAMES if name in self.names]
        if not self.map_names:
            raise ValueError("The selected CDF contains no supported L2C map variables.")

        self.lon_name = _first_available(self.cdf, LON_NAMES)
        self.lat_name = _first_available(self.cdf, LAT_NAMES)
        if self.lon_name is None or self.lat_name is None:
            raise ValueError("The selected CDF has no longitude/latitude pixel coordinates.")

        self.map_combo = QComboBox()
        self.map_combo.addItems(self.map_names)
        self.map_combo.currentTextChanged.connect(self._plot)
        self.epoch_combo = QComboBox()
        self.epoch_combo.currentIndexChanged.connect(self._plot)
        self.previous_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.previous_button.clicked.connect(self._previous_epoch)
        self.next_button.clicked.connect(self._next_epoch)
        self.status = QLabel()

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.axes = self.figure.add_subplot(111, projection="mollweide")
        self.colorbar = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Map:"))
        controls.addWidget(self.map_combo, 1)
        controls.addWidget(QLabel("Epoch:"))
        controls.addWidget(self.epoch_combo)
        controls.addWidget(self.previous_button)
        controls.addWidget(self.next_button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.setCentralWidget(central)
        self.setWindowTitle(f"IDEX L2C Map Quicklook — {self.filename.name}")
        self.resize(1100, 800)

        self._populate_epochs()
        self._plot()

    def _populate_epochs(self) -> None:
        epoch_count = int(_array(self.cdf, self.map_names[0]).shape[0])
        labels = []
        for index in range(epoch_count):
            label = f"{index + 1}"
            if self.impact_days is not None and index < len(self.impact_days):
                label += f" (DOY {int(self.impact_days[index])})"
            labels.append(label)
        self.epoch_combo.addItems(labels)

    def _time_description(self, epoch_index: int) -> str:
        """Return the selected daily window and event-time centroid."""
        parts = []
        if self.impact_days is not None and epoch_index < len(self.impact_days):
            day_of_year = int(self.impact_days[epoch_index])
            year = (
                int(self.start_date[:4])
                if self.start_date
                else datetime.now(timezone.utc).year
            )
            if self.start_date and day_of_year < datetime.strptime(
                self.start_date, "%Y%m%d"
            ).timetuple().tm_yday:
                year += 1
            window_start = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
                days=day_of_year - 1
            )
            window_end = window_start + timedelta(days=1)
            parts.append(
                f"daily window: {window_start:%Y-%m-%d %H:%M}–"
                f"{window_end:%Y-%m-%d %H:%M} UTC"
            )
        if self.epochs is not None and epoch_index < len(self.epochs):
            parts.append(
                f"event-time centroid: {_format_epoch(self.epochs[epoch_index])}"
            )
        return " | ".join(parts) or "time window unavailable"

    def _previous_epoch(self) -> None:
        self.epoch_combo.setCurrentIndex(max(0, self.epoch_combo.currentIndex() - 1))

    def _next_epoch(self) -> None:
        self.epoch_combo.setCurrentIndex(
            min(self.epoch_combo.count() - 1, self.epoch_combo.currentIndex() + 1)
        )

    def _plot(self) -> None:
        if not self.epoch_combo.count():
            return
        map_name = self.map_combo.currentText()
        values = _array(self.cdf, map_name)
        epoch_index = self.epoch_combo.currentIndex()
        values = np.squeeze(values[epoch_index])
        while values.ndim > 2:
            # Legacy charge/mass maps have an additional bin dimension. A map
            # selector can still preview the first bin without changing data.
            values = values[0]
        if values.ndim != 2:
            raise ValueError(f"Unsupported map shape for {map_name}: {values.shape}")

        longitude = np.ravel(_array(self.cdf, self.lon_name)).astype(float)
        latitude = np.ravel(_array(self.cdf, self.lat_name)).astype(float)
        if values.shape == (len(longitude), len(latitude)):
            image = values
        elif values.shape == (len(latitude), len(longitude)):
            image = values.T
        else:
            raise ValueError(
                f"Map shape {values.shape} does not match coordinate sizes "
                f"({len(longitude)}, {len(latitude)})."
            )

        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None
        # Mollweide uses longitudes from -180 to +180. Reorder the CDF's
        # conventional [0, 360) longitude grid without changing the values.
        projected_longitude = (longitude + 180.0) % 360.0 - 180.0
        longitude_order = np.argsort(projected_longitude)
        latitude_order = np.argsort(latitude)
        projected_longitude = projected_longitude[longitude_order]
        latitude = latitude[latitude_order]
        image = image[longitude_order, :][:, latitude_order]

        self.axes.clear()
        masked = np.ma.masked_invalid(image.astype(float, copy=False))
        mesh = self.axes.pcolormesh(
            np.deg2rad(projected_longitude),
            np.deg2rad(latitude),
            masked.T,
            shading="auto",
            cmap="viridis",
        )
        self.axes.set_xlabel("Longitude")
        self.axes.set_ylabel("Latitude")
        self.axes.set_title(
            f"{map_name} — epoch {epoch_index + 1} — {self.coordinate_frame}\n"
            f"{self._time_description(epoch_index)}"
        )
        self.axes.grid(True, alpha=0.35)
        self.colorbar = self.figure.colorbar(
            mesh, ax=self.axes, label=self._units(map_name)
        )
        self.status.setText(
            f"{self.filename.name} | {map_name} | shape {image.shape} | "
            f"displaying epoch {epoch_index + 1} of {self.epoch_combo.count()} | "
            f"coordinate frame: {self.coordinate_frame} | "
            f"{self._time_description(epoch_index)}"
        )
        self.canvas.draw_idle()

    def _units(self, map_name: str) -> str:
        attrs = self.cdf.varattsget(map_name)
        return str(attrs.get("UNITS", "value")) if attrs else "value"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", nargs="?", help="L2C CDF to open")
    args = parser.parse_args(argv)

    app = QApplication.instance() or QApplication(sys.argv)
    filename = args.filename
    if filename is None:
        filename, _ = QFileDialog.getOpenFileName(
            None,
            "Open IDEX L2C CDF",
            str(Path.cwd()),
            "CDF files (*.cdf);;All files (*)",
        )
    if not filename:
        return 0
    try:
        window = L2CMapWindow(filename)
    except Exception as exc:
        QMessageBox.critical(None, "Unable to open L2C CDF", str(exc))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    raise SystemExit(main())
