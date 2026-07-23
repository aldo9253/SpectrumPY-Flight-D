#!/usr/bin/env python3
"""Compare daily single-peak IDEX event rates with OMNI solar-wind conditions.

Solar-wind data are fetched from NASA SPDF OMNIWeb hourly OMNI2 listings. The
script computes daily proton flux as ``proton_density * flow_speed * 1e5`` so
the result is in ``cm^-2 s^-1`` when density is in ``cm^-3`` and speed is in
``km/s``.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from urllib import parse, request

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR / "l1b_peak_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "solar_wind_comparison"
DEFAULT_SUMMARY = DEFAULT_ANALYSIS_DIR / "l1b_tof_peak_fwhm_7sig_event_summary.csv"
DEFAULT_SINGLE_20NS = DEFAULT_ANALYSIS_DIR / "sat_aware_tof_7sig_single_peak_fwhm_ge_20ns_events.csv"
DEFAULT_SINGLE_40NS = DEFAULT_ANALYSIS_DIR / "sat_aware_tof_7sig_single_peak_fwhm_ge_40ns_events.csv"
OMNI_ENDPOINT = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# OMNIWeb dx1 variable IDs. See the OMNIWeb Data Explorer / command-line docs.
OMNI_VARIABLES = {
    "proton_density_cm3": "23",
    "flow_speed_km_s": "24",
    "flow_pressure_npa": "28",
    "scalar_b_nt": "8",
    "bz_gsm_nt": "16",
}

SOLAR_WINDOWS = (
    ("Mar 20-24", "2026-03-20", "2026-03-25"),
    ("Apr 1-5", "2026-04-01", "2026-04-06"),
    ("Apr 18-23", "2026-04-18", "2026-04-24"),
    ("May 15-20", "2026-05-15", "2026-05-21"),
    ("Jun 5-9", "2026-06-05", "2026-06-10"),
)


def apply_style() -> None:
    import sys

    if str(REPO_DIR) not in sys.path:
        sys.path.insert(0, str(REPO_DIR))
    try:
        from plot_style import apply_plot_style
    except Exception:
        plt.style.use("default")
    else:
        apply_plot_style("light")


def yyyymmdd(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y%m%d")


def fetch_omni_hourly(start: pd.Timestamp, stop: pd.Timestamp) -> str:
    params: list[tuple[str, str]] = [
        ("activity", "retrieve"),
        ("res", "hour"),
        ("spacecraft", "omni2"),
        ("start_date", yyyymmdd(start)),
        ("end_date", yyyymmdd(stop)),
        ("scale", "Linear"),
        ("view", "0"),
        ("table", "0"),
    ]
    for variable in OMNI_VARIABLES.values():
        params.append(("vars", variable))
    body = parse.urlencode(params).encode("ascii")
    req = request.Request(
        OMNI_ENDPOINT,
        data=body,
        headers={"User-Agent": "SpectrumPY-Flight solar wind comparison"},
    )
    with request.urlopen(req, timeout=120) as response:
        return response.read().decode("utf-8", "replace")


def parse_omni_listing(text: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | pd.Timestamp]] = []
    numeric = re.compile(r"^\s*(\d{4})\s+(\d{1,3})\s+(\d{1,2})\s+")
    for line in text.splitlines():
        if not numeric.match(line):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        year = int(parts[0])
        doy = int(parts[1])
        hour = int(parts[2])
        values = [float(value) for value in parts[3:8]]
        timestamp = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1, hours=hour)
        rows.append(
            {
                "time": timestamp,
                "proton_density_cm3": values[0],
                "flow_speed_km_s": values[1],
                "flow_pressure_npa": values[2],
                "scalar_b_nt": values[3],
                "bz_gsm_nt": values[4],
            }
        )
    if not rows:
        raise ValueError("No numeric OMNI rows found in OMNIWeb response")

    data = pd.DataFrame(rows)
    invalid_masks = {
        "proton_density_cm3": data["proton_density_cm3"] >= 900.0,
        "flow_speed_km_s": data["flow_speed_km_s"] >= 9000.0,
        "flow_pressure_npa": data["flow_pressure_npa"] >= 90.0,
        "scalar_b_nt": data["scalar_b_nt"] >= 900.0,
        "bz_gsm_nt": data["bz_gsm_nt"].abs() >= 900.0,
    }
    for column, mask in invalid_masks.items():
        data.loc[mask, column] = np.nan
    data["proton_flux_cm2_s"] = data["proton_density_cm3"] * data["flow_speed_km_s"] * 1.0e5
    return data


def load_or_fetch_omni(cache_path: Path, start: pd.Timestamp, stop: pd.Timestamp, refresh: bool) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["time"])
    text = fetch_omni_hourly(start, stop)
    hourly = parse_omni_listing(text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(cache_path, index=False)
    return hourly


def daily_omni(hourly: pd.DataFrame) -> pd.DataFrame:
    data = hourly.copy()
    data["date"] = data["time"].dt.normalize()
    return (
        data.groupby("date", as_index=False)
        .agg(
            omni_hour_count=("time", "size"),
            proton_density_cm3=("proton_density_cm3", "mean"),
            flow_speed_km_s=("flow_speed_km_s", "mean"),
            flow_pressure_npa=("flow_pressure_npa", "mean"),
            scalar_b_nt=("scalar_b_nt", "mean"),
            bz_gsm_nt_mean=("bz_gsm_nt", "mean"),
            bz_gsm_nt_min=("bz_gsm_nt", "min"),
            proton_flux_cm2_s=("proton_flux_cm2_s", "mean"),
        )
        .sort_values("date")
    )


def daily_event_counts(summary_path: Path, single_20ns_path: Path, single_40ns_path: Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)[["epoch_tt2000", "epoch_utc"]].drop_duplicates()
    summary["date"] = pd.to_datetime(summary["epoch_utc"]).dt.normalize()
    total = summary.groupby("date").size().rename("l1b_event_count")

    products = []
    for label, path in (("single_ge20ns_count", single_20ns_path), ("single_ge40ns_count", single_40ns_path)):
        events = pd.read_csv(path)
        events["date"] = pd.to_datetime(events["epoch_utc"]).dt.normalize()
        products.append(events.groupby("date").size().rename(label))

    daily = pd.concat([total, *products], axis=1).fillna(0).reset_index()
    daily["single_ge20ns_fraction"] = daily["single_ge20ns_count"] / daily["l1b_event_count"]
    daily["single_ge40ns_fraction"] = daily["single_ge40ns_count"] / daily["l1b_event_count"]
    return daily.sort_values("date")


def merged_daily(events: pd.DataFrame, omni: pd.DataFrame) -> pd.DataFrame:
    start = min(events["date"].min(), omni["date"].min())
    stop = max(events["date"].max(), omni["date"].max())
    dates = pd.DataFrame({"date": pd.date_range(start, stop, freq="D")})
    merged = dates.merge(events, on="date", how="left").merge(omni, on="date", how="left")
    for column in ("l1b_event_count", "single_ge20ns_count", "single_ge40ns_count"):
        merged[column] = merged[column].fillna(0)
    for column in ("single_ge20ns_fraction", "single_ge40ns_fraction"):
        merged[column] = merged[column].fillna(0.0)
    return merged


def shade_windows(axes: list[plt.Axes]) -> None:
    for label, start, stop in SOLAR_WINDOWS:
        start_time = pd.Timestamp(start)
        stop_time = pd.Timestamp(stop)
        for ax in axes:
            ax.axvspan(start_time, stop_time, color="#d8b365", alpha=0.14, linewidth=0)
        axes[0].text(
            start_time + (stop_time - start_time) / 2,
            0.97,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#6b5b2a",
        )


def plot_daily_comparison(data: pd.DataFrame, output: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 10.0), sharex=True)
    x = data["date"]

    axes[0].bar(x, data["single_ge20ns_count"], width=0.9, color="#0c5da5", alpha=0.72, label=">=20 ns")
    axes[0].bar(x, data["single_ge40ns_count"], width=0.55, color="#c62828", alpha=0.70, label=">=40 ns")
    axes[0].set_ylabel("Single-peak events / day")
    axes[0].set_yscale("log")
    axes[0].set_ylim(bottom=0.8)
    axes[0].legend(loc="upper left", frameon=False)

    axes[1].plot(x, data["proton_flux_cm2_s"], color="#7a5195", linewidth=1.4)
    axes[1].set_ylabel("Proton flux\n[cm$^{-2}$ s$^{-1}$]")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[2].plot(x, data["flow_pressure_npa"], color="#ef5675", linewidth=1.2, label="Pressure")
    axes[2].set_ylabel("Dynamic pressure [nPa]")
    axes[2].yaxis.label.set_color("#ef5675")
    axes[2].tick_params(axis="y", colors="#ef5675")
    ax_speed = axes[2].twinx()
    ax_speed.plot(x, data["flow_speed_km_s"], color="#ffa600", linewidth=1.0, alpha=0.8, label="Speed")
    ax_speed.set_ylabel("Speed [km/s]")
    ax_speed.yaxis.label.set_color("#ffa600")
    ax_speed.tick_params(axis="y", colors="#ffa600")

    axes[3].plot(x, data["scalar_b_nt"], color="#2f4b7c", linewidth=1.2, label="|B| daily mean")
    axes[3].plot(x, data["bz_gsm_nt_min"], color="#665191", linewidth=1.0, alpha=0.78, label="Bz GSM daily min")
    axes[3].axhline(0.0, color="#555555", linewidth=0.8, alpha=0.5)
    axes[3].set_ylabel("IMF [nT]")
    axes[3].legend(loc="upper left", frameon=False)

    for ax in axes:
        ax.grid(True, alpha=0.32)
    shade_windows(list(axes))

    locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Date")
    fig.suptitle("Daily Single-Peak IDEX Events vs OMNI Solar Wind", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_single_peak_histogram(data: pd.DataFrame, output: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(14.5, 4.2))
    x = data["date"]

    ax.bar(x, data["single_ge20ns_count"], width=0.9, color="#0c5da5", alpha=0.72, label=">=20 ns")
    ax.bar(x, data["single_ge40ns_count"], width=0.55, color="#c62828", alpha=0.70, label=">=40 ns")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8)
    ax.set_ylabel("Single-peak events / day")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.32)
    ax.legend(loc="upper left", frameon=False)

    for label, start, stop in SOLAR_WINDOWS:
        start_time = pd.Timestamp(start)
        stop_time = pd.Timestamp(stop)
        ax.axvspan(start_time, stop_time, color="#d8b365", alpha=0.14, linewidth=0)
        ax.text(
            start_time + (stop_time - start_time) / 2,
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#6b5b2a",
        )

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.suptitle("Daily Single-Peak IDEX Events", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--single-20ns", type=Path, default=DEFAULT_SINGLE_20NS)
    parser.add_argument("--single-40ns", type=Path, default=DEFAULT_SINGLE_40NS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-omni", action="store_true")
    args = parser.parse_args()

    events = daily_event_counts(args.summary, args.single_20ns, args.single_40ns)
    start = events["date"].min() - pd.Timedelta(days=1)
    stop = events["date"].max() + pd.Timedelta(days=1)

    cache_path = args.output_dir / "omni_hourly_solar_wind.csv"
    hourly = load_or_fetch_omni(cache_path, start, stop, args.refresh_omni)
    omni = daily_omni(hourly)
    daily = merged_daily(events, omni)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.output_dir / "single_peak_omni_daily_comparison.csv"
    plot_path = args.output_dir / "single_peak_omni_daily_comparison.png"
    histogram_path = args.output_dir / "single_peak_daily_histogram.png"
    daily.to_csv(daily_path, index=False)
    plot_daily_comparison(daily, plot_path)
    plot_single_peak_histogram(daily, histogram_path)

    print(f"Wrote {cache_path}")
    print(f"Wrote {daily_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {histogram_path}")


if __name__ == "__main__":
    main()
