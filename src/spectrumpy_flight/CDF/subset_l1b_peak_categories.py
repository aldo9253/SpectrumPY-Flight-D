#!/usr/bin/env python3
"""Create L1B CDF subsets for manual dust, wide-peak, and remaining events."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cdflib
import numpy as np
import pandas as pd
from cdflib import cdfwrite


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_L1B_DIR = SCRIPT_DIR / "l1b"
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR / "l1b_peak_analysis"
DEFAULT_EVENT_LIST = DEFAULT_ANALYSIS_DIR / "IDEX_Event_List.csv"
DEFAULT_SUMMARY = DEFAULT_ANALYSIS_DIR / "l1b_tof_peak_event_summary_default.csv"
DEFAULT_PEAKS = DEFAULT_ANALYSIS_DIR / "l1b_tof_peak_peaks_default.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "category_cdfs"
TT2000_TOLERANCE_NS = 1_000_000_000


@dataclass(frozen=True)
class EventRef:
    epoch_tt2000: int
    epoch_utc: str
    source_file: str
    record_index: int


def cdf_var_names(cdf: cdflib.CDF) -> list[str]:
    info = cdf.cdf_info()
    if isinstance(info, dict):
        return list(info.get("rVariables", [])) + list(info.get("zVariables", []))
    return list(info.rVariables) + list(info.zVariables)


def cdf_info_value(info, name: str):
    if isinstance(info, dict):
        return info[name]
    return getattr(info, name)


def var_spec(varinq, *, preserve_compression: bool = True) -> dict:
    def value(name: str):
        if isinstance(varinq, dict):
            return varinq[name]
        return getattr(varinq, name)

    spec = {
        "Variable": value("Variable"),
        "Data_Type": value("Data_Type"),
        "Num_Elements": value("Num_Elements"),
        "Rec_Vary": value("Rec_Vary"),
        "Dim_Sizes": list(value("Dim_Sizes")),
        "Var_Type": value("Var_Type"),
        "Sparse": value("Sparse"),
        "Compress": value("Compress"),
        "Block_Factor": value("Block_Factor"),
    }
    pad = varinq.get("Pad") if isinstance(varinq, dict) else getattr(varinq, "Pad", None)
    if pad is not None:
        spec["Pad"] = pad
    if value("Var_Type") == "rVariable":
        spec["Dim_Vary"] = list(value("Dim_Vary"))
    if not preserve_compression:
        spec["Compress"] = 0
        spec["Block_Factor"] = 1
    return spec


def concat_records(chunks: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(chunk) for chunk in chunks]
    if not arrays:
        raise ValueError("No record chunks to concatenate")
    return np.concatenate(arrays, axis=0)


def epoch_to_iso(epoch_tt2000: int) -> str:
    encoded = cdflib.cdfepoch.encode_tt2000(int(epoch_tt2000))
    if isinstance(encoded, list):
        return str(encoded[0]) if encoded else str(epoch_tt2000)
    return str(encoded)


def timestamp_to_tt2000(timestamp: pd.Timestamp) -> int:
    return int(
        cdflib.cdfepoch.compute_tt2000(
            [
                int(timestamp.year),
                int(timestamp.month),
                int(timestamp.day),
                int(timestamp.hour),
                int(timestamp.minute),
                int(timestamp.second),
                int(timestamp.microsecond // 1000),
                int(timestamp.microsecond % 1000),
                0,
            ]
        )
    )


def load_summary_events(path: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    required = {"epoch_tt2000", "epoch_utc", "source_file", "record_index"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    events = summary[list(required)].drop_duplicates().copy()
    events["epoch_tt2000"] = events["epoch_tt2000"].astype(np.int64)
    events["record_index"] = events["record_index"].astype(int)
    events = events.sort_values("epoch_tt2000").reset_index(drop=True)
    return events


def load_event_list(path: Path) -> pd.DataFrame:
    event_list = pd.read_csv(path, header=1)
    event_list.columns = [str(column).strip() for column in event_list.columns]
    required = {"Time", "Event type", "Dust category"}
    missing = required.difference(event_list.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    for column in required:
        event_list[column] = event_list[column].astype(str).str.strip()
    event_list["event_time"] = pd.to_datetime(
        event_list["Time"],
        format="%d-%b-%Y %H:%M:%S",
        errors="coerce",
    )
    event_list = event_list.dropna(subset=["event_time"]).copy()
    event_list["event_tt2000"] = event_list["event_time"].map(timestamp_to_tt2000).astype(np.int64)
    return event_list


def matched_epochs(
    summary_events: pd.DataFrame,
    event_list_rows: pd.DataFrame,
    *,
    tolerance_ns: int,
) -> set[int]:
    if event_list_rows.empty:
        return set()
    summary_epochs = summary_events["epoch_tt2000"].to_numpy(dtype=np.int64)
    matched: set[int] = set()
    for requested in event_list_rows["event_tt2000"].to_numpy(dtype=np.int64):
        deltas = np.abs(summary_epochs - requested)
        nearest = int(np.argmin(deltas))
        if int(deltas[nearest]) <= tolerance_ns:
            matched.add(int(summary_epochs[nearest]))
    return matched


def categorized_events(
    summary_events: pd.DataFrame,
    event_list: pd.DataFrame,
    peaks_path: Path,
    *,
    wide_threshold_us: float,
    tolerance_ns: int,
) -> dict[str, pd.DataFrame]:
    manual_rows = event_list[
        event_list["Event type"].str.casefold().eq("triggered")
        & event_list["Dust category"].str.casefold().eq("dust")
    ]
    pulser_rows = event_list[event_list["Event type"].str.casefold().eq("pulser")]

    manual_epochs = matched_epochs(summary_events, manual_rows, tolerance_ns=tolerance_ns)
    pulser_epochs = matched_epochs(summary_events, pulser_rows, tolerance_ns=tolerance_ns)

    peaks = pd.read_csv(peaks_path)
    required = {"epoch_tt2000", "width_us"}
    missing = required.difference(peaks.columns)
    if missing:
        raise ValueError(f"{peaks_path} is missing required columns: {', '.join(sorted(missing))}")
    peaks["epoch_tt2000"] = peaks["epoch_tt2000"].astype(np.int64)
    peaks["width_us"] = pd.to_numeric(peaks["width_us"], errors="coerce")
    wide_epochs = set(
        int(epoch)
        for epoch in peaks.loc[peaks["width_us"] > wide_threshold_us, "epoch_tt2000"].dropna().unique()
    )
    narrow_wide_epochs = set(
        int(epoch)
        for epoch in peaks.loc[peaks["width_us"] > 0.01, "epoch_tt2000"].dropna().unique()
    )
    peak_gt_0p02_epochs = set(
        int(epoch)
        for epoch in peaks.loc[peaks["width_us"] > 0.02, "epoch_tt2000"].dropna().unique()
    )
    peak_gt_0p04_epochs = set(
        int(epoch)
        for epoch in peaks.loc[peaks["width_us"] > 0.04, "epoch_tt2000"].dropna().unique()
    )
    event_peak_counts = peaks.groupby("epoch_tt2000").size()
    multi_peak_epochs = set(int(epoch) for epoch in event_peak_counts[event_peak_counts > 1].index)
    peak_gt_0p4_epochs = set(
        int(epoch)
        for epoch in peaks.loc[peaks["width_us"] > 0.4, "epoch_tt2000"].dropna().unique()
    )

    all_epochs = set(int(epoch) for epoch in summary_events["epoch_tt2000"])
    eligible_epochs = all_epochs - pulser_epochs
    manual_epochs &= eligible_epochs
    wide_nonmanual_epochs = (wide_epochs & eligible_epochs) - manual_epochs
    narrow_wide_nonmanual_epochs = (narrow_wide_epochs & eligible_epochs) - manual_epochs
    peak_gt_0p02_nonmanual_epochs = (peak_gt_0p02_epochs & eligible_epochs) - manual_epochs
    peak_gt_0p04_nonmanual_epochs = (peak_gt_0p04_epochs & eligible_epochs) - manual_epochs
    peak_gt_0p04_multi_peak_epochs = ((peak_gt_0p04_epochs & multi_peak_epochs) & eligible_epochs) - manual_epochs
    peak_gt_0p4_multi_peak_epochs = (peak_gt_0p4_epochs & multi_peak_epochs) & all_epochs
    rest_epochs = eligible_epochs - manual_epochs - wide_nonmanual_epochs

    categories = {
        "manual_triggered_dust": manual_epochs,
        "nonmanual_wide_peaks_gt_0p1us": wide_nonmanual_epochs,
        "nonmanual_peaks_gt_0p01us": narrow_wide_nonmanual_epochs,
        "nonmanual_peaks_gt_0p02us": peak_gt_0p02_nonmanual_epochs,
        "nonmanual_peaks_gt_0p04us": peak_gt_0p04_nonmanual_epochs,
        "all_events_peak_gt_0p04us_multi_peak": peak_gt_0p04_multi_peak_epochs,
        "all_events_peak_gt_0p4us_multi_peak": peak_gt_0p4_multi_peak_epochs,
        "remaining_nonpulser": rest_epochs,
    }

    return {
        name: summary_events[summary_events["epoch_tt2000"].isin(epochs)].sort_values("epoch_tt2000").reset_index(drop=True)
        for name, epochs in categories.items()
    }


def global_attrs_for_output(template: cdflib.CDF, output: Path, source_paths: list[Path], events: pd.DataFrame) -> dict:
    attrs = template.globalattsget()
    attrs["Logical_file_id"] = [output.with_suffix("").name]
    attrs["Data_version"] = ["001"]
    attrs["Parents"] = [path.name for path in source_paths]
    if not events.empty:
        start = str(events.iloc[0]["epoch_utc"])[:10].replace("-", "")
        attrs["Start_date"] = [start]
    return {name: {entry_number: value for entry_number, value in enumerate(values)} for name, values in attrs.items()}


def write_subset(
    source_dir: Path,
    output: Path,
    events: pd.DataFrame,
    *,
    preserve_compression: bool = True,
    variable_names: Iterable[str] | None = None,
) -> None:
    if events.empty:
        raise ValueError(f"No events selected for {output.name}")

    source_paths = {path.name: path for path in source_dir.glob("imap_idex_l1b_sci-10days_*.cdf")}
    missing_sources = sorted(set(events["source_file"]) - set(source_paths))
    if missing_sources:
        raise FileNotFoundError(f"Missing source CDF files: {', '.join(missing_sources)}")

    by_source: dict[str, list[int]] = defaultdict(list)
    for row in events.itertuples(index=False):
        by_source[str(row.source_file)].append(int(row.record_index))

    used_sources = [source_paths[name] for name in sorted(by_source)]
    template = cdflib.CDF(str(used_sources[0]))
    info = template.cdf_info()
    cdf_spec = {
        "Majority": cdf_info_value(info, "Majority"),
        "Encoding": cdf_info_value(info, "Encoding"),
        "Checksum": cdf_info_value(info, "Checksum"),
        "Compressed": cdf_info_value(info, "Compressed"),
        "rDim_sizes": list(cdf_info_value(info, "rDim_sizes")),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cdfwrite.CDF(str(output), cdf_spec=cdf_spec, delete=True)
    try:
        writer.write_globalattrs(global_attrs_for_output(template, output, used_sources, events))
        selected_variables = list(variable_names) if variable_names is not None else cdf_var_names(template)
        for varname in selected_variables:
            varinq = template.varinq(varname)
            attrs = template.varattsget(varname)
            rec_vary = varinq["Rec_Vary"] if isinstance(varinq, dict) else varinq.Rec_Vary
            if rec_vary:
                chunks = []
                for source in used_sources:
                    cdf = cdflib.CDF(str(source))
                    records = by_source[source.name]
                    chunks.append(np.asarray(cdf.varget(varname))[records])
                data = concat_records(chunks)
            else:
                data = template.varget(varname)
            writer.write_var(var_spec(varinq, preserve_compression=preserve_compression), var_attrs=attrs, var_data=data)
    finally:
        writer.close()


def verify_subset(path: Path, events: pd.DataFrame) -> None:
    cdf = cdflib.CDF(str(path))
    epochs = np.asarray(cdf.varget("epoch"), dtype=np.int64)
    expected = events["epoch_tt2000"].to_numpy(dtype=np.int64)
    if epochs.size != expected.size:
        raise ValueError(f"{path}: expected {expected.size} events, found {epochs.size}")
    if not np.array_equal(np.sort(epochs), np.sort(expected)):
        raise ValueError(f"{path}: output epochs do not match selected events")


def write_manifest(output_dir: Path, categories: dict[str, pd.DataFrame]) -> Path:
    manifest_rows = []
    for category, events in categories.items():
        for row in events.itertuples(index=False):
            manifest_rows.append(
                {
                    "category": category,
                    "epoch_tt2000": int(row.epoch_tt2000),
                    "epoch_utc": row.epoch_utc,
                    "source_file": row.source_file,
                    "record_index": int(row.record_index),
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    path = output_dir / "l1b_peak_category_manifest.csv"
    manifest.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1b-dir", type=Path, default=DEFAULT_L1B_DIR)
    parser.add_argument("--event-list", type=Path, default=DEFAULT_EVENT_LIST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--peaks", type=Path, default=DEFAULT_PEAKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wide-threshold-us", type=float, default=0.1)
    parser.add_argument("--event-match-tolerance-seconds", type=float, default=1.0)
    args = parser.parse_args()

    summary_events = load_summary_events(args.summary)
    event_list = load_event_list(args.event_list)
    categories = categorized_events(
        summary_events,
        event_list,
        args.peaks,
        wide_threshold_us=args.wide_threshold_us,
        tolerance_ns=int(args.event_match_tolerance_seconds * 1_000_000_000),
    )

    outputs = {
        "manual_triggered_dust": args.output_dir / "imap_idex_l1b_manual_triggered_dust_events_v001.cdf",
        "nonmanual_wide_peaks_gt_0p1us": args.output_dir / "imap_idex_l1b_nonmanual_wide_peak_events_gt_0p1us_v001.cdf",
        "nonmanual_peaks_gt_0p01us": args.output_dir / "imap_idex_l1b_nonmanual_peak_events_gt_0p01us_v001.cdf",
        "nonmanual_peaks_gt_0p02us": args.output_dir / "imap_idex_l1b_nonmanual_peak_events_gt_0p02us_v001.cdf",
        "nonmanual_peaks_gt_0p04us": args.output_dir / "imap_idex_l1b_nonmanual_peak_events_gt_0p04us_v001.cdf",
        "all_events_peak_gt_0p04us_multi_peak": args.output_dir
        / "imap_idex_l1b_events_peak_gt_0p04us_multi_peak_v001.cdf",
        "all_events_peak_gt_0p4us_multi_peak": args.output_dir
        / "imap_idex_l1b_events_peak_gt_0p4us_multi_peak_v001.cdf",
        "remaining_nonpulser": args.output_dir / "imap_idex_l1b_remaining_nonpulser_events_v001.cdf",
    }

    for category, events in categories.items():
        output = outputs[category]
        write_subset(args.l1b_dir, output, events)
        verify_subset(output, events)
        print(f"{category}: {len(events)} events -> {output}")

    manifest = write_manifest(args.output_dir, categories)
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
