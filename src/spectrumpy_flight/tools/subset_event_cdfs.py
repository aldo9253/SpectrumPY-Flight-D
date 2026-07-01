#!/usr/bin/env python3
"""Create L1B/L2A event subset CDFs from IDEX 10-day CDF products."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cdflib
import numpy as np
from cdflib import cdfwrite


DATE_RE = re.compile(r"_(\d{8})_v\d+\.cdf$")
TT2000_TOLERANCE_NS = 5_000_000
DEFAULT_TIMES_FILE = Path("CDF/custom_events/idex_dust_times_1-Jul-2026.txt")


def read_event_times(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Event time list not found: {path}")
    event_times = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not event_times:
        raise ValueError(f"No event times found in {path}")
    return event_times


def timestamp_to_tt2000(timestamp: str) -> int:
    date_part, time_part = timestamp.split("T", 1)
    year, month, day = (int(part) for part in date_part.split("-"))
    hour, minute, second_part = time_part.split(":")
    second_text, fraction = second_part.split(".", 1)
    millisecond = int(fraction[:3].ljust(3, "0"))
    microsecond = int(fraction[3:6].ljust(3, "0"))
    nanosecond = int(fraction[6:9].ljust(3, "0"))
    return int(
        cdflib.cdfepoch.compute_tt2000(
            [year, month, day, int(hour), int(minute), int(second_text), millisecond, microsecond, nanosecond]
        )
    )


def source_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse source date from {path}")
    return match.group(1)


def event_date(timestamp: str) -> str:
    return timestamp[:10].replace("-", "")


def list_sources(source_dir: Path) -> list[Path]:
    files = sorted(source_dir.glob("*.cdf"), key=source_date)
    if not files:
        raise FileNotFoundError(f"No CDF files found in {source_dir}")
    return files


def pick_source(timestamp: str, sources: list[Path]) -> Path:
    dates = [source_date(path) for path in sources]
    day = event_date(timestamp)
    candidates = [idx for idx, date in enumerate(dates) if date <= day]
    if not candidates:
        raise ValueError(f"No source file starts before {timestamp}")
    return sources[candidates[-1]]


def get_var_names(cdf: cdflib.CDF) -> list[str]:
    info = cdf.cdf_info()
    return list(info.rVariables) + list(info.zVariables)


def var_spec(varinq) -> dict:
    spec = {
        "Variable": varinq.Variable,
        "Data_Type": varinq.Data_Type,
        "Num_Elements": varinq.Num_Elements,
        "Rec_Vary": varinq.Rec_Vary,
        "Dim_Sizes": list(varinq.Dim_Sizes),
        "Var_Type": varinq.Var_Type,
        "Sparse": varinq.Sparse,
        "Compress": varinq.Compress,
        "Block_Factor": varinq.Block_Factor,
    }
    pad = getattr(varinq, "Pad", None)
    if pad is not None:
        spec["Pad"] = pad
    if varinq.Var_Type == "rVariable":
        spec["Dim_Vary"] = list(varinq.Dim_Vary)
    return spec


def concat_records(chunks: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(chunk) for chunk in chunks]
    if not arrays:
        raise ValueError("No record chunks to concatenate")
    return np.concatenate(arrays, axis=0)


def matched_records(path: Path, requested: list[tuple[int, str]]) -> list[tuple[int, str]]:
    cdf = cdflib.CDF(str(path))
    epochs = np.asarray(cdf.varget("epoch"), dtype=np.int64)
    found: list[tuple[int, str]] = []
    for tt2000, label in requested:
        deltas = np.abs(epochs - tt2000)
        index = int(np.argmin(deltas))
        if int(deltas[index]) > TT2000_TOLERANCE_NS:
            nearest = cdflib.cdfepoch.encode_tt2000(int(epochs[index]))
            raise ValueError(
                f"{path.name}: no epoch within {TT2000_TOLERANCE_NS} ns of {label}; "
                f"nearest is {nearest} at record {index}"
            )
        found.append((index, label))
    return found


def global_attrs_for_output(
    template: cdflib.CDF,
    output: Path,
    source_paths: list[Path],
    event_times: list[str],
) -> dict:
    attrs = template.globalattsget()
    logical_id = output.with_suffix("").name
    attrs["Logical_file_id"] = [logical_id]
    attrs["Start_date"] = [event_times[0][:10].replace("-", "")]
    attrs["Data_version"] = ["001"]
    attrs["Parents"] = [path.name for path in source_paths]
    return {
        name: {entry_number: value for entry_number, value in enumerate(values)}
        for name, values in attrs.items()
    }


def subset_level(source_dir: Path, output: Path, event_times: list[str]) -> None:
    sources = list_sources(source_dir)
    by_source: dict[Path, list[tuple[int, str]]] = defaultdict(list)
    for timestamp in event_times:
        by_source[pick_source(timestamp, sources)].append((timestamp_to_tt2000(timestamp), timestamp))

    record_map: dict[Path, list[tuple[int, str]]] = {}
    for source, requested in by_source.items():
        record_map[source] = matched_records(source, requested)

    used_sources = sorted(record_map, key=source_date)
    template = cdflib.CDF(str(used_sources[0]))
    info = template.cdf_info()
    cdf_spec = {
        "Majority": info.Majority,
        "Encoding": info.Encoding,
        "Checksum": info.Checksum,
        "Compressed": info.Compressed,
        "rDim_sizes": list(info.rDim_sizes),
    }

    writer = cdfwrite.CDF(str(output), cdf_spec=cdf_spec, delete=True)
    try:
        writer.write_globalattrs(global_attrs_for_output(template, output, used_sources, event_times))
        for varname in get_var_names(template):
            varinq = template.varinq(varname)
            attrs = template.varattsget(varname)
            if varinq.Rec_Vary:
                chunks = []
                for source in used_sources:
                    cdf = cdflib.CDF(str(source))
                    records = [record for record, _ in record_map[source]]
                    chunks.append(np.asarray(cdf.varget(varname))[records])
                data = concat_records(chunks)
            else:
                data = template.varget(varname)
            writer.write_var(var_spec(varinq), var_attrs=attrs, var_data=data)
    finally:
        writer.close()

    verify_output(output, event_times)


def verify_output(path: Path, event_times: list[str]) -> None:
    cdf = cdflib.CDF(str(path))
    epochs = np.asarray(cdf.varget("epoch"), dtype=np.int64)
    expected_records = len(event_times)
    if len(epochs) != expected_records:
        raise ValueError(f"{path}: expected {expected_records} records, found {len(epochs)}")
    expected = np.asarray([timestamp_to_tt2000(t) for t in event_times], dtype=np.int64)
    if not np.all(np.abs(epochs - expected) <= TT2000_TOLERANCE_NS):
        raise ValueError(f"{path}: output epochs do not match requested events")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cdf-root", type=Path, default=Path("CDF"))
    parser.add_argument("--output-dir", type=Path, default=Path("CDF/custom_events"))
    parser.add_argument("--times-file", type=Path, default=DEFAULT_TIMES_FILE)
    args = parser.parse_args()

    event_times = read_event_times(args.times_file)
    start_date = event_times[0][:10].replace("-", "")
    end_date = event_times[-1][:10].replace("-", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "l1b": args.output_dir / f"imap_idex_l1b_sci-events_{start_date}_{end_date}_v001.cdf",
        "l2a": args.output_dir / f"imap_idex_l2a_sci-events_{start_date}_{end_date}_v001.cdf",
    }
    for level, output in outputs.items():
        subset_level(args.cdf_root / level, output, event_times)
        print(output)


if __name__ == "__main__":
    main()
