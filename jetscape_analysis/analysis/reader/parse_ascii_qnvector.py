import os
from pathlib import Path
import awkward as ak
import numpy as np
from typing import Iterator, Union
import attrs

@attrs.frozen
class QnHeaderInfo:
    event_number: int = attrs.field()

def _parse_qn_header_line(line: str) -> QnHeaderInfo:
    """Parse Qn vector event header line."""
    values = line.split()
    if "Event" in values:
        event_number = int(values[2])  # Event number after "Event"
        return QnHeaderInfo(event_number=event_number)
    raise ValueError(f"Invalid Qn header format: {line}")

def _parse_qn_event(f: Iterator[str]) -> Iterator[Union[QnHeaderInfo, np.ndarray]]:
    """Parse Qn vector events."""
    current_event = []
    event_header = None
    for line in f:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            if "Event" in stripped_line and "End" not in stripped_line:
                if current_event and event_header:
                    yield event_header, np.array(current_event, dtype=np.float32)
                event_header = _parse_qn_header_line(stripped_line)
                current_event = []
            elif "End" in stripped_line:
                if current_event and event_header:
                    yield event_header, np.array(current_event, dtype=np.float32)
                break
        else:
            data = [float(x) if i else int(x) for i, x in enumerate(line.split())]
            current_event.append(data)

def read_qn_events_in_chunks(filename: Path, events_per_chunk: int = 10000) -> Iterator[dict]:
    """Read Qn vector events in chunks."""
    filename = Path(filename)
    with open(filename, "r") as f:
        read_lines = iter(f)
        current_chunk = {"event_headers": [], "particle_data": []}
        event_count = 0

        for header, particles in _parse_qn_event(read_lines):
            # Directly append without creating intermediate lists
            current_chunk["event_headers"].append(header.event_number)
            current_chunk["particle_data"].append(particles)

            event_count += 1
            if event_count >= events_per_chunk:
                yield current_chunk
                current_chunk = {"event_headers": [], "particle_data": []}
                event_count = 0

        if current_chunk["event_headers"]:
            yield current_chunk

def parse_qn_to_parquet(base_output_filename: str, input_filename: str, events_per_chunk: int):
    """Parse Qn vector ASCII and convert it to Parquet."""
    base_output_filename = Path(base_output_filename)
    base_output_filename.parent.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(read_qn_events_in_chunks(input_filename, events_per_chunk)):
        # Directly stack without creating intermediate lists
        particle_data = np.vstack(chunk["particle_data"])

        n_harmonics = (particle_data.shape[1] - 8) // 4

        # Pre-allocate arrays for vn data to avoid intermediate list creation
        vn_cos = np.zeros((particle_data.shape[0], n_harmonics), dtype=np.float32)
        vn_cos_err = np.zeros((particle_data.shape[0], n_harmonics), dtype=np.float32)
        vn_sin = np.zeros((particle_data.shape[0], n_harmonics), dtype=np.float32)
        vn_sin_err = np.zeros((particle_data.shape[0], n_harmonics), dtype=np.float32)

        for n in range(n_harmonics):
            vn_cos[:, n] = particle_data[:, 8 + n * 4]
            vn_cos_err[:, n] = particle_data[:, 9 + n * 4]
            vn_sin[:, n] = particle_data[:, 10 + n * 4]
            vn_sin_err[:, n] = particle_data[:, 11 + n * 4]

        dN_column_index = 8 + n_harmonics * 4
        dN = particle_data[:, dN_column_index].astype(np.int32)

        # Efficiently use Awkward Array for structured data
        ak_array = ak.Array({
            "event_ID": np.repeat(chunk["event_headers"], [len(p) for p in chunk["particle_data"]]),
            "pid": particle_data[:, 0].astype(np.int32),
            "pT": particle_data[:, 1].astype(np.float32),
            "pT_err": particle_data[:, 2].astype(np.float32),
            "y": particle_data[:, 3].astype(np.float32),
            "y_err": particle_data[:, 4].astype(np.float32),
            "ET": particle_data[:, 5].astype(np.float32),
            "dNdpTdy": particle_data[:, 6].astype(np.float32),
            "dNdpTdy_err": particle_data[:, 7].astype(np.float32),
            "vn_cos": ak.from_numpy(vn_cos),
            "vn_cos_err": ak.from_numpy(vn_cos_err),
            "vn_sin": ak.from_numpy(vn_sin),
            "vn_sin_err": ak.from_numpy(vn_sin_err),
            "dN": dN,
        })

        output_filename = base_output_filename.with_name(f"{base_output_filename.stem}_{i:02}.parquet")
        ak.to_parquet(ak_array, str(output_filename), compression="zstd")
