import os
from pathlib import Path
import awkward as ak
import numpy as np
from typing import Iterator, Union
import attrs

# Qn vector header info class
@attrs.frozen
class QnHeaderInfo:
    event_number: int = attrs.field()

# Function to parse the Qn header line
def _parse_qn_header_line(line: str) -> QnHeaderInfo:
    """Parse Qn vector event header line."""
    values = line.split()
    if "Event" in values:
        event_number = int(values[2])  # Event number after "Event"
        return QnHeaderInfo(event_number=event_number)
    raise ValueError(f"Invalid Qn header format: {line}")

# Function to parse Qn vector events
def _parse_qn_event(f: Iterator[str]) -> Iterator[Union[QnHeaderInfo, np.ndarray]]:
    """Parse Qn vector events."""
    current_event = []
    event_header = None
    for line in f:
        stripped_line = line.strip()
        if stripped_line.startswith("#"):
            if "Event" in stripped_line and "End" not in stripped_line:
                if current_event and event_header:
                    yield event_header, np.array(current_event)
                event_header = _parse_qn_header_line(stripped_line)
                current_event = []
            elif "End" in stripped_line:
                if current_event and event_header:
                    yield event_header, np.array(current_event)
                break
        else:
            # Convert line to numeric array
            data = [float(x) if i else int(x) for i, x in enumerate(line.split())]
            current_event.append(data)

# Function to read Qn vector events in chunks
def read_qn_events_in_chunks(filename: Path, events_per_chunk: int = 10000) -> Iterator[dict]:
    """Read Qn vector events in chunks."""
    filename = Path(filename)
    with open(filename, "r") as f:
        read_lines = iter(f)
        current_chunk = {"event_headers": [], "particle_data": []}

        for header, particles in _parse_qn_event(read_lines):
            current_chunk["event_headers"].append(header.event_number)
            current_chunk["particle_data"].append(particles)

            # Save chunk once it reaches the limit
            if len(current_chunk["event_headers"]) >= events_per_chunk:
                yield current_chunk
                current_chunk = {"event_headers": [], "particle_data": []}

        if current_chunk["event_headers"]:
            yield current_chunk

# Convert chunk to awkward array and save to Parquet
def parse_qn_to_parquet(base_output_filename: str, input_filename: str, events_per_chunk: int):
    """Parse Qn vector ASCII and convert it to Parquet."""
    base_output_filename = Path(base_output_filename)
    base_output_filename.parent.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(read_qn_events_in_chunks(input_filename, events_per_chunk)):
        # Convert data to numpy array for easier slicing
        particle_data = np.concatenate(chunk["particle_data"])

        # Number of harmonic orders (based on the number of vn columns)
        n_harmonics = (particle_data.shape[1] - 8) // 4  # 4 columns per harmonic: vncos, vncos_err, vnsin, vnsin_err

        # Prepare lists for vn data
        vn_cos = []
        vn_cos_err = []
        vn_sin = []
        vn_sin_err = []

        for n in range(1, n_harmonics + 1):
            vn_cos.append(particle_data[:, 8 + (n - 1) * 4].astype(np.float32))
            vn_cos_err.append(particle_data[:, 9 + (n - 1) * 4].astype(np.float32))
            vn_sin.append(particle_data[:, 10 + (n - 1) * 4].astype(np.float32))
            vn_sin_err.append(particle_data[:, 11 + (n - 1) * 4].astype(np.float32))

        # Stack vn values into lists grouped by events
        vn_cos_grouped = np.column_stack(vn_cos)
        vn_cos_err_grouped = np.column_stack(vn_cos_err)
        vn_sin_grouped = np.column_stack(vn_sin)
        vn_sin_err_grouped = np.column_stack(vn_sin_err)

        # Extract dN (last column)
        dN_column_index = 8 + n_harmonics * 4
        dN = particle_data[:, dN_column_index].astype(np.int32)

        # Create awkward array with all information
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
            "vn_cos": ak.from_numpy(vn_cos_grouped),
            "vn_cos_err": ak.from_numpy(vn_cos_err_grouped),
            "vn_sin": ak.from_numpy(vn_sin_grouped),
            "vn_sin_err": ak.from_numpy(vn_sin_err_grouped),
            "dN": dN,
        })

        # Save the Parquet file
        output_filename = base_output_filename.with_name(f"{base_output_filename.stem}_{i:02}.parquet")
        ak.to_parquet(ak_array, str(output_filename), compression="zstd")

        # print(f"Saved chunk {i + 1} to {output_filename}")

