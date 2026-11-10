"""
Step 3: Inverse Dynamics — joint torques from .mot + scaled .osim model.

Inputs:
  - Subject-scaled .osim model  (outputs/scaled/<subject>/<subject>_scaled.osim)
  - OpenSim IK .mot file        (outputs/ik/<subject>/<exercise>.mot)

Outputs:
  - Joint torques .sto file     (outputs/id/<subject>/<exercise>_id.sto)
  - Parsed dict                 {coord_name: np.ndarray shape (N,)} in N·m or N

Pipeline position:
  Step 1 (scale_runner) → Step 2 (ik_runner) → Step 3 (id_runner) → Step 4 (rl_data)

Usage:
  python src/opensim/id_runner.py \
      --fit3d  /path/to/fit3d_root \
      --scaled outputs/scaled \
      --ik     outputs/ik \
      --output outputs/id \
      --subjects s03 s04 \
      --exercise squat

OpenSim Python bindings must be on PYTHONPATH.  See:
  https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIT3D_FPS: float = 50.0

# Coordinates produced by ik_runner that have physical torque meaning.
# Translational DOFs (pelvis_tx/ty/tz) produce forces (N), not torques (N·m).
TORQUE_COORDS: Tuple[str, ...] = (
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_r",
    "knee_angle_l",
    "ankle_angle_r",
    "ankle_angle_l",
    "subtalar_angle_r",
    "subtalar_angle_l",
    "lumbar_extension",
    "lumbar_bending",
    "lumbar_rotation",
)

FORCE_COORDS: Tuple[str, ...] = (
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
)

# ---------------------------------------------------------------------------
# OpenSim import helper
# ---------------------------------------------------------------------------

def _import_opensim():
    """Import opensim; raise ImportError with a clear message if missing."""
    try:
        import opensim as osim  # type: ignore
        return osim
    except ImportError as exc:
        raise ImportError(
            "OpenSim Python bindings not found. "
            "Install via conda: conda install -c opensim-org opensim=4.5"
        ) from exc


# ---------------------------------------------------------------------------
# .sto parser (no OpenSim dependency)
# ---------------------------------------------------------------------------

def parse_sto(sto_path: str | Path) -> Dict[str, np.ndarray]:
    """Parse an OpenSim Storage (.sto) file into a dict of numpy arrays.

    Args:
        sto_path: Path to the .sto file.

    Returns:
        Dict mapping column name → 1-D float64 array (length = number of rows).
        The 'time' column is always present.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed.
    """
    path = Path(sto_path)
    if not path.exists():
        raise FileNotFoundError(f"STO file not found: {path}")

    lines: List[str] = path.read_text(encoding="utf-8").splitlines()

    # Find endheader
    header_end = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break
    if header_end == -1:
        raise ValueError(f"No 'endheader' found in {path}")

    # Column names are on the line immediately after endheader
    col_line = header_end + 1
    if col_line >= len(lines):
        raise ValueError(f"No column header after 'endheader' in {path}")

    columns = lines[col_line].split("\t")
    data_lines = [l for l in lines[col_line + 1:] if l.strip()]

    if not data_lines:
        raise ValueError(f"No data rows found in {path}")

    rows = np.array(
        [[float(v) for v in row.split("\t")] for row in data_lines],
        dtype=np.float64,
    )

    if rows.shape[1] != len(columns):
        raise ValueError(
            f"Column count mismatch: header has {len(columns)} but "
            f"data rows have {rows.shape[1]} in {path}"
        )

    return {col: rows[:, i] for i, col in enumerate(columns)}


# ---------------------------------------------------------------------------
# ID XML setup builder
# ---------------------------------------------------------------------------

def _build_id_setup_xml(
    model_path: str,
    mot_path: str,
    output_sto: str,
    t_start: float,
    t_end: float,
    low_pass_cutoff_hz: float = 6.0,
) -> str:
    """Return an InverseDynamicsTool XML setup string.

    Args:
        model_path:          Absolute path to the scaled .osim model.
        mot_path:            Absolute path to the IK .mot file.
        output_sto:          Absolute path for the output .sto file.
        t_start:             Start time (seconds).
        t_end:               End time (seconds).
        low_pass_cutoff_hz:  Butterworth low-pass cutoff for kinematics smoothing.
                             6 Hz is standard for biomechanics at 50 Hz capture.

    Returns:
        XML string ready to write to a .xml file.
    """
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8" ?>
        <OpenSimDocument Version="40000">
            <InverseDynamicsTool name="id_tool">
                <model_file>{model_path}</model_file>
                <coordinates_file>{mot_path}</coordinates_file>
                <output_gen_force_file>{output_sto}</output_gen_force_file>
                <results_directory>{str(Path(output_sto).parent)}</results_directory>
                <time_range> {t_start:.6f} {t_end:.6f} </time_range>
                <lowpass_cutoff_frequency_for_coordinates>{low_pass_cutoff_hz}</lowpass_cutoff_frequency_for_coordinates>
                <joints_to_report_body_forces />
            </InverseDynamicsTool>
        </OpenSimDocument>
    """)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_id(
    model_path: str | Path,
    mot_path: str | Path,
    output_sto: str | Path,
    low_pass_cutoff_hz: float = 6.0,
) -> Dict[str, np.ndarray]:
    """Run OpenSim Inverse Dynamics for one (model, mot) pair.

    Args:
        model_path:          Path to scaled .osim model.
        mot_path:            Path to IK .mot file from ik_runner.
        output_sto:          Destination path for the .sto torque file.
        low_pass_cutoff_hz:  Low-pass cutoff for coordinate smoothing (default 6 Hz).

    Returns:
        Dict mapping coordinate name → generalised force array (N·m or N).

    Raises:
        FileNotFoundError: If model or mot file is missing.
        RuntimeError:      If OpenSim ID fails.
    """
    model_path = Path(model_path).resolve()
    mot_path = Path(mot_path).resolve()
    output_sto = Path(output_sto).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Scaled model not found: {model_path}")
    if not mot_path.exists():
        raise FileNotFoundError(f"IK .mot file not found: {mot_path}")

    output_sto.parent.mkdir(parents=True, exist_ok=True)

    # Determine time range from the .mot file
    mot_data = parse_sto(mot_path)
    t_start = float(mot_data["time"][0])
    t_end = float(mot_data["time"][-1])
    n_frames = len(mot_data["time"])
    log.info(
        "ID  model=%s  mot=%s  frames=%d  t=%.3f–%.3fs",
        model_path.name, mot_path.name, n_frames, t_start, t_end,
    )

    osim = _import_opensim()

    # Write setup XML to a temp file
    xml_str = _build_id_setup_xml(
        model_path=str(model_path),
        mot_path=str(mot_path),
        output_sto=str(output_sto),
        t_start=t_start,
        t_end=t_end,
        low_pass_cutoff_hz=low_pass_cutoff_hz,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_id_setup.xml", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(xml_str)
        setup_path = tmp.name

    try:
        tool = osim.InverseDynamicsTool(setup_path)
        ok = tool.run()
        if not ok:
            raise RuntimeError(
                f"InverseDynamicsTool.run() returned False for {mot_path}"
            )
    finally:
        os.unlink(setup_path)

    if not output_sto.exists():
        raise RuntimeError(
            f"ID ran without error but output not found: {output_sto}"
        )

    torques = parse_sto(output_sto)
    log.info(
        "ID done  coords=%d  rows=%d  → %s",
        len(torques) - 1,  # exclude time
        len(torques["time"]),
        output_sto.name,
    )
    return torques


# ---------------------------------------------------------------------------
# Subject-level pipeline
# ---------------------------------------------------------------------------

def process_subject(
    subject_id: str,
    scaled_dir: str | Path,
    ik_dir: str | Path,
    output_dir: str | Path,
    exercises: Optional[List[str]] = None,
    low_pass_cutoff_hz: float = 6.0,
) -> List[Path]:
    """Run ID for all exercises of one subject.

    Args:
        subject_id:          e.g. "s03".
        scaled_dir:          Root of Step 1 scaled models (scaled/<subject>/).
        ik_dir:              Root of Step 2 .mot files   (ik/<subject>/).
        output_dir:          Root for ID outputs          (id/<subject>/).
        exercises:           List of exercise names; None → process all found .mot files.
        low_pass_cutoff_hz:  Passed through to run_id.

    Returns:
        List of Path objects pointing to the written .sto files.
    """
    scaled_dir = Path(scaled_dir)
    ik_dir = Path(ik_dir)
    output_dir = Path(output_dir)

    model_path = scaled_dir / subject_id / f"{subject_id}_scaled.osim"
    mot_root = ik_dir / subject_id
    out_root = output_dir / subject_id
    out_root.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        log.error("Scaled model missing: %s — skipping %s", model_path, subject_id)
        return []

    if not mot_root.exists():
        log.error("IK directory missing: %s — skipping %s", mot_root, subject_id)
        return []

    # Discover .mot files
    if exercises:
        mot_files = [mot_root / f"{ex}.mot" for ex in exercises]
    else:
        mot_files = sorted(mot_root.glob("*.mot"))

    if not mot_files:
        log.warning("No .mot files found for %s", subject_id)
        return []

    written: List[Path] = []
    for mot_path in mot_files:
        if not mot_path.exists():
            log.warning("MOT file not found: %s — skipping", mot_path)
            continue
        exercise_name = mot_path.stem
        output_sto = out_root / f"{exercise_name}_id.sto"
        try:
            run_id(
                model_path=model_path,
                mot_path=mot_path,
                output_sto=output_sto,
                low_pass_cutoff_hz=low_pass_cutoff_hz,
            )
            written.append(output_sto)
        except Exception as exc:
            log.error(
                "ID failed for %s / %s: %s", subject_id, exercise_name, exc
            )

    return written


# ---------------------------------------------------------------------------
# Sanity check (no OpenSim needed)
# ---------------------------------------------------------------------------

def run_sanity_check(sto_path: str | Path) -> None:
    """Print a quick summary of an ID .sto output file."""
    data = parse_sto(sto_path)
    n_frames = len(data["time"])
    fps = 1.0 / float(np.mean(np.diff(data["time"])))
    print(f"\n--- ID sanity check: {Path(sto_path).name} ---")
    print(f"  frames : {n_frames}")
    print(f"  fps    : {fps:.1f} Hz")
    print(f"  duration: {data['time'][-1]:.2f} s")
    print(f"  coords : {list(data.keys())}")
    for coord in TORQUE_COORDS:
        if coord in data:
            arr = data[coord]
            print(
                f"  {coord:<30s}  min={arr.min():8.2f}  max={arr.max():8.2f}  "
                f"mean={arr.mean():8.2f}  N·m"
            )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 3: Inverse Dynamics — .mot + scaled .osim → joint torques .sto"
    )
    p.add_argument(
        "--scaled",
        required=True,
        help="Root directory of scaled .osim models (outputs/scaled/)",
    )
    p.add_argument(
        "--ik",
        required=True,
        help="Root directory of IK .mot files (outputs/ik/)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Root directory for ID .sto outputs (outputs/id/)",
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        default=["s03", "s04", "s05", "s06", "s07", "s08", "s09", "s11"],
        help="Subject IDs to process (default: all 8 Fit3D subjects)",
    )
    p.add_argument(
        "--exercise",
        dest="exercises",
        nargs="+",
        default=None,
        help="Exercise name(s) to process (default: all .mot files found)",
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Low-pass cutoff frequency in Hz for coordinate smoothing (default: 6.0)",
    )
    p.add_argument(
        "--sanity-check",
        metavar="STO_PATH",
        default=None,
        help="Parse and print summary of an existing .sto file, then exit",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.sanity_check:
        run_sanity_check(args.sanity_check)
        return

    total_written: List[Path] = []
    for subject_id in args.subjects:
        written = process_subject(
            subject_id=subject_id,
            scaled_dir=args.scaled,
            ik_dir=args.ik,
            output_dir=args.output,
            exercises=args.exercises,
            low_pass_cutoff_hz=args.cutoff,
        )
        total_written.extend(written)
        if written:
            log.info(
                "Subject %s: wrote %d .sto file(s)", subject_id, len(written)
            )

    log.info("Done. Total .sto files written: %d", len(total_written))


if __name__ == "__main__":
    main()
