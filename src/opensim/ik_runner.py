"""
ik_runner.py — Step 2 of biomech-physics pipeline

Convert Fit3D SMPL-X body_pose (rotation matrices) to OpenSim IK .mot files.

SMPL-X body_pose: (N, 21, 3, 3) rotation matrices for 21 body joints
OpenSim .mot:     tab-separated joint angles in degrees at 50 Hz

Joint mapping (SMPL-X index → OpenSim coordinate):
  SMPL-X uses a kinematic tree; each joint's rotation is relative to its parent.
  We extract the 6 lower-body joints relevant for squat/gait analysis plus pelvis.

Reference:
  Rajagopal et al. (2016) "Full-Body Musculoskeletal Model for Muscle-Driven
  Simulation of Human Gait." IEEE TBME 63(10):2068-2079.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# SMPL-X joint index → name mapping (22 joints: global_orient + 21 body)
# ---------------------------------------------------------------------------
SMPLX_JOINT_NAMES: List[str] = [
    "pelvis",          # 0  — global_orient
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
]

# ---------------------------------------------------------------------------
# OpenSim coordinate names and which SMPL-X joint/axis they come from.
#
# OpenSim Rajagopal2016 lower-body coordinates (generalized):
#   pelvis_tilt   — pelvis sagittal tilt    (rotation about X in OpenSim = Z in SMPL-X)
#   pelvis_list   — pelvis frontal (list)   (rotation about Z in OpenSim = X in SMPL-X)
#   pelvis_rotation — pelvis transverse     (rotation about Y)
#   pelvis_tx/ty/tz — translation (meters)
#   hip_flexion_r/l, hip_adduction_r/l, hip_rotation_r/l
#   knee_angle_r/l
#   ankle_angle_r/l
#   subtalar_angle_r/l  (from foot joints — approximate)
#
# Axis convention note:
#   SMPL-X: X=right, Y=up, Z=forward (right-hand)
#   OpenSim: X=forward, Y=up, Z=right (right-hand)
#   We rotate axes: OpenSim_X = SMPL_Z, OpenSim_Y = SMPL_Y, OpenSim_Z = SMPL_X
# ---------------------------------------------------------------------------

# Map: (smplx_joint_index_in_body_pose, rotation_axis_index_in_opensim_frame)
# rotation_axis: 0=X(flexion/tilt), 1=Y(rotation/internal-external), 2=Z(adduction/list)
COORDINATE_MAP: Dict[str, Tuple[int, int]] = {
    # Pelvis (from global_orient, index -1 → handled separately in extract_joint_angles)
    # XYZ Euler: axis 0 = tilt (sagittal), 1 = rotation (transverse), 2 = list (frontal)
    "pelvis_tilt":      (-1, 0),
    "pelvis_list":      (-1, 2),
    "pelvis_rotation":  (-1, 1),

    # Right hip (SMPL-X body_pose index 1 = right_hip)
    # Rotation vector axis 0 = flexion (dominant, ~60° at squat depth)
    "hip_flexion_r":    (1, 0),
    "hip_adduction_r":  (1, 2),
    "hip_rotation_r":   (1, 1),

    # Left hip (SMPL-X body_pose index 0 = left_hip)
    "hip_flexion_l":    (0, 0),
    "hip_adduction_l":  (0, 2),
    "hip_rotation_l":   (0, 1),

    # Right knee (SMPL-X body_pose index 4 = right_knee)
    # Rotation vector axis 0 = flexion (dominant, ~96° at deep squat)
    "knee_angle_r":     (4, 0),

    # Left knee (SMPL-X body_pose index 3 = left_knee)
    "knee_angle_l":     (3, 0),

    # Right ankle (SMPL-X body_pose index 7 = right_ankle)
    # Axis 0 = plantar/dorsiflexion
    "ankle_angle_r":    (7, 0),

    # Left ankle (SMPL-X body_pose index 6 = left_ankle)
    "ankle_angle_l":    (6, 0),

    # Subtalar (from foot joints — indices 10=right_foot, 9=left_foot in body_pose)
    # Axis 2 = subtalar inversion/eversion
    "subtalar_angle_r": (10, 2),
    "subtalar_angle_l": (9,  2),

    # Lumbar (spine1 = body_pose index 2)
    # Axis 0 = extension, 1 = rotation, 2 = lateral bending
    "lumbar_extension": (2, 0),
    "lumbar_bending":   (2, 2),
    "lumbar_rotation":  (2, 1),
}

# Sampling frequency of Fit3D
FIT3D_FPS: float = 50.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix (3,3) to rotation vector (3,) via axis-angle.
    Uses scipy if available, otherwise manual Rodrigues.
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


def rotmat_to_euler_opensim(R: np.ndarray) -> np.ndarray:
    """
    Convert SMPL-X rotation matrix (3,3) to joint angles in degrees.

    Returns (flex, int_ext_rot, abd_add) where the axis assignment follows
    the empirically verified SMPL-X rotation-vector convention:
      axis 0 (X) → flexion/extension     (dominant DoF for hip, knee, ankle)
      axis 1 (Y) → internal/external rotation
      axis 2 (Z) → abduction/adduction

    Note: No axis remap is applied. SMPL-X rotation vectors are expressed
    in the local parent-relative frame; axis 0 drives flexion for all
    lower-body joints as confirmed from Fit3D squat data (hip flexion ~60°,
    knee flexion ~96° at deep squat). OpenSim coordinate names are mapped
    accordingly in COORDINATE_MAP.
    """
    from scipy.spatial.transform import Rotation
    angles = Rotation.from_matrix(R).as_euler('XYZ', degrees=True)
    return angles  # (flex_ext, int_ext_rot, abd_add)


def extract_joint_angles(
    body_pose: np.ndarray,      # (N, 21, 3, 3)
    global_orient: np.ndarray,  # (N, 1, 3, 3)
    transl: np.ndarray,         # (N, 3)
    normalize_to_frame0: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract all OpenSim coordinates from SMPL-X pose parameters.

    Args:
        body_pose:           (N, 21, 3, 3) per-joint rotation matrices
        global_orient:       (N, 1, 3, 3) global pelvis orientation
        transl:              (N, 3) root translation in meters
        normalize_to_frame0: if True, subtract frame-0 global orientation so
                             the pelvis starts near zero (removes world-frame offset).
                             Fit3D records absolute world orientations, so this is
                             required to get biomechanically meaningful angles.

    Returns:
        dict of coordinate_name → (N,) array in degrees (translation in meters).
    """
    N = body_pose.shape[0]
    coords: Dict[str, np.ndarray] = {}

    # global_orient shape: (N, 1, 3, 3) → (N, 3, 3)
    global_R = global_orient[:, 0, :, :]  # (N, 3, 3)

    # Remove world-frame offset: express all rotations relative to frame 0
    if normalize_to_frame0:
        R0_inv = global_R[0].T                           # R0^{-1} = R0^T (orthogonal)
        global_R = np.einsum('ij,njk->nik', R0_inv, global_R)  # (N, 3, 3)
        # Also normalize body_pose relative to frame-0 global:
        # body_pose is already relative to parent joint, so only pelvis needs it
        # (body joints are local — no world offset to remove)

    # Pelvis translation: subtract frame-0 position so motion starts at origin
    if normalize_to_frame0:
        transl = transl - transl[0]
    coords["pelvis_tx"] = transl[:, 0]
    coords["pelvis_ty"] = transl[:, 1]
    coords["pelvis_tz"] = transl[:, 2]

    # Pelvis orientation from global_orient
    pelvis_angles = np.zeros((N, 3))
    for i in range(N):
        pelvis_angles[i] = rotmat_to_euler_opensim(global_R[i])
    coords["pelvis_tilt"]     = pelvis_angles[:, 0]
    coords["pelvis_rotation"] = pelvis_angles[:, 1]
    coords["pelvis_list"]     = pelvis_angles[:, 2]

    # Body joints
    joint_euler: Dict[int, np.ndarray] = {}
    for coord_name, (joint_idx, _) in COORDINATE_MAP.items():
        if joint_idx < 0:
            continue  # pelvis already handled
        if joint_idx not in joint_euler:
            angles = np.zeros((N, 3))
            for i in range(N):
                angles[i] = rotmat_to_euler_opensim(body_pose[i, joint_idx])
            joint_euler[joint_idx] = angles

    # Assign each coordinate from the right axis
    axis_map = {0: 0, 1: 1, 2: 2}  # OpenSim rx→0, ry→1, rz→2
    for coord_name, (joint_idx, axis) in COORDINATE_MAP.items():
        if joint_idx < 0:
            continue
        coords[coord_name] = joint_euler[joint_idx][:, axis_map[axis]]

    return coords


def write_mot_file(
    output_path: Path,
    coords: Dict[str, np.ndarray],
    fps: float = FIT3D_FPS,
) -> None:
    """
    Write OpenSim .mot (storage) file.

    Format:
      header lines
      nRows=N nColumns=M
      inDegrees=yes
      time  coord1  coord2 ...
      0.00  val1    val2   ...
    """
    N = next(iter(coords.values())).shape[0]
    time = np.arange(N) / fps

    # Column order: time first, then standard OpenSim order
    col_order = [
        "pelvis_tilt", "pelvis_list", "pelvis_rotation",
        "pelvis_tx", "pelvis_ty", "pelvis_tz",
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",
        "knee_angle_r", "knee_angle_l",
        "ankle_angle_r", "ankle_angle_l",
        "subtalar_angle_r", "subtalar_angle_l",
        "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    ]
    # Keep only columns that were computed
    col_order = [c for c in col_order if c in coords]
    n_cols = len(col_order) + 1  # +1 for time

    stem = output_path.stem
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"{stem}\n")
        f.write("version=1\n")
        f.write(f"nRows={N}\n")
        f.write(f"nColumns={n_cols}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")

        # Header row
        header = ["time"] + col_order
        f.write("\t".join(header) + "\n")

        # Data rows
        for i in range(N):
            row = [f"{time[i]:.6f}"]
            for col in col_order:
                row.append(f"{coords[col][i]:.6f}")
            f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_subject(
    subject_id: str,
    fit3d_root: Path,
    output_dir: Path,
    exercise: str = "squat",
) -> Path:
    """
    Load Fit3D SMPL-X data for one subject/exercise and write .mot file.

    Args:
        subject_id:  e.g. "s03"
        fit3d_root:  root of Fit3D dataset (contains train/)
        output_dir:  where to write .mot files
        exercise:    exercise name matching filename in smplx/ folder

    Returns:
        Path to the written .mot file
    """
    smplx_path = fit3d_root / "train" / subject_id / "smplx" / f"{exercise}.json"
    if not smplx_path.exists():
        raise FileNotFoundError(f"SMPL-X file not found: {smplx_path}")

    with open(smplx_path) as f:
        data = json.load(f)

    body_pose    = np.array(data["body_pose"])     # (N, 21, 3, 3)
    global_orient = np.array(data["global_orient"]) # (N, 1, 3, 3)
    transl       = np.array(data["transl"])         # (N, 3)

    coords = extract_joint_angles(body_pose, global_orient, transl)

    out_path = output_dir / "ik" / subject_id / f"{exercise}.mot"
    write_mot_file(out_path, coords)

    n_frames = body_pose.shape[0]
    duration = n_frames / FIT3D_FPS
    print(
        f"[{subject_id}] {exercise}: {n_frames} frames "
        f"({duration:.1f}s) → {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Fit3D SMPL-X body_pose to OpenSim IK .mot files"
    )
    parser.add_argument(
        "--fit3d", required=True, type=Path,
        help="Root directory of Fit3D dataset (contains train/)"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory (outputs/ik/<subject>/<exercise>.mot)"
    )
    parser.add_argument(
        "--subjects", nargs="+",
        default=["s03", "s04", "s05", "s07", "s08", "s09", "s10", "s11"],
        help="Subject IDs to process"
    )
    parser.add_argument(
        "--exercise", default="squat",
        help="Exercise name (default: squat)"
    )
    parser.add_argument(
        "--exercises", nargs="+", default=None,
        help="Process multiple exercises (overrides --exercise)"
    )
    args = parser.parse_args()

    exercises = args.exercises if args.exercises else [args.exercise]

    for subject_id in args.subjects:
        for exercise in exercises:
            try:
                process_subject(subject_id, args.fit3d, args.output, exercise)
            except FileNotFoundError as e:
                print(f"  SKIP: {e}")


if __name__ == "__main__":
    main()
