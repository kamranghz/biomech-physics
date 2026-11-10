"""
tests/test_ik_runner.py — Unit tests for Step 2: SMPL-X → OpenSim IK .mot

Tests run without Fit3D data (synthetic rotation matrices).
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

import sys, os
# When running from tests/ inside the repo, go up one level to find src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.opensim.ik_runner import (
    rotmat_to_euler_opensim,
    extract_joint_angles,
    write_mot_file,
    FIT3D_FPS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def identity_pose(N: int = 10):
    """Return identity body_pose (N,21,3,3), global_orient (N,1,3,3), transl (N,3)."""
    body_pose    = np.tile(np.eye(3), (N, 21, 1, 1))
    global_orient = np.tile(np.eye(3), (N, 1, 1, 1))
    transl       = np.zeros((N, 3))
    return body_pose, global_orient, transl


def rotation_matrix_x(angle_deg: float) -> np.ndarray:
    """Pure X-axis rotation matrix."""
    a = np.radians(angle_deg)
    return np.array([
        [1,          0,           0],
        [0,  np.cos(a), -np.sin(a)],
        [0,  np.sin(a),  np.cos(a)],
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRotmatToEuler:
    def test_identity_gives_zero_angles(self):
        angles = rotmat_to_euler_opensim(np.eye(3))
        np.testing.assert_allclose(angles, [0., 0., 0.], atol=1e-10)

    def test_pure_x_rotation_captured_in_axis0(self):
        """A pure X rotation should appear in axis 0 (flexion)."""
        R = rotation_matrix_x(45.0)
        angles = rotmat_to_euler_opensim(R)
        assert abs(angles[0] - 45.0) < 0.1, f"Expected ~45° in axis 0, got {angles}"
        assert abs(angles[1]) < 1.0, "Axis 1 should be near zero"
        assert abs(angles[2]) < 1.0, "Axis 2 should be near zero"

    def test_output_shape(self):
        angles = rotmat_to_euler_opensim(np.eye(3))
        assert angles.shape == (3,)


class TestExtractJointAngles:
    def test_identity_pose_zero_angles(self):
        """Identity rotations (anatomical pose) should give near-zero joint angles."""
        body_pose, global_orient, transl = identity_pose(5)
        coords = extract_joint_angles(body_pose, global_orient, transl)
        for name in ["knee_angle_r", "knee_angle_l", "hip_flexion_r", "hip_flexion_l"]:
            np.testing.assert_allclose(
                coords[name], 0.0, atol=1e-8,
                err_msg=f"{name} should be zero for identity pose"
            )

    def test_knee_flexion_captured(self):
        """A 90° X-axis rotation on the right knee joint should give ~90° knee_angle_r."""
        body_pose, global_orient, transl = identity_pose(3)
        R90 = rotation_matrix_x(90.0)
        body_pose[:, 4] = R90   # right knee = body_pose index 4
        coords = extract_joint_angles(body_pose, global_orient, transl)
        np.testing.assert_allclose(coords["knee_angle_r"], 90.0, atol=0.5)

    def test_normalization_removes_frame0_offset(self):
        """With normalize_to_frame0=True, pelvis starts at zero."""
        body_pose, global_orient, transl = identity_pose(5)
        # Apply a constant global orient offset
        R_offset = rotation_matrix_x(45.0)
        global_orient[:, 0] = R_offset
        coords = extract_joint_angles(body_pose, global_orient, transl, normalize_to_frame0=True)
        np.testing.assert_allclose(coords["pelvis_tilt"][0], 0.0, atol=1e-6)

    def test_all_expected_coordinates_present(self):
        body_pose, global_orient, transl = identity_pose(2)
        coords = extract_joint_angles(body_pose, global_orient, transl)
        required = [
            "pelvis_tilt", "pelvis_list", "pelvis_rotation",
            "pelvis_tx", "pelvis_ty", "pelvis_tz",
            "hip_flexion_r", "hip_flexion_l",
            "knee_angle_r", "knee_angle_l",
            "ankle_angle_r", "ankle_angle_l",
        ]
        for name in required:
            assert name in coords, f"Missing coordinate: {name}"

    def test_translation_normalized_to_frame0(self):
        body_pose, global_orient, transl = identity_pose(5)
        transl[:] = np.array([1.0, 2.0, 3.0])  # constant offset
        coords = extract_joint_angles(body_pose, global_orient, transl, normalize_to_frame0=True)
        np.testing.assert_allclose(coords["pelvis_tx"], 0.0, atol=1e-10)
        np.testing.assert_allclose(coords["pelvis_ty"], 0.0, atol=1e-10)
        np.testing.assert_allclose(coords["pelvis_tz"], 0.0, atol=1e-10)

    def test_N_frames_preserved(self):
        N = 50
        body_pose, global_orient, transl = identity_pose(N)
        coords = extract_joint_angles(body_pose, global_orient, transl)
        for v in coords.values():
            assert v.shape == (N,), f"Expected shape ({N},), got {v.shape}"


class TestWriteMotFile:
    def test_mot_file_header(self):
        body_pose, global_orient, transl = identity_pose(10)
        coords = extract_joint_angles(body_pose, global_orient, transl)
        with tempfile.NamedTemporaryFile(suffix=".mot", delete=False) as f:
            path = Path(f.name)
        write_mot_file(path, coords)
        content = path.read_text()
        assert "version=1" in content
        assert "inDegrees=yes" in content
        assert "endheader" in content
        assert "nRows=10" in content

    def test_mot_file_time_column(self):
        N = 5
        body_pose, global_orient, transl = identity_pose(N)
        coords = extract_joint_angles(body_pose, global_orient, transl)
        with tempfile.NamedTemporaryFile(suffix=".mot", delete=False) as f:
            path = Path(f.name)
        write_mot_file(path, coords, fps=FIT3D_FPS)
        lines = path.read_text().splitlines()
        # Find data start (after endheader + col header)
        data_start = next(i for i, l in enumerate(lines) if l.startswith("time")) + 1
        times = [float(lines[data_start + i].split("\t")[0]) for i in range(N)]
        expected = [i / FIT3D_FPS for i in range(N)]
        np.testing.assert_allclose(times, expected, atol=1e-5)

    def test_mot_file_correct_row_count(self):
        N = 20
        body_pose, global_orient, transl = identity_pose(N)
        coords = extract_joint_angles(body_pose, global_orient, transl)
        with tempfile.NamedTemporaryFile(suffix=".mot", delete=False) as f:
            path = Path(f.name)
        write_mot_file(path, coords)
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        data_start = next(i for i, l in enumerate(lines) if l.startswith("time")) + 1
        data_rows = lines[data_start:]
        assert len(data_rows) == N
