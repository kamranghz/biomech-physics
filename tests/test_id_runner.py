"""
Unit tests for src/opensim/id_runner.py
Tests cover parse_sto, _build_id_setup_xml, and run_sanity_check.
OpenSim itself is not required; run_id / process_subject are integration-only.

Run from repo root:
    python -m pytest tests/test_id_runner.py -v
"""

import io
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.opensim.id_runner import (
    FIT3D_FPS,
    TORQUE_COORDS,
    FORCE_COORDS,
    _build_id_setup_xml,
    parse_sto,
    run_sanity_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sto(columns, rows, extra_header="") -> str:
    """Build a minimal .sto file string."""
    header = textwrap.dedent(f"""\
        Results
        version=1
        nRows={len(rows)}
        nColumns={len(columns)}
        inDegrees=no
        {extra_header}
        endheader
    """)
    col_line = "\t".join(columns)
    data_lines = "\n".join("\t".join(f"{v:.6f}" for v in row) for row in rows)
    return header + col_line + "\n" + data_lines + "\n"


def _write_sto(tmp_dir, columns, rows):
    path = Path(tmp_dir) / "test.sto"
    path.write_text(_make_sto(columns, rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# TestParseSto
# ---------------------------------------------------------------------------

class TestParseSto:
    def test_basic_parse(self, tmp_path):
        cols = ["time", "hip_flexion_r", "knee_angle_r"]
        rows = [[0.0, 10.0, 20.0], [0.02, 11.0, 22.0], [0.04, 12.0, 24.0]]
        path = _write_sto(tmp_path, cols, rows)
        data = parse_sto(path)
        assert set(data.keys()) == set(cols)
        np.testing.assert_allclose(data["time"], [0.0, 0.02, 0.04])
        np.testing.assert_allclose(data["knee_angle_r"], [20.0, 22.0, 24.0])

    def test_all_arrays_float64(self, tmp_path):
        cols = ["time", "hip_flexion_r"]
        rows = [[i * 0.02, float(i)] for i in range(10)]
        path = _write_sto(tmp_path, cols, rows)
        data = parse_sto(path)
        assert data["time"].dtype == np.float64
        assert data["hip_flexion_r"].dtype == np.float64

    def test_row_count(self, tmp_path):
        cols = ["time", "knee_angle_l"]
        rows = [[i * 0.02, i * 3.0] for i in range(50)]
        path = _write_sto(tmp_path, cols, rows)
        data = parse_sto(path)
        assert len(data["time"]) == 50

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_sto("/nonexistent/path/file.sto")

    def test_no_endheader_raises(self, tmp_path):
        bad = "version=1\nnRows=1\ntime\thip\n0.0\t1.0\n"
        p = tmp_path / "bad.sto"
        p.write_text(bad)
        with pytest.raises(ValueError, match="endheader"):
            parse_sto(p)

    def test_column_mismatch_raises(self, tmp_path):
        content = "version=1\nendheader\ntime\thip\n0.0\t1.0\t2.0\n"
        p = tmp_path / "bad.sto"
        p.write_text(content)
        with pytest.raises(ValueError, match="mismatch"):
            parse_sto(p)

    def test_time_step_at_50hz(self, tmp_path):
        cols = ["time", "hip_flexion_r"]
        rows = [[i / FIT3D_FPS, 0.0] for i in range(100)]
        path = _write_sto(tmp_path, cols, rows)
        data = parse_sto(path)
        dt = float(np.mean(np.diff(data["time"])))
        assert abs(dt - 1.0 / FIT3D_FPS) < 1e-6


# ---------------------------------------------------------------------------
# TestBuildIdSetupXml
# ---------------------------------------------------------------------------

class TestBuildIdSetupXml:
    def test_contains_model_path(self):
        xml = _build_id_setup_xml(
            model_path="/data/s03_scaled.osim",
            mot_path="/data/squat.mot",
            output_sto="/data/squat_id.sto",
            t_start=0.0,
            t_end=10.0,
        )
        assert "/data/s03_scaled.osim" in xml

    def test_contains_mot_path(self):
        xml = _build_id_setup_xml(
            model_path="/m.osim",
            mot_path="/data/squat.mot",
            output_sto="/out/squat_id.sto",
            t_start=0.0,
            t_end=5.0,
        )
        assert "/data/squat.mot" in xml

    def test_time_range_in_xml(self):
        xml = _build_id_setup_xml(
            model_path="/m.osim",
            mot_path="/in.mot",
            output_sto="/out.sto",
            t_start=1.5,
            t_end=27.9,
        )
        assert "1.500000" in xml
        assert "27.900000" in xml

    def test_cutoff_frequency_in_xml(self):
        xml = _build_id_setup_xml(
            model_path="/m.osim",
            mot_path="/in.mot",
            output_sto="/out.sto",
            t_start=0.0,
            t_end=1.0,
            low_pass_cutoff_hz=8.0,
        )
        assert "8.0" in xml

    def test_xml_has_endtag(self):
        xml = _build_id_setup_xml(
            model_path="/m.osim",
            mot_path="/in.mot",
            output_sto="/out.sto",
            t_start=0.0,
            t_end=1.0,
        )
        assert "</InverseDynamicsTool>" in xml
        assert "</OpenSimDocument>" in xml


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_fps(self):
        assert FIT3D_FPS == 50.0

    def test_torque_coords_contains_key_joints(self):
        for coord in [
            "hip_flexion_r", "hip_flexion_l",
            "knee_angle_r", "knee_angle_l",
            "ankle_angle_r", "ankle_angle_l",
            "lumbar_extension",
        ]:
            assert coord in TORQUE_COORDS

    def test_force_coords_are_translational(self):
        for coord in FORCE_COORDS:
            assert coord.startswith("pelvis_t")

    def test_no_overlap_between_torque_and_force(self):
        assert len(set(TORQUE_COORDS) & set(FORCE_COORDS)) == 0


# ---------------------------------------------------------------------------
# TestSanityCheck
# ---------------------------------------------------------------------------

class TestSanityCheck:
    def test_sanity_check_runs_without_error(self, tmp_path, capsys):
        cols = ["time"] + list(TORQUE_COORDS[:4])
        rows = [[i * 0.02] + [float(i * 10)] * 4 for i in range(50)]
        path = _write_sto(tmp_path, cols, rows)
        run_sanity_check(path)  # should not raise
        captured = capsys.readouterr()
        assert "frames" in captured.out
        assert "fps" in captured.out

    def test_sanity_check_reports_fps(self, tmp_path, capsys):
        cols = ["time", "hip_flexion_r"]
        rows = [[i / FIT3D_FPS, float(i)] for i in range(100)]
        path = _write_sto(tmp_path, cols, rows)
        run_sanity_check(path)
        captured = capsys.readouterr()
        assert "50.0" in captured.out
