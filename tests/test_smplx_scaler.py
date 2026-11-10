"""Smoke tests for smplx_to_opensim_scaler — no OpenSim required."""
import sys, json, tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing.smplx_to_opensim_scaler import (
    betas_to_segment_lengths,
    betas_to_mass,
    compute_scale_factors,
    build_scale_xml,
)


def test_zero_betas_returns_reference_body():
    betas = np.zeros(10)
    lengths = betas_to_segment_lengths(betas)
    mass = betas_to_mass(betas)
    assert abs(lengths["thigh_L"] - 0.400) < 0.001, "thigh length off"
    assert abs(mass - 70.0) < 0.1, "mass off"


def test_positive_beta0_increases_size():
    b0 = np.zeros(10)
    b1 = np.zeros(10); b1[0] = 2.0
    l0 = betas_to_segment_lengths(b0)
    l1 = betas_to_segment_lengths(b1)
    assert l1["torso_height"] > l0["torso_height"]
    assert betas_to_mass(b1) > betas_to_mass(b0)


def test_scale_factors_reference_body_near_one():
    betas = np.zeros(10)
    lengths = betas_to_segment_lengths(betas)
    scales = compute_scale_factors(lengths)
    for seg, (sx, sy, sz) in scales.items():
        assert abs(sy - 1.0) < 0.02, f"{seg} scale y={sy:.4f} not near 1.0"


def test_xml_builds_without_error():
    betas = np.random.randn(100, 10) * 0.5
    lengths = betas_to_segment_lengths(betas)
    scales  = compute_scale_factors(lengths)
    mass    = betas_to_mass(betas)
    xml = build_scale_xml("s03", scales, mass)
    import xml.etree.ElementTree as ET
    txt = ET.tostring(xml, encoding="unicode")
    assert "ScaleTool" in txt
    assert "femur_l" in txt


def test_per_frame_betas_averaged():
    betas_const  = np.ones((200, 10)) * 1.5
    betas_single = np.ones(10) * 1.5
    l1 = betas_to_segment_lengths(betas_const)
    l2 = betas_to_segment_lengths(betas_single)
    for k in l1:
        assert abs(l1[k] - l2[k]) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
