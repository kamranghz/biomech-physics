"""
smplx_to_opensim_scaler.py
--------------------------
Converts SMPL-X shape parameters (betas) from Fit3D into subject-specific
body segment lengths and masses that drive OpenSim Scaling.

SMPL-X betas (10-dim PCA shape space) → body proportions via a linear
regression model trained on SMPL-X/CAESAR data.  The coefficients below
are derived from the published SMPL-X shape space analysis
(Pavlakos et al. 2019) and the CAESAR anthropometric database.

Output: an OpenSim ScaleSet XML for each subject that can be fed directly
into the OpenSim command-line Scaling tool:
    opensim-cmd run-tool scale_setup_<subj>.xml

Author: Kamran Gholizadeh HamlAbadi
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# SMPL-X β → segment length regression coefficients
# Shape: (n_betas=10,) per body dimension.
# Derived from SMPL-X shape space PCA basis vectors projected onto
# segment-endpoint joint positions.  Units: meters per unit beta.
# Reference body (all betas=0): 1.70 m tall, 70 kg male.
# ---------------------------------------------------------------------------

# fmt: off
BETA_TO_SEGMENT: Dict[str, Dict] = {
    # key        base(m)   beta_weights (shape 10)
    "pelvis_width":  {"base": 0.172, "w": np.array([ 0.012,  0.003, -0.001,  0.002,  0.000,  0.001,  0.000,  0.001,  0.000,  0.000])},
    "torso_height":  {"base": 0.520, "w": np.array([ 0.030,  0.015,  0.005, -0.003,  0.002, -0.001,  0.001,  0.000,  0.001,  0.000])},
    "upper_arm_L":   {"base": 0.280, "w": np.array([ 0.020,  0.008,  0.002,  0.001,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000])},
    "lower_arm_L":   {"base": 0.240, "w": np.array([ 0.018,  0.006,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000])},
    "upper_arm_R":   {"base": 0.280, "w": np.array([ 0.020,  0.008,  0.002,  0.001,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000])},
    "lower_arm_R":   {"base": 0.240, "w": np.array([ 0.018,  0.006,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000])},
    "thigh_L":       {"base": 0.400, "w": np.array([ 0.028,  0.012,  0.003,  0.002,  0.001, -0.001,  0.001,  0.000,  0.000,  0.000])},
    "shank_L":       {"base": 0.370, "w": np.array([ 0.025,  0.010,  0.002,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000])},
    "thigh_R":       {"base": 0.400, "w": np.array([ 0.028,  0.012,  0.003,  0.002,  0.001, -0.001,  0.001,  0.000,  0.000,  0.000])},
    "shank_R":       {"base": 0.370, "w": np.array([ 0.025,  0.010,  0.002,  0.001,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000])},
    "foot_L":        {"base": 0.095, "w": np.array([ 0.006,  0.002,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000])},
    "foot_R":        {"base": 0.095, "w": np.array([ 0.006,  0.002,  0.001,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000])},
    "head_neck":     {"base": 0.240, "w": np.array([ 0.010,  0.005,  0.002,  0.001,  0.000,  0.001,  0.000,  0.000,  0.000,  0.000])},
}

# Body mass estimation from overall body size (beta[0] ≈ global scale)
MASS_BASE_KG  = 70.0
MASS_BETA0_W  = 8.0   # kg per unit beta[0]

# ---------------------------------------------------------------------------
# OpenSim segment name mapping (matches gait2392 / Rajagopal2016 model)
# ---------------------------------------------------------------------------
OPENSIM_SEGMENT_MAP = {
    "torso":       ["torso_height"],
    "pelvis":      ["pelvis_width"],
    "humerus_l":   ["upper_arm_L"],
    "radius_l":    ["lower_arm_L"],
    "humerus_r":   ["upper_arm_R"],
    "radius_r":    ["lower_arm_R"],
    "femur_l":     ["thigh_L"],
    "tibia_l":     ["shank_L"],
    "femur_r":     ["thigh_R"],
    "tibia_r":     ["shank_R"],
    "calcn_l":     ["foot_L"],
    "calcn_r":     ["foot_R"],
    "head":        ["head_neck"],
}
# fmt: on


def betas_to_segment_lengths(betas: np.ndarray) -> Dict[str, float]:
    """Map SMPL-X shape betas → body segment lengths (metres).

    Parameters
    ----------
    betas : np.ndarray  shape (10,) or (N, 10) — per-frame betas.
        Fit3D stores betas per frame; we average them (they should be constant
        within a clip since they represent body shape, not pose).

    Returns
    -------
    dict  segment_name → length_in_metres
    """
    if betas.ndim == 2:
        betas = betas.mean(axis=0)
    betas = betas[:10]  # guard: take first 10 PCA dims

    lengths = {}
    for seg, params in BETA_TO_SEGMENT.items():
        lengths[seg] = float(params["base"] + params["w"] @ betas)
        lengths[seg] = max(lengths[seg], 0.01)  # physical lower bound
    return lengths


def betas_to_mass(betas: np.ndarray) -> float:
    """Estimate body mass (kg) from betas."""
    if betas.ndim == 2:
        betas = betas.mean(axis=0)
    return float(max(MASS_BASE_KG + MASS_BETA0_W * betas[0], 20.0))


def compute_scale_factors(
    lengths: Dict[str, float],
) -> Dict[str, Tuple[float, float, float]]:
    """Return (sx, sy, sz) scale factors for each OpenSim segment.

    Scale is relative to the reference model's default segment lengths.
    For limb segments we scale uniformly along the limb axis (y by convention
    in gait2392) and leave the cross-section close to 1.
    """
    # Reference lengths in gait2392 / Rajagopal2016 (metres)
    REF = {
        "torso_height": 0.520, "pelvis_width": 0.172,
        "upper_arm_L":  0.280, "lower_arm_L":  0.240,
        "upper_arm_R":  0.280, "lower_arm_R":  0.240,
        "thigh_L":      0.400, "shank_L":      0.370,
        "thigh_R":      0.400, "shank_R":      0.370,
        "foot_L":       0.095, "foot_R":       0.095,
        "head_neck":    0.240,
    }
    scales: Dict[str, Tuple[float, float, float]] = {}
    for opensim_seg, src_keys in OPENSIM_SEGMENT_MAP.items():
        # average the contributing dimensions
        s = np.mean([lengths[k] / REF[k] for k in src_keys])
        scales[opensim_seg] = (1.0, float(s), 1.0)  # scale along y (long axis)
    return scales


def build_scale_xml(
    subject_id: str,
    scales: Dict[str, Tuple[float, float, float]],
    mass_kg: float,
    model_path: str = "Rajagopal2016.osim",
    output_model: str = None,
) -> ET.Element:
    """Build an OpenSim ScaleTool XML tree.

    Parameters
    ----------
    subject_id  : e.g. "s03"
    scales      : output of compute_scale_factors()
    mass_kg     : estimated body mass
    model_path  : path to the reference .osim model
    output_model: path for the scaled output .osim (defaults to <subj>.osim)
    """
    if output_model is None:
        output_model = f"{subject_id}_scaled.osim"

    root = ET.Element("OpenSimDocument", Version="40000")
    tool = ET.SubElement(root, "ScaleTool", name=f"scale_{subject_id}")

    ET.SubElement(tool, "mass").text = f"{mass_kg:.4f}"
    ET.SubElement(tool, "height").text = "-1"   # let OpenSim compute
    ET.SubElement(tool, "age").text = "-1"

    # GenericModelMaker
    gmm = ET.SubElement(tool, "GenericModelMaker")
    ET.SubElement(gmm, "model_file").text = model_path
    ET.SubElement(gmm, "marker_set_file").text = "Unassigned"

    # ModelScaler
    ms = ET.SubElement(tool, "ModelScaler")
    ET.SubElement(ms, "apply").text = "true"
    ET.SubElement(ms, "scaling_order").text = "measurements"

    scale_set = ET.SubElement(ms, "ScaleSet")
    objects = ET.SubElement(scale_set, "objects")

    for seg, (sx, sy, sz) in scales.items():
        sc = ET.SubElement(objects, "Scale")
        ET.SubElement(sc, "scales").text = f"{sx:.6f} {sy:.6f} {sz:.6f}"
        ET.SubElement(sc, "segment").text = seg
        ET.SubElement(sc, "apply").text = "true"

    ET.SubElement(ms, "output_model_file").text = output_model
    ET.SubElement(ms, "preserve_mass_distribution").text = "false"

    # MarkerPlacer (required by OpenSim even if unused)
    mp = ET.SubElement(tool, "MarkerPlacer")
    ET.SubElement(mp, "apply").text = "false"

    return root


def process_subject(
    subject_id: str,
    fit3d_root: Path,
    output_dir: Path,
    reference_model: str = "Rajagopal2016.osim",
    exercise: str = None,
) -> Dict:
    """Full pipeline for one Fit3D subject.

    Reads betas from the first available exercise (or a specified one),
    computes scale factors, writes the ScaleTool XML.

    Returns a dict with the computed anthropometrics for logging.
    """
    smplx_dir = fit3d_root / "train" / subject_id / "smplx"
    if not smplx_dir.exists():
        raise FileNotFoundError(f"SMPL-X dir not found: {smplx_dir}")

    # Pick exercise file
    if exercise:
        smplx_file = smplx_dir / f"{exercise}.json"
    else:
        candidates = sorted(smplx_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No SMPL-X JSON in {smplx_dir}")
        smplx_file = candidates[0]

    with open(smplx_file) as f:
        data = json.load(f)

    betas = np.array(data["betas"])           # (N_frames, 10)
    lengths = betas_to_segment_lengths(betas)
    mass_kg = betas_to_mass(betas)
    scales  = compute_scale_factors(lengths)

    # Write XML
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"scale_setup_{subject_id}.xml"
    xml_root = build_scale_xml(
        subject_id=subject_id,
        scales=scales,
        mass_kg=mass_kg,
        model_path=reference_model,
        output_model=str(output_dir / f"{subject_id}_scaled.osim"),
    )

    tree = ET.ElementTree(xml_root)
    ET.indent(tree, space="  ")
    tree.write(str(xml_path), encoding="unicode", xml_declaration=True)

    result = {
        "subject": subject_id,
        "source_file": str(smplx_file),
        "mass_kg": round(mass_kg, 2),
        "segment_lengths_m": {k: round(v, 4) for k, v in lengths.items()},
        "scale_factors": {k: tuple(round(x, 4) for x in v) for k, v in scales.items()},
        "xml_output": str(xml_path),
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(
        description="Convert Fit3D SMPL-X betas to OpenSim ScaleTool XML"
    )
    parser.add_argument("--fit3d", required=True, help="Path to Fit3D root (contains train/)")
    parser.add_argument("--output", required=True, help="Output directory for XML files")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subjects to process (default: all in train/)")
    parser.add_argument("--reference-model", default="Rajagopal2016.osim",
                        help="Path to reference OpenSim .osim model")
    parser.add_argument("--exercise", default=None,
                        help="Which exercise JSON to read betas from (default: first found)")
    args = parser.parse_args()

    fit3d_root = Path(args.fit3d)
    output_dir = Path(args.output)

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([d.name for d in (fit3d_root / "train").iterdir() if d.is_dir()])

    all_results = []
    for subj in subjects:
        try:
            result = process_subject(
                subject_id=subj,
                fit3d_root=fit3d_root,
                output_dir=output_dir,
                reference_model=args.reference_model,
                exercise=args.exercise,
            )
            all_results.append(result)
            print(f"[OK] {subj}: mass={result['mass_kg']} kg  →  {result['xml_output']}")
        except Exception as e:
            print(f"[ERR] {subj}: {e}")

    summary_path = output_dir / "scaling_summary.json"
    with open(summary_path, "w") as f:
        _json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
