# biomech-physics-pipeline

Physics-grounded biomechanical analysis pipeline: Fit3D + SMPL-X → OpenSim → AdaBiomechanics.

Provides subject-specific musculoskeletal ground truth (joint torques, muscle activations)
for physically-validated exercise coaching research targeting IEEE TPAMI.

## Pipeline

```
Fit3D SMPL-X (betas + pose)
    └─► smplx_to_opensim_scaler.py  →  scale_setup_<subj>.xml
            └─► opensim-cmd run-tool scale_setup_<subj>.xml
                    └─► <subj>_scaled.osim
                            └─► Inverse Kinematics  →  <subj>_<exercise>_ik.mot
                                    └─► Inverse Dynamics  →  joint torques .sto
                                            └─► AdaBiomechanics RL refinement
```

## Dataset

- **Fit3D** (`train/`): 8 subjects × 47 exercises, SMPL-X params per frame
- **AMASS** (planned): cross-dataset generalization

## Setup

```bash
pip install -r requirements.txt
# OpenSim 4.x must be installed separately: https://opensim.stanford.edu
```

## Step 1 — Subject-specific scaling

```bash
python src/preprocessing/smplx_to_opensim_scaler.py \
    --fit3d D:/Datasets/Fit3D \
    --output outputs/scaling \
    --reference-model configs/Rajagopal2016.osim
```

Outputs one `scale_setup_<subj>.xml` per subject → feed into OpenSim Scaling tool.

## Project Structure

```
src/
  preprocessing/   smplx_to_opensim_scaler.py  (Step 1: betas → ScaleTool XML)
  opensim/         ik_runner.py, id_runner.py   (Steps 2-3: IK + ID)
  adabiomechanics/ rl_refinement.py             (Step 4: physics-consistent motion)
  evaluation/      metrics.py                   (Step 5: compare GT vs PEG-Reason)
scripts/           run_full_pipeline.py
configs/           default.yaml, Rajagopal2016.osim (not committed — user provides)
```

## Author

Kamran Gholizadeh HamlAbadi — MCRLab, University of Ottawa
