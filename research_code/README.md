# Research code

This directory contains both curated portable scripts and the complete available first-party PolypDG-Lite source snapshot. The curated scripts preserve the original experiment logic with environment-variable configuration; the snapshot retains every available model and experiment script with its data root changed to `./data`.

## Expected layout

Set `POLYPDG_DATA_ROOT` to a directory containing the experiment outputs below. Set `POLYPDG_LOCO_ROOT` directly when running training stages.

```text
experiment-root/
├── loco_1537_clean/
│   ├── fold_test_C1/
│   └── ... fold_test_C6/
├── stage2C_segformer_b2_consistency_improved_selfcons_official/
├── stage3B_segformer_b0_loco_baseline/
├── stage3E_segformer_b0_loco_kd_20260428_173054/
└── stage4_deployment_benchmark/
```

## Stages

- `stage2/segformer_b2_consistency.py`: six-fold robust teacher training.
- `stage3/segformer_b0_kd.py`: teacher-student distillation.
- `stage4/segformer_b0_kd_fp16.py`: FP16 LOCO re-evaluation.
- `stage4/benchmark_fp32_fp16.py`: runtime and memory benchmark.
- `stage4/power_joule_per_frame.py`: `nvidia-smi` power sampling and J/frame analysis.
- `analysis/`: final table generation and qualitative failure analysis.

## Complete model source

[`full_source_snapshot/`](full_source_snapshot/) contains the complete available first-party source for every reported model and experiment variant:

- Stage 1 U-Net + LOCO notebook and model definition;
- Stage 2 SegFormer-B2 consistency and perturbation-bank consistency teachers;
- Stage 3 SegFormer-B0, DDRNet-23-slim, and BiSeNetV2 baselines and KD variants; and
- Stage 4 FP16, performance, power, energy, and all-stage validation tooling.

Original source filenames are deliberately retained for traceability and compatibility with the all-stage validation runner. See its README for the model-to-file map.

## Reproducibility note

The scripts require the original PolypGen-derived LOCO split and trained fold checkpoints. Dataset images and checkpoints are deliberately excluded. Run commands should first be tested in a separate environment because the imported research scripts can create output directories and long-running GPU jobs.
