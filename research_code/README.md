# Research code

These are the selected scripts used to build the evidence chain reported in the paper. They preserve the original experiment logic while replacing user-specific absolute paths with environment-variable configuration.

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

The Stage 1 U-Net experiment originated as a development notebook and is not included until it is cleaned into a portable script. The verified Stage 1 aggregate results remain available in `results/stagewise_summary.csv`.

## Reproducibility note

The scripts require the original PolypGen-derived LOCO split and trained fold checkpoints. Dataset images and checkpoints are deliberately excluded. Run commands should first be tested in a separate environment because the imported research scripts can create output directories and long-running GPU jobs.
