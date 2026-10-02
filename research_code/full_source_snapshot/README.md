# Complete model source snapshot

This directory restores the complete available PolypDG-Lite model and experiment source snapshot. Original file names are retained because the all-stage validation runner imports several scripts by name.

The original machine-specific data root has been replaced with `./data`. Place the LOCO split, checkpoints, and experiment-output folders beneath that directory, or update the configuration block at the top of the relevant script. Dataset images, prediction files, and `.pth` checkpoints are intentionally not included.

| Scope | Source files |
|---|---|
| Stage 1: U-Net + LOCO | `U-Net+LOCO(C1~C6).ipynb`, `Stage1_UNet_LOCO_model.py` |
| Stage 2: robust teachers | `segformer_B2+loco+consistency learning.py`, `SegFormer_B2+LOCO+Perturbation Bank+Consistency Learning.py` |
| Stage 3: student baselines and distillation | `SegFormer-B0+LOCO.py`, `SegForer-B0+LOCO+KD.py`, `DDRNet23slim+LOCO.py`, `ddrnet+loco.py`, `DDRNET-23-SLIM+LOCO+KD.py`, `BiSeNetV2+LOCO.py`, `BisSeNetV2+LOCO+KD.py` |
| Stage 4: deployment | `segformer-b0+loco+kd_fp16.py`, `benchmark_fp32_fp16.py`, `low-power&J frame.py` |
| Analysis and validation | `final_results.py`, `qualitative_failure_analysis.py`, `03_recompute_all_stages_unified_validation_and_signed_gap.py`, `430check.py` |

The existing `stage2/`, `stage3/`, `stage4/`, and `analysis/` directories provide curated portable copies of selected scripts. This snapshot preserves every available first-party source file, including the previously omitted U-Net, SegFormer-B0 baseline, DDRNet-23-slim, and BiSeNetV2 implementations. It does not redistribute external comparison-model source code.
