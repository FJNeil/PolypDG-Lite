# PolypDG-Lite

### A deployment-oriented approach for cross-center colonoscopic polyp segmentation via domain generalization and low-power knowledge distillation

[![Project page](https://img.shields.io/badge/Project_Page-Live-d7ff43?style=flat-square&labelColor=14211c)](https://polypdg-lite-research.fjff.chatgpt.site)
[![Full paper](https://img.shields.io/badge/Full_Paper-PDF-bb2b2b?style=flat-square&labelColor=14211c)](docs/PolypDG-Lite-full-paper.pdf)
[![Dataset](https://img.shields.io/badge/Dataset-PolypGen-6f8178?style=flat-square)](https://doi.org/10.1038/s41597-023-01981-y)
[![Protocol](https://img.shields.io/badge/Evaluation-6--center_LOCO-6f8178?style=flat-square)](#evaluation-protocol)

**Shih-Wei Fan Chiang**<sup>1</sup>, **Yen-Ching Chang**<sup>1,2</sup><br>
<sup>1</sup> Department of Medical Informatics, Chung Shan Medical University<br>
<sup>2</sup> Department of Medical Imaging, Chung Shan Medical University Hospital

- **Student researcher and first author:** Shih-Wei Fan Chiang
- **Faculty advisor and research co-author:** Yen-Ching Chang

PolypDG-Lite is a four-stage framework for robust colonoscopic polyp segmentation under cross-center domain shift. It combines a strict leave-one-center-out (LOCO) protocol, consistency-trained SegFormer-B2 teachers, knowledge distillation into compact students, and FP16 low-power deployment validation.

## First-author contribution

As the student researcher and first author, Shih-Wei Fan Chiang led the research implementation and artifact preparation represented in this repository:

- designed and implemented the six-center LOCO evaluation workflow;
- developed the consistency-trained teacher and knowledge-distillation experiments;
- implemented FP32/FP16 deployment, memory, speed, power, and energy benchmarking;
- performed result aggregation, qualitative failure analysis, and reproducibility packaging;
- prepared the full paper, conference presentation, and research showcase.

Yen-Ching Chang served as faculty advisor and research co-author.

## Portfolio quick tour

This repository accompanies the ICATI 2026 full paper by student researcher Shih-Wei Fan Chiang and faculty advisor Yen-Ching Chang. For a short review, follow this order:

1. Read the [key results](#key-results) and [four-stage framework](#framework) below.
2. Open the [16-page full paper](docs/PolypDG-Lite-full-paper.pdf).
3. Inspect the verified, machine-readable tables in [`results/`](results/).
4. Review the selected experiment implementation and reproducibility notes in [`research_code/`](research_code/).
5. Open the [project page](https://polypdg-lite-research.fjff.chatgpt.site) for a full-screen view of the conference presentation.

This research artifact intentionally contains selected portable scripts rather than patient data, model checkpoints, or the complete raw experiment workspace.

## Key results

| Configuration | Mean Dice | Mean IoU | C4 Dice | Parameters | FPS | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| U-Net + LOCO | 63.40% | - | 43.45% | - | - | - |
| SegFormer-B2 + Consistency | **79.61%** | **73.38%** | **66.13%** | 27.35M | 31.44 | 422.35 MB |
| SegFormer-B0 + KD (FP32) | 77.05% | 70.38% | 60.44% | **3.71M** | 87.05 | 125.64 MB |
| **SegFormer-B0 + KD (FP16)** | **77.05%** | **70.38%** | **60.44%** | **3.71M** | **95.26** | **55.43 MB** |

The final FP16 student has an estimated 7.08 MB parameter footprint, mean power draw of 14.77 W, and energy cost of 0.3818 J/frame. FLOPs were not reported in the verified experiment artifacts and are intentionally omitted.

## Framework

![Four-stage PolypDG-Lite research framework](docs/framework-overview.png)

1. **Cross-center diagnosis** - establish a U-Net baseline under center-level LOCO.
2. **Robust teacher** - train SegFormer-B2 with prediction consistency across appearance-perturbed views.
3. **Lightweight distillation** - transfer teacher probability maps to DDRNet-23-slim, BiSeNetV2, and SegFormer-B0 students.
4. **Deployment validation** - re-evaluate the selected SegFormer-B0 + KD student under FP16 and profile speed, memory, power, and energy per frame.

### Stage 2 - Teacher training and selection

![Stage 2 teacher training and selection with consistency learning](docs/stage2-teacher-training-and-selection.png)

The SegFormer-B2 teacher processes an original view and an appearance-perturbed view. Consistency learning minimizes the difference between their probability maps, encouraging predictions that remain stable under brightness, color, blur, and compression changes.

### Stage 3 - Knowledge distillation

![Stage 3 knowledge distillation from SegFormer-B2 to a lightweight student](docs/stage3-knowledge-distillation.png)

The selected teacher produces a soft probability map for the lightweight student. Training combines knowledge-distillation loss with supervised segmentation loss, transferring cross-center behavior while retaining direct ground-truth supervision.

## Evaluation protocol

Experiments use the clean, center-labeled subset of the [PolypGen](https://doi.org/10.1038/s41597-023-01981-y) dataset: **1,537 image-mask pairs from six clinical centers** (C1-C6). Each LOCO fold holds out one complete center as an unseen test domain; validation data comes only from the remaining training centers.

| Center | C1 | C2 | C3 | C4 | C5 | C6 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Images | 256 | 301 | 457 | 227 | 208 | 88 | **1,537** |

Results include region metrics (Dice, IoU), boundary metrics (HD95, ASSD), worst-center performance, and validation-to-test degradation. Machine-readable verified tables are in [`results/`](results/).

## Qualitative evidence

This diagnostic figure asks two questions: **Does the final student suppress false positives on empty-mask frames?** and **Where does it still miss a true lesion?** Read each row from left to right: input image, ground truth, U-Net baseline, robust teacher, undistilled student, and final distilled student.

- **Green overlay:** ground-truth lesion mask.
- **Red overlay:** model prediction.
- **Dice = 100% with no colored mask:** both ground truth and prediction are empty. This is correct rejection of a false positive, not perfect lesion delineation.
- **Remaining failure:** the final row documents a true lesion missed by the distilled student, preventing the evidence from showing only successful cases.

Because the full grid is large and intended for audit rather than first-glance presentation, it is collapsed by default.

<details>
<summary><strong>Open the full five-case C4 comparison</strong></summary>

<br>

![Five qualitative C4 cases comparing ground truth and four model configurations](public/results/qualitative-c4-comparison.png)

</details>

## Code map

```text
docs/             # full paper, conference presentation, and framework figures
research_code/    # selected teacher, distillation, deployment, and analysis scripts
results/          # verified machine-readable experiment tables
app/              # full-screen paper viewer
public/           # website images and qualitative evidence
worker/           # website deployment entry point
tests/            # rendered-site checks
```

The repository contains selected research scripts recovered from the experiment workspace. Local machine paths have been replaced with environment variables, but the full training pipeline still requires the PolypGen-derived LOCO directory layout and fold checkpoints described in [`research_code/README.md`](research_code/README.md).

Folders such as `db/`, `drizzle/`, and `examples/` support the research-showcase starter and are not part of the model-training pipeline.

## Research artifacts

| Artifact | Purpose |
|---|---|
| [Full paper](docs/PolypDG-Lite-full-paper.pdf) | Sixteen-page ICATI 2026 manuscript with methodology, experiments, discussion, and references |
| [Conference presentation](docs/PolypDG-Lite-conference-presentation.pdf) | Fifteen-slide conference talk retained as a supplementary overview |
| [`results/stagewise_summary.csv`](results/stagewise_summary.csv) | Verified stage-by-stage segmentation results |
| [`results/deployment_benchmark.csv`](results/deployment_benchmark.csv) | FP32/FP16 speed, memory, power, and energy measurements |
| [`results/dataset_loco_split.csv`](results/dataset_loco_split.csv) | Six-center LOCO dataset accounting |
| [`results/qualitative_c4_case_selection.csv`](results/qualitative_c4_case_selection.csv) | Auditable selection of qualitative C4 cases |
| [`research_code/README.md`](research_code/README.md) | Reproducibility scope, expected paths, and limitations |

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Set the dataset or experiment root before running a stage:

```bash
# Linux/macOS
export POLYPDG_DATA_ROOT=/path/to/experiment-root
export POLYPDG_LOCO_ROOT=/path/to/loco_1537_clean

# PowerShell
$env:POLYPDG_DATA_ROOT = "D:\path\to\experiment-root"
$env:POLYPDG_LOCO_ROOT = "D:\path\to\loco_1537_clean"
```

## Data and model weights

The PolypGen images, clinical-center data, and trained checkpoints are **not redistributed** in this repository. Obtain the dataset from its official source and follow its license and data-use terms. Model checkpoints will be released separately only after redistribution rights and release packaging are confirmed.

## Citation

```bibtex
@inproceedings{chiang2026polypdglite,
  title     = {PolypDG-Lite: A Deployment-Oriented Approach for Cross-Center Colonoscopic Polyp Segmentation via Domain Generalization and Low-Power Knowledge Distillation},
  author    = {Chiang, Shih-Wei Fan and Chang, Yen-Ching},
  booktitle = {The 11th International Conference on Advanced Technology Innovation},
  year      = {2026}
}
```

## Responsible use

This repository is a research artifact, not a medical device. Predictions must not be used for diagnosis or patient care without appropriate clinical validation, regulatory review, and human oversight. No patient-identifiable information is included.
