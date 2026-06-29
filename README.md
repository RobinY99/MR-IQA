# MR-IQA

**MR-IQA: A Unified Margin View for Image Quality Assessment**

This repository contains the training, validation, and evaluation scaffolding for MR-IQA. It is designed for reproducible image quality assessment experiments with explicit separation between code, manifests, local image roots, and generated outputs.

Released model weights are available on Hugging Face: [RobinY99/MR-IQA](https://huggingface.co/RobinY99/MR-IQA).

Private images, large raw datasets, checkpoints, and generated experiment outputs are intentionally not committed.

## 1. Environment Setup

Create an isolated Python environment before installing project dependencies:

```bash
conda create -n mr-iqa python=3.10 -y
conda activate mr-iqa
pip install -r requirements.txt
```

If your cluster manages environments with `venv` instead of conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver if the default wheel is not suitable for your machine. `CUDA_HOME` is only needed when packages compile CUDA extensions; the launch scripts add `src/` to `PYTHONPATH` automatically.

Optional runtime overrides:

```bash
export CUDA_HOME=<cuda-toolkit-root>
export PYTHON_BIN=python3
export REPORT_TO=none
```

## 2. Data Preparation

Prepare a training manifest in JSON or JSONL format. Each row should contain an image path and a human quality score:

```json
{"image": "000001.png", "score": 3.72, "std": 0.41}
```

Supported score keys include `score`, `mos`, `rating`, `human_score`, `normalized_score`, and `gt_score_norm`. Supported uncertainty keys include `std`, `score_std`, `mos_std`, `source_std`, and `std_norm`.

The committed manifests use relative image paths only. Point `IMAGE_ROOT` or `VAL_IMAGE_ROOT` to the corresponding image directory on your machine or cluster.

Expected layout:

```text
data/
  manifest_checksums.json
  train_manifest/
    train.jsonl
  val_manifests/
    koniq_val_200_seed42.json
  test_manifests/
    agiqa3k.json
    csiq.json
    kadid_full.json
    koniq.json
    livew.json
    pipal.json
    spaq_full.json
    tid2013.json
```

## 3. Training

The launch scripts do not contain local absolute paths. Set the model, manifest, and image-root paths explicitly for each machine.

Run 8-GPU full fine-tuning without validation:

```bash
MODEL_PATH=<hf-model-id-or-local-model-dir> \
DATA_FILES=data/train_manifest/train.jsonl \
IMAGE_ROOT=<train-image-root> \
OUTPUT_DIR=outputs/mr-iqa-2b \
VARIANCE_MODE=unit \
bash scripts/train_mr_iqa_2b_8gpu.sh
```

Run 8-GPU full fine-tuning followed by 8-GPU validation:

```bash
MODEL_PATH=<hf-model-id-or-local-model-dir> \
DATA_FILES=data/train_manifest/train.jsonl \
IMAGE_ROOT=<train-image-root> \
VAL_DATA_FILE=data/val_manifests/koniq_val_200_seed42.json \
VAL_IMAGE_ROOT=<val-image-root> \
OUTPUT_DIR=outputs/mr-iqa-2b \
VAL_OUTPUT_JSON=outputs/mr-iqa-2b/validation/final.json \
VARIANCE_MODE=unit \
bash scripts/train_mr_iqa_2b_8gpu_with_val.sh
```

`VARIANCE_MODE` controls the margin scale:

```text
unit   uses variance scale 1
sigma  uses the paired ground-truth score sigma
```

For a 4B backbone, use:

```bash
bash scripts/train_mr_iqa_4b_8gpu.sh
```

or the validation-enabled wrapper:

```bash
bash scripts/train_mr_iqa_4b_8gpu_with_val.sh
```

## 4. Testing

Single-dataset evaluation:

```bash
python src/mr_iqa/evaluate_mr_iqa.py \
  --model_name_or_path <model-or-checkpoint> \
  --data_file data/test_manifests/koniq.json \
  --image_root <image-root> \
  --output_json outputs/eval/koniq.json
```

8-GPU validation-only evaluation:

```bash
MODEL_DIR=<model-or-checkpoint> \
VAL_DATA_FILE=data/val_manifests/koniq_val_200_seed42.json \
IMAGE_ROOT=<image-root> \
OUT_JSON=outputs/validation/koniq_val.json \
bash scripts/validation_eval_8gpu.sh
```

8-GPU generalization evaluation:

```bash
MODEL_DIR=<model-or-checkpoint> \
DATA_DIR=data/test_manifests \
IMAGE_ROOT=<image-root> \
OUT_DIR=outputs/generalization \
bash scripts/generalization_eval_8gpu.sh
```

Override the default generalization set with `DATASETS="koniq spaq_full"` if you only want a subset.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Citation

```bibtex
@article{mriqa,
  title={MR-IQA: A Unified Margin View for Image Quality Assessment},
  author={TODO},
  journal={TODO}
}
```
