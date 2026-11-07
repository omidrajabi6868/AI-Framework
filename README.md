# AI Framework

A PyTorch computer-vision framework for training and evaluating custom object-detection, instance-segmentation, pose-estimation, fall-detection, and object-tracking workflows. It includes a custom YOLO-style model family, dataset conversion, augmentation, class balancing, ONNX export, validation pipelines, and a Django job interface.

## Capabilities

- COCO, Pascal VOC, and multi-source global dataset workflows
- Detection and instance-segmentation training
- Pose-estimation evaluation through MediaPipe-style wrappers
- Fall detection and object tracking on image/video sources
- Mosaic, multiscale, geometric, color, and noise augmentation
- Weighted sampling and class-balanced losses
- Automatic mixed precision and configurable learning-rate scheduling
- Checkpoint resume and compatibility key migration
- ONNX export for trained models
- Django-based training-job submission

## Repository structure

```text
main.py                  # Main command-line workflow
Network/                 # Models, heads, predictors, and ONNX wrappers
Train/                   # Training loop and utilities
Validation/              # Detection, segmentation, pose, tracking, fall
scripts/                  # Checkpoint migration utilities
web_interface/            # Django job interface
Main.ipynb                # Notebook experiments
```

## Installation

Create a Python environment appropriate for your CUDA/PyTorch version:

```bash
git clone https://github.com/omidrajabi6868/AI-Framework.git
cd AI-Framework
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy pandas pillow opencv-python scipy scikit-learn django
```

Additional packages may be required by the selected model wrapper, augmentation pipeline, or export backend.

## Dataset integration

`main.py` imports dataset classes and augmentation builders from a `Dataset` package and accepts `coco`, `voc`, or `global` through `--dataset_name`. Provide that package and its dataset configuration in the working tree/environment before running the main workflow. Dataset paths and metadata can be supplied as JSON objects or JSON files.

A typical metadata object contains paths such as:

```json
{
  "class_file": "/path/to/classes.txt",
  "data_file": "/path/to/dataset.data"
}
```

## Run configuration

The CLI supports individual flags and a JSON run configuration. The default config path is `json_run.json`:

```bash
python main.py --run_config /path/to/run_config.json
```

Run configurations may define dataset paths, metadata, model settings, augmentation, optimization, checkpointing, and task-specific evaluation parameters. Command-line arguments can also be used directly; inspect all options with:

```bash
python main.py --help
```

## Supported tasks

The current `main.py` execution paths support:

```text
detection
instance_segmentation
pose_estimation
fall_detection
object_tracking
```

The primary custom detector path uses `--model_name ipc8`. Detection and instance-segmentation workflows can train, validate, resume from checkpoints, and optionally export ONNX models.

## Example detection workflow

```bash
python main.py \
  --task detection \
  --dataset_name coco \
  --model_name ipc8 \
  --batch_size 16 \
  --image_size 640 \
  --epoch_num 100 \
  --learning_rate 0.001 \
  --saved_name detector
```

Supply the required image, annotation, metadata, output, and checkpoint paths through the run configuration or their corresponding CLI options.

## Django job interface

The `web_interface` Django project stores training jobs and launches `main.py` with a temporary JSON configuration:

```bash
cd web_interface
python manage.py migrate
python manage.py runserver
```

The service records job status and invokes the ML workflow as a subprocess from the repository root.

## Checkpoint migration

Use `scripts/migrate_checkpoint_keys.py` when an older checkpoint uses parameter names that do not match the current model layout. The model loader also contains compatibility remapping for supported formats.

## Project status

This is an active research framework. Dataset modules, paths, model weights, and environment-specific dependencies should be configured for the target experiment before execution.
