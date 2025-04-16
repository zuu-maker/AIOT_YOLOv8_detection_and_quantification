# Defect Detection and Analysis

## Installation

This repository contains tools for training YOLOv8 models for crack/spalls detection and analyzing detected cracks/spalls.

## Scripts

- **finetune.py**: Train and fine-tune our modified yolov8 model
- **crack_quantifiery.py**: Detect and quantify cracks in images
- **spall_quantifiery.py**: Detect and quantify spalls in images

## Usage

### Training a model

```bash
python work/finetune.py --model-type YOLO --model-cfg models_cfg/model.yaml --model-weights pretrained_models/yolov8s.pt --data defects/dataset.yaml --epochs 100
```

### Analyzing cracks

```bash
python work/quantifiers/crack_quantifieryy.py --model-path models/best.pt --image-path images/crack_sample.jpg --output-path results/analysis.jpg --pixel-to-mm-ratio 10.0
```

```bash
python work/quantifiers/spall_quantifieryy.py --model-path models/best.pt --image-path images/crack_sample.jpg --output-path results/analysis.jpg --pixel-to-cm-ratio 10.0
```
