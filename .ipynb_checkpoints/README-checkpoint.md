Pipeline for Spatio-Temporal Action Detection using SlowFast Backbone

## Requirements

``` bash
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson av psutil opencv-python tensorboard moviepy cython
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
## Installation

``` bash
git clone https://github.com/mradul2/ava-hacks
```
## Feature Extraction

Please follow the instructions provided in [DATASET.md](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) by official PySlowFast to prepare the AVA dataset.

```bash
python3 feature_extraction.py --cfg config.yaml
```

## Contributors

This repository is maintained by [Video-Language Understanding Research Group](https://makarandtapaswi.github.io) (IIIT Hyderabad)