Pipeline for Spatio-Temporal Action Detection using SlowFast Backbone

Project Slides: [Atomic Visual Actions](https://docs.google.com/presentation/d/1OUVZ513kxAnBVbeMvSL1BrT87A0IhdieOdpO5xmOrAc/edit?usp=sharing)

## Requirements

``` bash
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson av psutil opencv-python tensorboard moviepy cython
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
## Installation

``` bash
pip install -r requirements.txt
git clone https://github.com/mradul2/ava-hacks
```
## Feature Extraction

Please follow the instructions provided in [DATASET.md](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) by official PySlowFast to prepare the AVA dataset.

Pre-trained Slow-fast model with 29.4 mAP on AVA dataset can be downloaded from here: [Model Link](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_64x2_R101_50_50.pkl)

```bash
python3 feature_extraction.py --cfg configs/slowfast.yaml
```

## Training and Evaluation

A classification head can be trained and evaluated using the following two bash commands after making appropriate changes in the config file.

```bash 
python3 train.py --cfg configs/config.yamk
```

```bash
python3 test.py --cfg configs/config.yaml
```

## Contributors

This repository is maintained by [Video-Language Understanding Research Group](https://makarandtapaswi.github.io) (IIIT Hyderabad)
