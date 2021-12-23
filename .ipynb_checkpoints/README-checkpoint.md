Pipeline for Spatio-Temporal Action Detection using SlowFast Backbone

## Requirements

``` bash
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson av psutil opencv-python tensorboard moviepy cython
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
## Installation

``` bash
git clone https://github.com/mradul2/ava-feature-extraction
```
## Usage

For feature extraction: 

```bash
python3 feature_extraction.py --cfg config.yaml
```