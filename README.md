# Data Augmentation GAN

This implementation is based on [amurthy1/dagan_torch](https://github.com/amurthy1/dagan_torch) which is a PyTorch implementation for the paper [Data Augmentation Generative Adversarial Networks](https://arxiv.org/abs/1711.04340).

The architecture is changed to consider one class instead of multiple classes. This simplfies the overall architecture. The following image is a slightly modified version of the original DAGAN architecture (see [paper](https://arxiv.org/abs/1711.04340)):

<p align="center">
  <img src="resources/dagan_model.png" style="width:40%"/>
</p>

## Start Training

### Dataset
The dataset is assumed to be stored in [npz](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) format with the following keys:
- `orig`: numpy array holding all original images
- `aug`: numpy array holding all corresponding augmentation images

The names of the dataset used in the implementation are:
- `train.npz`
- `val.npz`

### Train

1. Create virtual environment
  ```
  conda create -n my_env python=3.8
  ```
2. Install the dependencies in `requirements.txt`
  ```
  pip install -r requirements.txt
  ```
3. Install DAGAN package
  ```
  pip install -e .
  ```
4. Run train script (for more configuration options, see `.utils/parser.py`)
  ```
  python train.py PATH_TO_DATASETS --model_path PATH_TO_STORE_MODELS -n NAME_OF_WANDB_RUN_AND_MODELS_DIR
  ```

