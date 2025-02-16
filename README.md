# BQN-demo

This repository contains a PyTorch implementation of BQN.

## Runtime Environment

* python 3.9
* numpy 1.26.3
* pytorch 2.0.0
* scikit-learn 1.5.1

## Dataset

We provide the ABIDE dataset and placed it in the `/dataset` directory. Please place all datasets in the `/dataset`
directory and update the corresponding paths in `parse.py`.

## Usage

1. Change the `data_dir` attribute in file `parse.py` to the path of your dataset.

2. Change the `root_path` attribute in file `parse.py` to the path of your project path.

3. Run the following command to train the model.
```bash
python main.py
```




