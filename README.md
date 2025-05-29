# Do We Really Need Message Passing in Brain Network Modeling?

This repository contains a PyTorch implementation of the Brain Quadratic Network (BQN), an open-source 
implementation of the ICML 2025 paper [Do We Really Need Message Passing in Brain Network Modeling?](https://openreview.net/forum?id=KRosBwvhDx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions)).

## Runtime Environment

* python 3.9
* numpy 1.26.3
* pytorch 2.0.0
* scikit-learn 1.5.1

## Dataset

Download the ABIDE dataset from [here](https://drive.google.com/file/d/1OkWTorNXjInYzmFH34KsYH4YA06xDUAQ/view?usp=drive_link)

## Usage

1. Change the `data_dir` attribute in file `parse.py` to the path of your dataset.

2. Change the `root_path` attribute in file `parse.py` to the path of your project path.

3. Run the following command to train the model.
```bash
python main.py
```

## Citation
If you find our code and model useful, please cite our work. Thank you!
```bibtex
@inproceedings{
yang2025bqn,
title={Do We Really Need Message Passing in Brain Network Modeling?},
author={Liang Yang and Yuwei Liu and Jiaming Zhuo and Di Jin and Chuan Wang and Zhen Wang and Xiaochun Cao},
booktitle={Forty-second International Conference on Machine Learning},
year={2025}
}
