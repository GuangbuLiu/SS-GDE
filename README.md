# SS-GDE

PyTorch implementation of Structure-Sensitive Graph Dictionary Embedding for Graph Classification(https://arxiv.org/abs/2306.10505)

![architecture](/fig/SS-GDE_architecture.jpg)


## Requirements

* pytorch = 1.11.0+cu113
* networkx = 3.1
* scipy = 1.10.1
* mpu = 0.23.1

Or type `pip install -r requirements.txt` to automatically install the environment. 

## Run

`python main.py --dataset=imdb_action_romance` 

## Citation

< @article{liu2023structure,  
title={Structure-Sensitive Graph Dictionary Embedding for Graph Classification},  
author={Liu, Guangbu and Zhang, Tong and Wang, Xudong and Zhao, Wenting and Zhou, Chuanwei and Cui, Zhen},  
journal={arXiv preprint arXiv:2306.10505},  
year={2023}  
} >
