# SS-GDE

PyTorch implementation of Structure-Sensitive Graph Dictionary Embedding for Graph Classification(https://arxiv.org/abs/2306.10505)

![architecture](/fig/SS-GDE_architecture.jpg)


## Requirements

* pytorch = 1.11.0+cu113
* networkx = 3.1
* scipy = 1.10.1
* mpu = 0.23.1

or type `pip install -r requirements.txt` to automatically install the environment. 

## Run

`python main.py --dataset=imdb_action_romance` 
