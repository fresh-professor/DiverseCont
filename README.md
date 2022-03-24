## System Dependencies
- Python >= 3.6.1
- CUDA >= 9.0 supported GPU with at least 4GB memory

## Installation
Using virtual env is recommended.
```
$ conda create --name DiverseCont python=3.6
```
Install pytorch==1.5.0 and torchvision==0.6.0.
Then, install the rest of the requirements.
```
$ pip install -r requirements.txt
```

## Data and Log directory set-up
create `checkpoints` and `data` directories.
We recommend symbolic links as below. Note the `Destination Paths` are decided during the dataset creation [here](../dataset).
```
$ mkdir data
$ ln -s [MNIST Destination Path] data/MNIST
$ ln -s [SVHN Destination Path] data/SVHN
```

## Run
Specify parameters in `config` yaml, `episodes` yaml and `episode` yaml files.
```
python main.py --log-dir [log directory path] --c [config file path] --e [episode file path] --override "|"

```
