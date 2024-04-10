# CoSLight
This is the offical implementaion of CoSLight: Co-optimizing Collaborator Selection and Decision-making to Enhance Multi-intersection Traffic Signal Control

## Environment
### 1. Install

```python
# create conda env 
conda create -n tsc python=3.8 

# install torch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install current path
pip install -e .

# install other package
pip install eclipse-sumo==1.14.0 traci sumolib libsumo tqdm==4.59.0 setproctitle absl-py gym tensorboardX wandb imageio 

# NOTE this version of numpy
!pip install numpy==1.21.6

```

### 2. Set the environment path

```shell
export SUMO_HOME=/your python env path/lib/python3.6/site-packages/sumo
export PYTHONPATH=${PYTHONPATH}:/your own folder/root directory of this folder
```

### 3. unzip resco scenarios' files

```shell
cd ./CoSLight/
./decompress.sh zip_dir/       ### unzip the main folder
./decompress.sh zip_dir/envs/  ### unzip the child folder
mv zip_dir/envs .              ### move the 'envs' to the current path
```


## Training

```shell
cd ./CoSLight/scripts/train/
python train_sumo.py
```

