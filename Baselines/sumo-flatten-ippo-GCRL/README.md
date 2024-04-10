## MAPPO & SUMO Framework
- **actor**: using frap to extract the feature
- **critic**: using frap to predict the value

## Environments supported:

- [SUMO]()

## 1. Conda Installation


``` Bash
# create conda environment
conda env create -f environment.yml
```

```
# install on-policy package
cd sumo-frap-mappo-ippo
pip install -e .
```


## 2. SUMO Installation





```shell

#### unzip sumo_files_new.zip
cd sumo-frap-mappo-ippo/onpolicy/envs/
unzip sumo_files_new.zip

#### package
eclipse-sumo==1.14.0
traci
sumolib
libsumo
tqdm==4.59.0


#### export path
export SUMO_HOME=/your python env path/lib/python3.6/site-packages/sumo
export PYTHONPATH=${PYTHONPATH}:/your own folder/root directory of this folder

```


## 3.Train

```
cd onpolicy/scripts/train
python train_sumo.py
```

### Some additional configuration
- Algorithm switch : (`vim onpolicy/scripts/train/train_sumo.py`)
  - IPPO training   :-> args[3] = 'ippo' ;  
  - MAPPO training  :->  args[3] = 'mappo'
- Scenario switch : (`vim onpolicy/envs/sumo_files_new/config.py`)
  - change the absolute path about : `config['environment']['output_path']`
  - change the absolute path about : `config['model_save']['path']`
  - change `config['environment']['port_start']`
  - change `config['environment']['sumocfg_files']`
  
