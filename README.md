## NOTE

This repository is still not fully built. 

 In the next two months, the documentation and code structure will be gradually completed.
 
## :hammer_and_wrench: Installation

### Environment Setup

**!!! Some repositories are actually in this repository, can directly install dependencies !!!**

#### 1. **Preparing conda env**

Assuming you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) installed, let's prepare a conda env:
```
conda_env_name=h3vlfm_world
conda create -n $conda_env_name python=3.9 cmake=3.14.0
conda activate $conda_env_name
```

Install proper version of torch:
```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 2. **Habitat-sim & Habitat-lab**
Following [Habitat-lab](https://github.com/facebookresearch/habitat-lab.git)'s instruction, install Habitat-sim:
```
conda install habitat-sim=0.3.1 withbullet -c conda-forge -c aihabitat
```
Then install Habitat-lab

##### Clone Project
<!-- ```
git clone --branch v0.3.1 https://github.com/facebookresearch/habitat-lab.git
``` -->

##### Install
```
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..
```

#### 3. **Mobile-SAM**
Following [Mobile-SAM](https://github.com/ChaoningZhang/MobileSAM)'s instruction:

##### Install
```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

#### 4. **GroundingDINO**

Following [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)'s instruction:

##### Clone Project
<!-- ```
git clone https://github.com/IDEA-Research/GroundingDINO.git
``` -->

##### Install 

May define CUDA_HOME <= 11.8
export CUDA_HOME=/path/to/cuda-11.8

```
cd GroundingDINO/
pip install -e . --no-dependencies
```
Then place the pretrained model weights:
```
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

#### 5. **BLIP2**

##### Install
```
pip install salesforce-lavis==1.0.2
```

#### 6. **RedNet**
We use finetuned version of semantic segmentation model [RedNet](https://github.com/JindongJiang/RedNet). 

Therefore, you need to download the [segmentation model](https://drive.google.com/file/d/1U0dS44DIPZ22nTjw0RfO431zV-lMPcvv/view) in RedNet/model path.

#### 7. D-Fine
```
git clone https://github.com/Peterande/D-FINE.git
```

#### 8. Places365
```
git clone https://github.com/CSAILVision/places365.git
```

#### 9. Others
```
pip install flask
pip install open3d 
pip install dash
pip install scikit-learn 
pip install joblib 
pip install seaborn
pip install faster_coco_eval
pip install calflops
pip install flash-attn --no-build-isolation
pip install modelscope
pip install opencv-python==4.10.0.84
pip install transformers==4.37.0
pip install openpyxl
pip install supervision==0.25.1
pip install yapf==0.43.0
```


### Datasets Setup

- Download Scene & Episode Datasets

Following the instructions for **HM3D** and **MatterPort3D** in Habitat-lab's [Datasets.md](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md).

- Locate Datasets


The file structure should look like this:
```
data
└── datasets
    └── objectnav
        ├── hm3d
        │   └── v1
        │        ├── train
        │        │    ├── content
        │        │    └── train.json.gz
        │        └── val
        │             ├── content
        │             └── val.json.gz
        └── mp3d
            └── v1
                 ├── train
                 │    ├── content
                 │    └── train.json.gz
                 └── val
                      ├── content
                      └── val.json.gz
```

### Command

Run the following commands:
```
./scripts/launch_vlm_servers_qwen25_gdino_with_ram.sh
python -u -m falcon.run --config-name=experiments/qwen25_gdino_objectnav_hm3d_debug_scene.yaml habitat_baselines.num_environments=1 > debug/20250219/eval_llm_single_floor_gdino.log 2>&1
```