<p align="center">
<img src="https://github.com/aimagelab/mil4wsi/blob/main/logo.png" width=70% height=70%>
</p>

# Introduction
Welcome to the mil4wsi Framework – your gateway to state-of-the-art Multiple Instance Learning (MIL) model implementations for gigapixel whole slide images. This comprehensive open-source repository empowers researchers, developers, and enthusiasts to explore and leverage cutting-edge MIL techniques.

# Automatic Installation

```bash
conda create -n wsissl python=3.10
conda activate wsissl
conda env update --file environment.yml
```

# Manual Installation
create Environment
```bash
conda create -n ENV_NAME python=3.10
conda activate ENV_NAME
```
1) Install torch; 2) Install pytorch_geometric; 3) Install additional packages for visualization and log as:

```bash
pip install submitit joblib pandas wandb openslide-python==1.2.0 scikit-image wsiprocess scikit-learn matplotlib nystrom_attention
```

Example with torch==2.4.0; cuda==11.8

```bash
conda create -n ENV_NAME python=3.10 && conda activate ENV_NAME && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html && pip install submitit joblib pandas wandb openslide-python==1.2.0 scikit-image wsiprocess scikit-learn matplotlib nystrom_attention
```
# Data Preprocessing

This work uses [CLAM](https://github.com/mahmoodlab/CLAM) to filter out background patches. After the .h5 coordinate generation, use:

- [H5-to-jpg](0-extract_patches/readme.md): It converts .h5 coordinates into jpg images
- [Sort images](1-sort_images/readme.md): It reorganizes patches into hierarchical folders
- [Dino Training](https://github.com/facebookresearch/dino): Given the patches, train dino with the `vit_small` option
- [Feature Extraction](t2/readme.md): It extracts patch features and adjacency matrices
- [Geometric Dataset Conversion](3-prepare-geomDataset/readme.md): It  allows to work with graphs architectures and PyTorch geometric

# Available Models
- MaxPooling
- MeanPooling
- ABMIL
- DSMIL
- DASMIL
- BUFFERMIL
- TRANSMIL
- HIPT

# DASMIL
<p align="center">
<img src="https://github.com/aimagelab/mil4wsi/blob/main/models/dasmil/model.png" width=100% height=100%>
</p>

```
@inproceedings{Bontempo2023_MICCAI,
    author={Bontempo, Gianpaolo and Porrello, Angelo and Bolelli, Federico and Calderara, Simone and Ficarra, Elisa},
    title={{DAS-MIL: Distilling Across Scales for MIL Classification of Histological WSIs}},
    booktitle={Medical Image Computing and Computer Assisted Intervention – MICCAI 2023},
    pages={248--258},
    year=2023,
    month={Oct},
    publisher={Springer},
    doi={https://doi.org/10.1007/978-3-031-43907-0_24},
    isbn={978-3-031-43906-3}
}


@ARTICLE{Bontempo2024_TMI,
  author={Bontempo, Gianpaolo and Bolelli, Federico and Porrello, Angelo and Calderara, Simone and Ficarra, Elisa},
  journal={IEEE Transactions on Medical Imaging}, 
  title={A Graph-Based Multi-Scale Approach With Knowledge Distillation for WSI Classification}, 
  year={2024},
  volume={43},
  number={4},
  pages={1412-1421},
  keywords={Feature extraction;Proposals;Spatial resolution;Knowledge engineering;Graph neural networks;Transformers;Prediction algorithms;Whole slide images (WSIs);multiple instance learning (MIL);(self) knowledge distillation;weakly supervised learning},
  doi={10.1109/TMI.2023.3337549}}
```

## Training

```bash
python main.py --datasetpath DATASETPATH --dataset [cam or lung]
```

## Reproducibility

### Pretrained models

|    DINO Camelyon16    |       DINO LUNG       |
| :-------------------: | :-------------------: |
| [x5](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x5/checkpoint.pth.gz) ~0.65GB | [x5](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x5/checkpoint.pth.gz) ~0.65GB |
| [x10](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x10/checkpoint.pth.gz) ~0.65GB | [x10](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x10/checkpoint.pth.gz) ~0.65GB |
| [x20](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x20/checkpoint.pth.gz) ~0.65GB | [x20](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x20/checkpoint.pth.gz) ~0.65GB |

|    DASMIL Camelyon16    |       DASMIL LUNG       |
| :---------------------: | :---------------------: |
| [model](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/model_cam.pt) ~9MB | [model](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/Lung.pt) ~15MB|
| ACC: 0.945 | ACC: 0.92 |
| AUC: 0.967 | AUC: 0.966 |

### Pytorch Geometric - Extracted Features

|     Camelyon16    |        LUNG       |
| :---------------------: | :---------------------: |
| [Dataset](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/cam_feats.zip) ~4.25GB | [Dataset](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/lung_feats.zip) ~17.5GB |

## Eval

setup checkpoints and datasets paths in utils/experiment.py
then
```bash
python eval.py --datasetpath DATASETPATH --checkpoint CHECKPOINTPATH --dataset [cam or lung]
```

# Contributing

We encourage and welcome contributions from the community to help improve the MIL Models Framework and make it even more valuable for the entire machine-learning community. 
