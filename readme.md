<p align="center">
<img src="https://github.com/aimagelab/mil4wsi/blob/main/logo.png" width=70% height=70%>
</p>

# Introduction
Welcome to the mil4wsi Framework – your gateway to state-of-the-art Multiple Instance Learning (MIL) model implementations for gigapixel whole slide images. This comprehensive open-source repository empowers researchers, developers, and enthusiasts to explore and leverage cutting-edge MIL techniques.

# Installation

```bash
conda create -n wsissl python=3.9
conda activate wsissl
conda env update --file environment.yml
```

# Data Preprocessing

This work uses [CLAM](https://github.com/mahmoodlab/CLAM) to filter out background patches. After the .h5 coordinate generation, use:

- [H5-to-jpg](0-extract_patches/readme.md): It converts .h5 coordinates into jpg images
- [Sort images](1-sort_images/readme.md): It reorganizes patches into hierarchical folders
- [Dino Training](https://github.com/facebookresearch/dino): Given the patches, train dino with the `vit_small` option
- [Feature Extraction](2-extract_feats/readme.md): It extracts patch features and adjacency matrices
- [Geometric Dataset Conversion](3-prepare-geomDataset/readme.md): It  allows to work with graphs architectures and PyTorch geometric

# Available Models
- [MaxPooling](https://github.com/aimagelab/mil4wsi/tree/main/models/maxpooling)
- [MeanPooling](https://github.com/aimagelab/mil4wsi/tree/main/models/meanpooling)
- [ABMIL](https://github.com/aimagelab/mil4wsi/tree/main/models/abmil)
- [DSMIL](https://github.com/aimagelab/mil4wsi/tree/main/models/dsmil)
- [DASMIL](https://github.com/aimagelab/mil4wsi/tree/main/models/dasmil)
- [BUFFERMIL](https://github.com/aimagelab/mil4wsi/tree/main/models/buffermil)
- [TRANSMIL](https://github.com/aimagelab/mil4wsi/tree/main/models/transmil)
- [HIPT](https://github.com/aimagelab/mil4wsi/tree/main/models/hipt)

# DASMIL
<p align="center">
<img src="https://github.com/aimagelab/mil4wsi/blob/main/models/dasmil/model.png" width=70% height=70%>
</p>
`
@inproceedings{Bontempo2023,
  title={{DAS-MIL: Distilling Across Scales for MIL Classification of Histological WSIs}},
  author={Bontempo, Gianpaolo and Porrello, Angelo and Bolelli, Federico and Calderara, Simone and Ficarra, Elisa},
  booktitle={{Medical Image Computing and Computer Assisted Intervention – MICCAI 2023}},
  year={2023}
}
`

## Training

```bash
python main.py --datasetpath DATASETPATH --dataset [cam or lung]
```

## Reproducibility

### Pretrained models

|    DINO Camelyon16    |       DINO LUNG       |
| :-------------------: | :-------------------: |
| [x5](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x5/checkpoint.pth.gz) ~0.65GB | [x5](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/lung/dino/x5/checkpoint.pth.gz) ~0.65GB |
| [x10](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x10/checkpoint.pth.gz) ~0.65GB | [x10](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x10/checkpoint.pth.gz) ~0.65GB |
| [x20](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x20/checkpoint.pth.gz) ~0.65GB | [x20](https://ailb-web.ing.unimore.it/publicfiles/miccai_dasmil_checkpoints/dasmil/camelyon16/dino/x20/checkpoint.pth.gz) ~0.65GB |

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
