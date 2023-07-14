# DAS-MIL

In this work, intending to leverage the full potential of pyramidal structured WSI, we propose a graph-based multi-scale MIL approach, DAS-MIL, that exploits message passing to let information flows across multiple scales. Employing a knowledge distillation schema, the alignment between the latent space representation at different resolutions is encouraged while preserving the diversity in the informative content.

# Installation

```bash
conda create -n wsissl python=3.9
conda activate wsissl
conda env update --file environment.yml
```

# Data Preprocessing

This work uses [CLAM](https://github.com/mahmoodlab/CLAM) to filter out background patches. After the .h5 coordinate generation, use:

- [H5-to-jpg](0-extract_patches/readme.md) to convert h5 coordinates into jpg images
- [Sort images](1-sort_images/readme.md) to reorganize patches into hierarchical folders
- [Dino Training](https://github.com/facebookresearch/dino): Given the patches, train dino with the `vit_small` option
- [Feature Extraction](2-extract_feats/readme.md): extract patch features and adjacency matrices
- [Geometric Dataset Conversion](3-prepare-geomDataset/readme.md): to easily work with graphs architectures

# Training

```bash
python main.py --datasetpath PATH --model DASMIL
```

# Reproducibility

## Pretrained models

|    DINO Camelyon16    |       DINO LUNG       |
| :-------------------: | :-------------------: |
| [x5]() Not Available  | [x5]() Not Available  |
| [x10]() Not Available | [x10]() Not Available |
| [x20]() Not Available | [x20]() Not Available |

|    DASMIL Camelyon16    |       DASMIL LUNG       |
| :---------------------: | :---------------------: |
| [model]() Not Available | [model]() Not Available |

## Pytorch Geometric - Extracted Features

|        Camelyon16         |           LUNG            |
| :-----------------------: | :-----------------------: |
|   [x5]() Not Available    |   [x5]() Not Available    |
|   [x10]() Not Available   |   [x10]() Not Available   |
|   [x20]() Not Available   |   [x20]() Not Available   |
| [x5-x20]() Not Available  | [x5-x20]() Not Available  |
| [x10-x20]() Not Available | [x10-x20]() Not Available |

# TODOs

- [ ] Refactor
- [ ] Upload checkpoints
- [ ] Upload feats

# References
