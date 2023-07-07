# DAS-MIL

In this work, with the objective of leveraging the full potential of pyramidal structured WSI, we propose a graph-based multi-scale MIL approach, DAS-MIL, that exploits message passing to let information flows across multiple scales. By means of a knowledge distillation schema, the alignment between the latent space representation at different resolutions is encouraged while preserving the diversity in the informative content.

# Installation

```bash
'conda create -n wsissl python=3.9'
'conda env update --file environment.yml
```

# Data preparation
Extract patches (".jpg") with your prefered patch extractor (e.g., CLAM) and save them in different folders for each resolution
## Preprocessing (Hierarchical  patch organization)
Given the folders of the different patch resolutions, it is possible to reorganize them in a hierarchical way 

```bash
via 1-sort_images/sort_hierarchy.py
```

## Feature extraction 
Install DINO in a different folder and save the code location into the environment variable "DINO_REPO". Given the hierarchical patch organization it is possible to extract the embeddings, given the pretrained model path via 

```bash
2-extract_feats/run_with_submitit.py
```
## Pytorch Geometric Adaptation

```bash
3-prepare-geomDataset/prepare_dataset.py
```

# Launch dasmil
```bash
main.py
```


# Download Pretrained Models

At this [Link]() you can download the DINO feature extractors(as well as dasmil model) for both datasets


