# Extract DINO feats 

- download [DINO](https://github.com/facebookresearch/dino) repository
- store repository path into Environment variable ```export DINO_REPO=PATH```  ```export MIL4WSI_PATH=PATH```
- collect a csv with three columns: slide_name (image), label (0/1), phase (train/test)
  example:


| image  | label | phase |
| ------------- | ------------- | ------------|
| name_1  | 0  | train |
| name_2  | 1  | test  |


Launch the feature extraction through submitit!
```
python run_with_submitit.py --extractedpatchespath HIERARCHICAL_PATH --savepath DESTINATION_PATH --pretrained_weights1 CHECKPOINTDINO20x --pretrained_weights2 CHECKPOINTDINO10x --pretrained_weights3 CHECKPOINTDINO5x --propertiescsv CSV_PATH
```

EXAMPLE:
```
python run_with_submitit.py --extractedpatchespath /mnt/beegfs/work/H2020DeciderFicarra/decider/decider_multi --savepath /mnt/beegfs/work/H2020DeciderFicarra/decider/feats/hr --pretrained_weights1 /mnt/beegfs/work/H2020DeciderFicarra/dinodecider20/checkpoint.pth --pretrained_weights2 /mnt/beegfs/work/H2020DeciderFicarra/dinodecider10/checkpoint.pth --pretrained_weights3 /mnt/beegfs/work/H2020DeciderFicarra/dinodecider5/checkpoint.pth --propertiescsv hr.csv
```
