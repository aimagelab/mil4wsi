# Extract DINO feats 

- download [DINO](https://github.com/facebookresearch/dino) repository
- store repository path into Environment variable ```export DINO_REPO=PATH```
- collect a csv with three columns: slide_name, label (0/1), phase (train/test)
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
python run_with_submitit.py --extractedpatchespath /work/H2020DeciderFicarra/WP2/HR/step1_output/ --savepath /work/H2020DeciderFicarra/WP2/HR/step2_output/x5-x20 --pretrained_weights1 /work/H2020DeciderFicarra/dinodecider20/checkpoint.pth --pretrained_weights2 /work/H2020DeciderFicarra/dinodecider10/checkpoint.pth --pretrained_weights3 /work/H2020DeciderFicarra/dinodecider5/checkpoint.pth --propertiescsv /work/H2020DeciderFicarra/fmiccolis/WP2/HR/PDS/PDS_embeddings_extraction.csv --levels 1 3


python run_with_submitit.py --extractedpatchespath /work/H2020DeciderFicarra/WP2/HR/step1_output/ --savepath /work/H2020DeciderFicarra/WP2/HR/step2_output/x5 --pretrained_weights1 /work/H2020DeciderFicarra/dinodecider20/checkpoint.pth --pretrained_weights2 /work/H2020DeciderFicarra/dinodecider10/checkpoint.pth --pretrained_weights3 /work/H2020DeciderFicarra/dinodecider5/checkpoint.pth --propertiescsv /work/H2020DeciderFicarra/WP2/HR/HR/PDS/PDS_embeddings_extraction.csv --levels 1



,2,3 specifica i livelli di risoluzione per cui estrarre (1=x5, 3=x20)
```
