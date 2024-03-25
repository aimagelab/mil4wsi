# Patch Extraction
 [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/create_patches_fp.py) extracts .h5 coordinates of each patch.


```
python CLAM/create_patches_fp.py --source  /mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022 --save_dir /mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/H168_prova/easy/x20 --patch_size 256 --step_size 128 --seg --patch --stitch --patch_level 0

```

Since this work requires jpg, this [script](https://github.com/aimagelab/mil4wsi/blob/main/0-extract_patches/convert_h5_to_jpg.py) converts the coordinates into images.

Use this script inside the [CLAM](https://github.com/mahmoodlab/CLAM) repository



# Launch

```
python /work/H2020DeciderFicarra/fmiccolis/WP2/mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir /mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/HR_pool/hard --source_dir /mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022 --slide_ext .mrxs
```
N.B. Do not forget the dot as the first character of the extension

python /work/H2020DeciderFicarra/fmiccolis/WP2/mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir /mnt/beegfs/work/H2020DeciderFicarra/fmiccolis/WP2/CLAM_output/HR_pool/hard/x5 --source_dir /mnt/beegfs/work/H2020DeciderFicarra/DECIDER/WSI_24_11_2022 --slide_ext .mrxs




Da mandare : 
python /work/H2020DeciderFicarra/fmiccolis/WP2/mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir /work/H2020DeciderFicarra/fmiccolis/PRINN/CLAM_patches/x20_patchsize256/easy_contour/x20 --source_dir /work/H2020DeciderFicarra/fmiccolis/PRINN/data --slide_ext .ndpi

python /work/H2020DeciderFicarra/fmiccolis/WP2/mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir /work/H2020DeciderFicarra/fmiccolis/PRINN/CLAM_patches/x20_patchsize256/easy_contour/x10 --source_dir /work/H2020DeciderFicarra/fmiccolis/PRINN/data --slide_ext .ndpi

python /work/H2020DeciderFicarra/fmiccolis/WP2/mil4wsi/0-extract_patches/convert_h5_to_jpg.py --output_dir /work/H2020DeciderFicarra/fmiccolis/PRINN/CLAM_patches/x20_patchsize256/easy_contour/x5 --source_dir /work/H2020DeciderFicarra/fmiccolis/PRINN/data --slide_ext .ndpi


