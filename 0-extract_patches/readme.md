# Patch Extraction
 [CLAM](https://github.com/mahmoodlab/CLAM/blob/master/create_patches_fp.py) extracts .h5 coordinates of each patch.

Since this work  requires jpg, this [script](https://github.com/aimagelab/mil4wsi/blob/main/0-extract_patches/convert_h5_to_jpg.py) converts the coordinates into images.

Use this script inside the [CLAM](https://github.com/mahmoodlab/CLAM) repository



This script is designed to extract patches for a single resolution only. To accommodate multiple scales, you must rerun CLAM and this script for each desired scale. Once you have the patches for all scales, you can hierarchically organize them using the 1_sort_images script, ensuring you provide the correct paths for each scale.

Besides, use the following as the config file in CLAM create_patches_fp.py (presets/*.csv).

```
seg_level,sthresh,mthresh,close,use_otsu,a_t,a_h,max_n_holes,vis_level,line_thickness,white_thresh,black_thresh,use_padding,contour_fn,keep_ids,exclude_ids
-1,8,7,4,TRUE,25,4,8,-1,100,5,50,TRUE,four_pt,none,none
```

However, the contour_fn in this file is relatively lenient to patches with too much blank. Change it according to your applications. Reference: https://github.com/mahmoodlab/CLAM/blob/master/docs/README.md
