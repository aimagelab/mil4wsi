import pandas as pd
import glob
import os

listanames=glob.glob("/mnt/beegfs/work/H2020DeciderFicarra/gbontempo/patches/cam_multi/*")
listanames=[name.split("/")[-1] for name in listanames]
labels=[int(name.split("_")[-1]=="tumor") for name in listanames]
types=[int(name.split("_")[0]=="normal") for name in listanames]

pd.DataFrame({"image":listanames,"label":labels,"phase":types}).to_csv("cam_multi.csv",index=False)
