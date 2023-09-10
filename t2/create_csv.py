import pandas as pd
import glob
import os

listanames=glob.glob("./cam_multi/*")
listanames=[name.split("/")[-1] for name in listanames]
labels=[int(name.split("_")[-1]=="tumor") for name in listanames]
types=["test" if name.split("_")[0]=="test" else "train" for name in listanames]


pd.DataFrame({"image":listanames,"label":labels,"phase":types}).to_csv("cam_multi.csv",index=False)
