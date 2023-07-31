import copy

def launch_DASMIL_cam(args):
    args1= copy.copy(args)
    args1.modeltype="DASMIL"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.5
    args1.scale="23"
    args1.dataset="cam"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.005
    args1.c_hidden=256
    args1.layer_name="GAT"
    args1.residual=False
    args1.datasetpath="/mnt/beegfs/work/H2020DeciderFicarra/gbontempo/feats/1_XDASMIL/camGraph_23"
    args1.checkpoint="/mnt/beegfs/work/H2020DeciderFicarra/gbontempo/dasmil/camelyon16/mil/model_cam.pt"
    args1.project="wandbproject"
    return [args1]


def launch_DASMIL_lung(args):
    args1= copy.copy(args)
    args1.modeltype="DASMIL"
    args1.lamb=1
    args1.beta=1
    args1.temperature=1.5
    args1.scale="13"
    args1.dataset="lung"
    args1.seed=42
    args1.lr=0.0002
    args1.weight_decay=0.005
    args1.c_hidden=384
    args1.layer_name="GAT"
    args1.residual=True
    args1.datasetpath="/mnt/beegfs/work/H2020DeciderFicarra/gbontempo/feats/1_XDASMIL/lungGraph_13"
    args1.checkpoint="/homes/gbontempo/DASMIL/Lung.pt"
    args1.project="wandbproject"
    return [args1]
