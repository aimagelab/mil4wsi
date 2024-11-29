import torch
from models.dasmil.dasmil import DASMIL
from models.dsmil.dsmil import DSMIL
from models.abmil.abmil import ABMIL
from models.maxpooling.maxpooling import MaxPooling
from models.meanpooling.meanpooling import MeanPooling
from models.transmil.transmil import TransMIL
from models.buffermil.buffermil import Buffermil
#from models.hipt.hipt import HIPT_LGP_FC

# Dictionary of multi-scale models

multi_scales_models = {
    "DASMIL":  {"model": DASMIL, "kl": "lower", "target": "higher"},
    # "hipt": {"model":HIPT_LGP_FC,"kl":None,"target":"higher"},
}

# Dictionary of single-scale models
single_scales_models = {
    "DSMIL":  {"model": DSMIL, "kl": None, "target": "higher"},
    "ABMIL":  {"model": ABMIL, "kl": None, "target": "higher"},
    "MaxPooling":  {"model": MaxPooling, "kl": None, "target": "higher"},
    "MeanPooling":  {"model": MeanPooling, "kl": None, "target": "higher"},
    "TransMIL":  {"model": TransMIL, "kl": None, "target": "higher"},
    #"Buffermil":  {"model": Buffermil, "kl": None, "target": "higher"},
}


def selectModel(args):
    """
    Selects the appropriate model based on the provided arguments.

    Args:
        args (Namespace): Command-line arguments.

    Returns:
        model (torch.nn.Module): The selected model.
    """
    # Load the state dict weights from the checkpoint

    if args.checkpoint is not None:
        state_dict_weights = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    else:
        state_dict_weights= None
    print("model "+ args.modeltype)
    if len(args.scale)>1:
        d= multi_scales_models[args.modeltype]
    else:
        d = single_scales_models[args.modeltype]

    args.kl = d["kl"]
    args.target = d["target"]
    model = d["model"](args, state_dict_weights)
    try:
        model.load_state_dict(state_dict_weights, strict=False)
    except:
        print("error loading")

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    # Move the model to GPU
    model = model.cuda()
    # Calculate and print memory usage in GB
    memory_usage = torch.cuda.memory_allocated(
        device="cuda") / 1e9  # in gigabytes

    print(f"Memory usage: {memory_usage} GB")
    return model
