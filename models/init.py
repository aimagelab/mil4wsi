import torch
from models.dasmil import DASMIL


multi_scales_models = {
    "WithGraph_y_Higher_kl_Lower":  {"model": DASMIL, "kl": "lower", "target": "higher"},
}

single_scales_models = {}


def selectModel(args):

    state_dict_weights = torch.load(
        args.checkpoint, map_location=torch.device('cpu'))

    print("model " + args.modeltype)
    if len(args.scale) > 1:
        d = multi_scales_models[args.modeltype]
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
    model = model.cuda()
    memory_usage = torch.cuda.memory_allocated(
        device="cuda") / 1e9  # in gigabytes

    print(f"Memory usage: {memory_usage} GB")
    return model
