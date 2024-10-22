from utilsmil4wsi.test import test
from torch.nn import BCEWithLogitsLoss
import torch
import os
import wandb
from argparse import Namespace
import time


def args_to_dict(args):
    # Convert argparse.Namespace object to a dictionary
    d = {
        "name": args.modeltype,
        "optimizer": {"lr": args.lr,
                      "weight_decay": args.weight_decay
                      },
        "gnn": {"residual": args.residual,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "dropout_rate": args.dropout,
                "layer_name": args.layer_name,
                "heads": args.heads
                },
        "training": {"seed": args.seed,
                     "n_epoch": args.n_epoch},
        "dimensions": {"n_classes": args.n_classes,
                       "c_hidden": args.c_hidden,
                       "input_size": args.input_size},
        "dataset": {"scale": args.scale,
                    "dataset_name": args.dataset,
                    "dataset_path": args.datasetpath,
                    "root": args.root},
        "distillation": {"lamb": args.lamb,
                         "beta": args.beta,
                         "tau": args.temperature},

        "main": {"target": args.target,
                 "kl": args.kl},

    }
    return d


def train(model: torch.nn.Module,
          trainloader: torch.nn.Module,
          valloader: torch.nn.Module,
          testloader: torch.nn.Module,
          args: Namespace) -> torch.nn.Module:
    """train model

    Args:
        model (torch.nn.Module): model
        trainloader (torch.nn.Module): train loader
        valloader (torch.nn.Module): validation loader
        testloader (torch.nn.Module): test loader
        args (Namespace): configurations

    Returns:
        torch.nn.Module: trained model
    """

    # Initialize wandb run
    run = wandb.init(project=args.project, name=args.wandbname, save_code=True,
                     settings=wandb.Settings(start_method='fork'), tags=[args.tag])
    wandb.config.update(args)
    # -----

    epochs = args.n_epoch
    model.train()
    model = model.cuda()
    loss_module_instance = BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(
        0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.000005)
    # Test the initial model
    with torch.no_grad():
        start_test = time.time()
        metrics = test(model, testloader=testloader)
        end_test = time.time()
        avg_score_higher_test, avg_score_lower_test, auc_value_higher_test, auc_value_lower_test, predictions, _, labels = metrics

        wandb.log({
            "acc_higher_test": avg_score_higher_test,
            "acc_lower_test": avg_score_lower_test,
            "auc_higher_test": auc_value_higher_test,
            "epoch": -1,
            "lr": scheduler.get_last_lr()[0]
        })
    BestPerformance = 0
    # Start training
    for epoch in range(epochs):
        start_training = time.time()
        if hasattr(model,"preloop"):
            model.preloop(epoch,trainloader)
        # Iterate over the training data
        for _, data in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            data = data.cuda()
            x, edge_index, childof, level = data.x, data.edge_index, data.childof, data.level
            # Check if additional edge indices are present
            if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
                edge_index2, edge_index3 = data.edge_index_2, data.edge_index_3
            else:
                edge_index2 = None
                edge_index3 = None

            try:
                results = model(x, edge_index, level, childof,edge_index2, edge_index3)
            except:
                continue
            bag_label = data.y.float()
            loss = model.compute_loss(loss_module_instance, results, bag_label)
            loss.backward()
            optimizer.step()
        end_training = time.time()
        scheduler.step()

        if epoch > 15:
            with torch.no_grad():
                # try:
                start_test = time.time()
                metrics = test(model, testloader=testloader)
                end_test = time.time()
                avg_score_higher_test, avg_score_lower_test, auc_value_higher_test, auc_value_lower_test, predictions, _, labels = metrics

                wandb.log({
                    "acc_higher_test": avg_score_higher_test,
                    "acc_lower_test": avg_score_lower_test,
                    "auc_higher_test": auc_value_higher_test,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0]
                })

                performance = float(auc_value_higher_test)

                if performance > BestPerformance:
                    # metrics=test(model,valloader)
                    # save_masks_new(model,testloader)
                    wandb.log({"best accuracy": avg_score_higher_test,
                               "best auc higher": auc_value_higher_test,
                               "best auc lower": auc_value_lower_test,
                               })
                    BestPerformance = performance
                    model.eval()
                    torch.save(model.state_dict(), os.path.join(
                        wandb.run.dir, "model.pt"))
                    wandb.save(os.path.join(wandb.run.dir, "model.pt"))

        print("training_time: %s  seconds" % (end_training-start_training))
        print("test_time: %s  seconds" % (end_test-start_test))

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "final.pt"))
    wandb.save(os.path.join(wandb.run.dir, "final.pt"))
    wandb.finish()
    return model
