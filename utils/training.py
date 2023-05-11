from utils.test import test,save_masks_new
from torch.nn import BCEWithLogitsLoss
import torch
import os
import wandb
from argparse import Namespace


def args_to_dict(args):
    d={
        "name":args.modeltype,
        "optimizer":{"lr":args.lr,
                    "weight_decay":args.weight_decay
                },
        "gnn":{"residual":args.residual,
               "num_layers":args.num_layers,
               "dropout":args.dropout,
               "dropout_rate":args.dropout,
               "layer_name":args.layer_name,
               "heads":args.heads
            },
        "training":{"seed":args.seed,
                    "n_epoch":args.n_epoch},
        "dimensions":{"n_classes":args.n_classes,
                      "c_hidden":args.c_hidden,
                      "input_size":args.input_size},
        "dataset":{"scale":args.scale,
                   "dataset_name":args.dataset,
                   "root":args.root},
        "distillation":{"lamb":args.lamb,
                        "beta":args.beta,
                        "tau":args.temperature},

        "main":{"target":args.target,
            "kl":args.kl},

    }
    return d


def train(model: torch.nn.Module,
          trainloader: torch.nn.Module,
          valloader: torch.nn.Module,
          testloader: torch.nn.Module,
          args: Namespace)-> torch.nn.Module:
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

    #config wandb
    run=wandb.init(project=args.project,name=args.scale+"_"+args.modeltype,settings=wandb.Settings(start_method='fork'),tags=[args.tag])
    wandb.config.update(args_to_dict(args))
    #-----
    epochs=args.n_epoch
    model.train()
    model= model.cuda()
    loss_module_instance = BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.000005)

    BestPerformance=0
    #start training
    for epoch in range(epochs):
        for _,data in enumerate(trainloader):
            optimizer.zero_grad()
            data= data.cuda()
            x, edge_index,childof,level = data.x, data.edge_index,data.childof,data.level
            if data.__contains__("edge_index_2") and data.__contains__("edge_index_3"):
                edge_index2,edge_index3=data.edge_index_2,data.edge_index_3
            else:
                edge_index2=None
                edge_index3=None

            results = model(x, edge_index,level,childof,edge_index2,edge_index3)
            bag_label=data.y.float()
            loss= model.compute_loss(loss_module_instance,results,bag_label)
            wandb.log({"loss":loss.item()})
            loss.backward()
            optimizer.step()
        scheduler.step()

        with torch.no_grad():
            #try:
            metrics=test(model,testloader=testloader)
            avg_score_higher_test,avg_score_lower_test,auc_value_higher_test,auc_value_lower_test,predictions,_,labels=metrics

            wandb.log({
                        "acc_higher_test":avg_score_higher_test,
                        "acc_lower_test":avg_score_lower_test,
                        "auc_higher_test":auc_value_higher_test,
                        "epoch":epoch,
                        "lr": scheduler.get_last_lr()[0]
                    })

            performance= (float(auc_value_higher_test)+float(avg_score_higher_test))/2
            if epoch>5:
                if performance > BestPerformance:
                    #metrics=test(model,valloader)
                    wandb.log({"best accuracy":avg_score_higher_test,
                            "best auc higher":auc_value_higher_test,
                            "best auc lower":auc_value_lower_test,
                            })
                    BestPerformance = performance
                    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=labels, preds=predictions,
                        class_names=["short","long"])})
                    save_masks_new(model,testloader)
                    #save model
                    torch.save(model.state_dict(),os.path.join(wandb.run.dir, "model.pt"))
                    wandb.save(os.path.join(wandb.run.dir, "model.pt"))
    wandb.finish()
    return model