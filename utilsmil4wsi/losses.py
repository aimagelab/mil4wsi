# Copyright 2023 Bontempo Gianpaolo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
import torch_geometric.nn as geom_nn

def computeKL(student: torch.Tensor,teacher:torch.Tensor,temperature: int,n_classes:int,add_bias: bool)-> torch.Tensor:
    """compute the knowledge distillation loss

    Args:
        student (torch.Tensor): student representation
        teacher (torch.Tensor): teacher representation
        temperature (int): temperature
        n_classes (int): number of classes inside the representation
        add_bias (bool): introduce bias in the representation for stability reasons

    Returns:
        torch.Tensor: loss
    """

    loss=0
    if n_classes==1:
        #sigmoid
        student= torch.sigmoid(student/temperature)
        teacher= torch.sigmoid(teacher /temperature)
        #concat
        y_student= torch.concat([student,1-student],dim=1)
        y_teacher= torch.concat([teacher,1-teacher],dim=1).detach()
        #add bias for stability reasons
        if add_bias:
            y_student=(y_student + 1e-7)/(1+1e-8)
            y_teacher=(y_teacher + 1e-7)/(1+1e-8)
        kl=F.kl_div((y_student).log(),y_teacher,reduction="batchmean")
        loss+=kl
    else:
        for c in range(n_classes):
            y_student= torch.sigmoid(student[:,c]/temperature)
            y_teacher= torch.sigmoid(teacher[:,c] /temperature)
            y_student= torch.concat([y_student,1-y_student],dim=0)
            y_teacher= torch.concat([y_teacher,1-y_teacher],dim=0).detach()
            if add_bias:
                y_student=(y_student + 1e-7)/(1+1e-8)
                y_teacher=(y_teacher + 1e-7)/(1+1e-8)
            loss+= F.kl_div(y_student.log(),y_teacher,reduction="batchmean",)
    return loss

def computeMSE(lowerscale: torch.Tensor,higherscale:torch.Tensor,childOfIndeces:torch.Tensor)->torch.Tensor:
    """measure mean square error loss between scales

    Args:
        lowerscale (torch.Tensor): node representations at the lower scale
        higherscale (torch.Tensor):  node representations at the higher scale
        childOfIndeces (torch.Tensor): connection between higher and lower scale

    Returns:
        torch.Tensor: loss
    """
    higherscale,_=geom_nn.pool.avg_pool_x(x=higherscale,cluster=childOfIndeces,batch=childOfIndeces)
    mse_loss=F.mse_loss(lowerscale,higherscale.detach())
    return mse_loss

def computeMSE_max(lowerscale: torch.Tensor,higherscale:torch.Tensor,childOfIndeces:torch.Tensor)->torch.Tensor:
    """measure mean square error loss between scales

    Args:
        lowerscale (torch.Tensor): node representations at the lower scale
        higherscale (torch.Tensor):  node representations at the higher scale
        childOfIndeces (torch.Tensor): connection between higher and lower scale

    Returns:
        torch.Tensor: loss
    """
    higherscale,_=geom_nn.pool.max_pool_x(x=higherscale,cluster=childOfIndeces,batch=childOfIndeces)
    mse_loss=F.mse_loss(lowerscale,higherscale.detach())
    return mse_loss

def CELoss(criterion: torch.nn.Module,y_instance_pred:torch.Tensor,prediction_bag:torch.Tensor,bag_label:torch.Tensor,lamb: int=1.0,beta:int=1.0)-> torch.Tensor:
    """apply main loss

    Args:
        criterion (torch.nn.Module): criterion with the main loss is computed
        y_instance_pred (torch.Tensor): patch prediction
        prediction_bag (torch.Tensor): bag prediction
        bag_label (torch.Tensor): ground truth
        lamb (int, optional): contribution hyperparameter controlling the bag prediction contribution. Defaults to 1.0.
        beta (int, optional): contribution hyperparameter controlling the patch prediction contribution. Defaults to 1.0.

    Returns:
        torch.Tensor: main loss
    """

    max_prediction, _ = torch.max(y_instance_pred, 0)
    bag_loss=criterion(prediction_bag.view(1,-1), bag_label.view(1,-1))
    max_loss = criterion(max_prediction.view(1,-1),bag_label.view(1,-1))
    loss = lamb*bag_loss + beta*max_loss
    return loss

def CELossv2(criterion: torch.nn.Module,prediction_bag:torch.Tensor,bag_label:torch.Tensor)-> torch.Tensor:
    """apply main loss

    Args:
        criterion (torch.nn.Module): criterion with the main loss is computed
        y_instance_pred (torch.Tensor): patch prediction
        prediction_bag (torch.Tensor): bag prediction
        bag_label (torch.Tensor): ground truth
        lamb (int, optional): contribution hyperparameter controlling the bag prediction contribution. Defaults to 1.0.
        beta (int, optional): contribution hyperparameter controlling the patch prediction contribution. Defaults to 1.0.

    Returns:
        torch.Tensor: main loss
    """
    bag_loss=criterion(prediction_bag.view(1,-1), bag_label.view(1,-1))
    return bag_loss