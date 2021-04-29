from transformers import Trainer
import torch.nn as nn
import torch.nn.functional as F
import torch

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs,return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # print('logits : ',outputs.logits)
        loss_fct = Cross_FocalLoss()
        loss = loss_fct(logits,labels)

        return (loss, outputs) if return_outputs else loss


class Cross_FocalLoss(nn.Module):
    def __init__(self, weight=None,
    gamma=2., reduction='mean', **kwargs):

        nn.Module.__init__(self)
        self.weight=weight
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs,targets):

        loss_fct_cross = nn.CrossEntropyLoss()
        loss_cross = loss_fct_cross(inputs, targets)

        loss_fct_focal = FocalLoss()
        loss_focal = loss_fct_focal(inputs, targets)

        return loss_cross*0.75+loss_focal*0.25

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight) 
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

