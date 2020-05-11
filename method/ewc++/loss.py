import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
from copy import deepcopy

EPS = 1e-20
def normalize_fn(fisher):
    return (fisher - fisher.min()) / (fisher.max() - fisher.min() + EPS)

class EWCPPLoss(object):
    def __init__(self, model, model_old, alpha=0.9, fisher=None, normalize=True):
        self.model = model
        self.model_old = model_old
        self.model_old_dict = self.model_old.state_dict()

        self.alpha = alpha
        self.normalize = normalize

        if fisher is not None: # initialize as old Fisher Matrix
            self.fisher_old = fisher
            for key in self.fisher_old:
                self.fisher_old[key].requires_grad = False
                self.fisher_old[key] = self.fisher_old[key].cuda()
            self.fisher = deepcopy(fisher)
            if normalize:
                self.fisher_old = {n: normalize_fn(self.fisher_old[n]) for n in self.fisher_old}

        else: # initialize a new Fisher Matrix
            self.fisher_old = None
            self.fisher = {n:torch.zeros_like(p, requires_grad=False).cuda()
                           for n, p in self.model.named_parameters() if p.requires_grad}
    
    def get_fisher(self):
        return self.fisher

    def penalty(self):
        loss = 0
        if self.fisher_old is None:
            return 0.
        for n, p in self.model.named_parameters():
            loss += (self.fisher_old[n] * (p - self.model_old_dict[n]).pow(2)).sum()
        return loss

    def update(self):
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.fisher[n] = (self.alpha * p.grad.data.pow(2) + ((1-self.alpha)*self.fisher[n]))
