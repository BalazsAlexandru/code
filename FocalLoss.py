import torch
import torch.nn.functional as torchFunctional

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignoreIndex=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignoreIndex = ignoreIndex


    def forward(self, inputs, targets):

        # Compute the cross-entropy loss
        ce = torchFunctional.cross_entropy(inputs, targets, weight=self.alpha, ignoreIndex=self.ignoreIndex, reduction="none")
        pt = torch.exp(-ce)

        # Compute the focal loss using the formula
        focalLoss = (1 - pt) ** self.gamma * ce

        return focalLoss.mean()