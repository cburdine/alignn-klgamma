import torch

class KLGammaLoss(torch.nn.module):
    
    def __init__(self):
        super(KLGammaLoss,self).__init__()

    def forward(self, outputs, labels):
        return torch.sum(torch.reduce(outputs - labels)**2)
