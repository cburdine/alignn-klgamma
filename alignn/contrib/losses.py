import torch

def _kl_gamma_integrand(a,b,c,d):
    return -c*d/a - (b*torch.log(a) + torch.lgamma(b)) + \
            (b-1)*(torch.digamma(d) + torch.log(c))

class KLGammaLoss(torch.nn.Module):
    
    def __init__(self, min_mu=1e-10, min_theta=1e-10):
        super(KLGammaLoss,self).__init__()

        self.min_mu = min_mu
        self.min_theta = min_theta

    def forward(self, outputs, labels):
        
        # extract gamma distribution parameters:
        mu_q = torch.clamp(outputs[:,0],    min=self.min_mu)
        theta_q = torch.clamp(outputs[:,1], min=self.min_theta)
        mu_p = torch.clamp(labels[:,0],     min=self.min_mu)
        theta_p = torch.clamp(labels[:,1],  min=self.min_theta)
        
        # compute kappa parameters from mu and theta:
        kappa_q = mu_q / theta_q
        kappa_p = mu_p / theta_p

        # compute the kl divergence of true gamma distribution (p)
        # from predicted gamma distribution (q):
        batch_loss = (kappa_q - 1)*torch.digamma(kappa_q) - torch.log(theta_q) \
                     - kappa_q - torch.lgamma(kappa_q) + torch.lgamma(kappa_p) \
                     + kappa_p * torch.log(theta_p) \
                     - (kappa_p - 1)*(torch.digamma(kappa_q) + torch.log(theta_q)) \
                     + theta_q*kappa_q/theta_p
        
        return torch.mean(torch.log(
                    torch.maximum(1e-30,batch_loss)))
