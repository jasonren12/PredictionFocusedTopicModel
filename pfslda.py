#@title vsLDA  Theta GD
import torch
import torch.nn as nn
import torch.distributions as ds
from opt_einsum import contract
from util import s_term_normal, s_term_bernoulli

#@title pf-sLDA
class pfsLDA(nn.Module):

  """
  Implementation based on Ren et. al. pf-sLDA
  (https://arxiv.org/pdf/1910.05495.pdf).
  Constrained parameters stored in transformed way to allow
  unconstrained gradient updates.
  """
  def __init__(self, K, V, M, M_val, p, alpha_fixed, device):
    """
    Args: 
    K: # topics, 
    V: Vocab size, 
    M: # docs,
    M_val: # val docs,
    p : switch prior
    alpha_fixed: if true fix alpha
    devcie: specify cpu or gpu
    """
    super(pfsLDA, self).__init__()
    self.name = 'pfslda'
    self.K = K      
    self.V = V
    self.M = M
    self.M_val = M_val
    self.epsilon = 0.0000001
    self.alpha_fixed = alpha_fixed
    self.device = device

    # model parameters
    alpha = torch.ones(self.K).to(device) if self.alpha_fixed else \
      ds.Exponential(1).sample([self.K])        
    # beta stored pre-softmax (over V)
    beta = ds.Exponential(1).sample([self.K, self.V])
    beta = beta / beta.sum(dim=1, keepdim=True)   
    # pi stored pre-softmax (over V)
    pi = ds.Exponential(1).sample([self.V])
    pi = pi / pi.sum()  
    eta = ds.Normal(0,1).sample([self.K])
    # delta stored pre-exponentiated
    delta = ds.Normal(0,1).sample().abs() 
    
    # variational parameters
    gamma = torch.ones(self.M, self.K)
    gamma_val = torch.ones(self.M_val, self.K)
    # phi stored pre softmax (over K)
    phi = torch.ones(self.M, self.K, self.V)
    phi_val = torch.ones(self.M_val, self.K, self.V)
    # varphi stored pre-sigmoid
    varphi = torch.ones(self.V) * p 

    self.alpha = alpha if self.alpha_fixed else nn.Parameter(alpha)  
    self.beta = nn.Parameter(beta)
    self.gamma = nn.Parameter(gamma)
    self.phi = nn.Parameter(phi)   
    self.eta = nn.Parameter(eta)
    self.delta = nn.Parameter(delta)
    self.pi = nn.Parameter(pi)
    self.varphi = nn.Parameter(varphi)    
    self.phi_val = nn.Parameter(phi_val)
    self.gamma_val = nn.Parameter(gamma_val)
    self.p = p
      
 
  def ELBO(self, W_batch, phi_batch, gamma_batch, y_batch, version='real'):
    """
    Computes pf-sLDA ELBO:
    See appendix of https://arxiv.org/pdf/1910.05495.pdf for details
    first_term: log p(theta|alpha)
    second_term: log p(z|theta)
    third_term: log p(w|z,beta)
    fourth_term: log p(xi|p)
    fifth_term: log q(theta|gamma)
    sixth_term: log q(z|phi)
    seventh_term: log q(xi|varphi)
    s_term: log p(y|theta)
    """
    M = W_batch.shape[0]
    N_tot = W_batch.sum()
    ss = torch.digamma(gamma_batch) - \
      torch.digamma(gamma_batch.sum(dim=1, keepdim=True))
    
    # transform constrained parameters to be valid
    phi = phi_batch.softmax(dim=1)
    beta = self.beta.softmax(dim=1)
    pi = self.pi.softmax(dim=0)
    varphi = self.varphi.sigmoid() 
    p = self.p.sigmoid()
    delta = self.delta ** 2
         
    first_term = M \
      * (torch.lgamma(self.alpha.sum()) - torch.lgamma(self.alpha).sum()) \
      + contract('mk,k->', ss, self.alpha - 1, backend='torch')
    
    second_term = contract(
      'mkv,mk,mv->', 
      phi, ss, W_batch, 
      backend='torch'
    )  
                      
    third_term1 = contract(
      'mkv,mv,kv,v->', 
      phi, W_batch, beta.log(), varphi, 
      backend='torch'
    ) 
    third_term2 = contract(
      'mv,v,v->', 
      W_batch, pi.log(), varphi, 
      backend='torch'
    )
    third_term3 = contract(
      'mv,v->', 
      W_batch, pi.log(), 
      backend='torch'
    )
    third_term = third_term1 - third_term2 + third_term3
   
    fourth_term1 = contract(
      'mv,v->', 
      W_batch, varphi, 
      backend='torch'
    ) 
    fourth_term2 = contract(
      'mv,v->', 
      W_batch, 1 - varphi, 
      backend='torch'
    ) 
    fourth_term = p.log() * fourth_term1 + (1-p).log() * fourth_term2
    
    fifth_term = torch.lgamma(gamma_batch.sum(dim=1)).sum() - \
      torch.lgamma(gamma_batch).sum() + \
      contract('mk,mk->', ss, gamma_batch - 1, backend='torch')  

    sixth_term = contract(
      'mkv,mkv,mv->', 
      phi, phi.log(), W_batch, 
      backend='torch'
    )
              
    # to prevent overflows in log for values approaching 0/1
    orig_varphi = varphi
    if varphi.min() <= 0:
      c = varphi.min().detach()
      varphi = varphi - c + self.epsilon
    seventh_term1 = contract(
      'mv,v,v->', 
      W_batch, varphi, varphi.log(), 
      backend='torch'
    )
    varphi = orig_varphi
    if varphi.max() >= 1:
      c = varphi.max().detach()
      varphi = varphi - (c - 1) - self.epsilon
    seventh_term2 = contract(
      'mv,v,v->', 
      W_batch, 1 - varphi, (1 - varphi).log(), 
      backend='torch'
    )
    seventh_term = seventh_term1 + seventh_term2

    if version == 'real':
      s_term = s_term_normal(y_batch, gamma_batch, self.eta, delta, M)
    elif version == 'binary':
      s_term = s_term_bernoulli(y_batch, gamma_batch, self.eta)
    
    return first_term + second_term + third_term + fourth_term \
      - fifth_term - sixth_term - seventh_term + s_term

  
  # can also batch if needed, uses theta map as in pc-slda
  # (http://proceedings.mlr.press/v84/hughes18a/hughes18a.pdf)  
  def pred(self, W, y=None):
    theta_maps = self.theta_map(W)
    preds = torch.mv(theta_maps, self.eta)
    return theta_maps, preds
  
  
  # calculate theta map with SGD on the posterior of theta
  def theta_map(self, W, num_epochs = 500, lr = 0.005):      
    theta_maps = torch.ones(
      (W.shape[0], self.K), 
      requires_grad = True, 
      device=self.device
    )
    opt = torch.optim.Adam([theta_maps], lr = lr)
    for i in range(num_epochs):
      opt.zero_grad()
      score = self.theta_post(W, theta_maps)
      loss = -1 * score
      loss.sum().backward()
      opt.step()
    return theta_maps.softmax(dim=1)
        
  
  # calculate the posterior of theta
  def theta_post(self, W, theta):
    bl = contract(
      'kv,mk->mv', 
      self.beta.softmax(dim=1), 
      theta.softmax(dim=1), 
      backend='torch'
    )
    
    out1 = contract(
      'v,mv,mv->m', 
      self.varphi.sigmoid(), W, bl.log(), 
      backend='torch'
    ) 
    out2 = torch.mv(theta.softmax(dim=1).log(), self.alpha)
    
    return out1 + out2

 


    