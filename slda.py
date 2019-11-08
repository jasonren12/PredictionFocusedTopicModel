import torch
import torch.nn as nn
import torch.distributions as ds
from opt_einsum import contract
from sklearn.metrics import roc_auc_score as auc
from util import s_term_normal, s_term_bernoulli

class sLDA(nn.Module):
  """
  Implementation based on McAuliffe and Blei Supervised Topic Models
  (https://arxiv.org/pdf/1003.0783.pdf).
  Optimized with SGD on ELBO instead of coordinate ascent.
  Constrained parameters stored in transformed way to allow
  unconstrained gradient updates.
  """
  def __init__(self, K, V, M, M_val, alpha_fixed, device): 
    """
    Args: 
    K: # topics, 
    V: Vocab size, 
    M: # docs,
    M_val: # val docs,
    alpha_fixed: if true fix alpha
    device: specify cpu or gpu
    """
    super(sLDA, self).__init__()
    self.name = 'slda'
    self.K = K    
    self.V = V
    self.M = M
    self.epsilon = 0.0000001
    self.alpha_fixed = alpha_fixed
    self.device = device
   
    # model parameters
    alpha = torch.ones(self.K).to(device) if self.alpha_fixed else \
      ds.Exponential(1).sample([self.K]) 
    # beta stored pre-softmax (over V)
    beta = ds.Exponential(1).sample([self.K, self.V])
    beta = beta / beta.sum(dim=1, keepdim=True)   
    eta = ds.Normal(0,1).sample([self.K])
    # delta stored pre-exponentiated
    delta = ds.Normal(0,1).sample().abs()   
    
    # variational parameters
    gamma = torch.ones((self.M, self.K))
    gamma_val = torch.ones((M_val, self.K))
    # phi stored pre-softmax (over K)
    phi = torch.ones((self.M, self.K, self.V))
    phi_val = torch.ones((M_val, self.K, self.V))
    
    self.alpha = alpha if self.alpha_fixed else nn.Parameter(alpha)  
    self.beta = nn.Parameter(beta)
    self.gamma = nn.Parameter(gamma)
    self.phi = nn.Parameter(phi)   
    self.eta = nn.Parameter(eta)
    self.delta = nn.Parameter(delta)
    self.phi_val = nn.Parameter(phi_val)
    self.gamma_val = nn.Parameter(gamma_val)


  def ELBO(self, W_batch, phi_batch, gamma_batch, y_batch, version='real'):
    """
    Computes sLDA ELBO:
    first_term: log p(theta|alpha)
    second_term: log p(z|theta)
    third_term: log p(w|z,beta)
    fourth_term: log q(theta|gamma)
    fifth_term: log q(z|phi)
    s_term: log p(y|theta)
    """
    M = W_batch.shape[0]
    ss = torch.digamma(gamma_batch) \
      - torch.digamma(gamma_batch.sum(dim=1, keepdim=True))
    
    # transform constrained parameters to be valid
    phi = phi_batch.softmax(dim=1)
    beta = self.beta.softmax(dim=1)
    delta = self.delta.exp()

    first_term = M * (torch.lgamma(self.alpha.sum()) \
      - torch.lgamma(self.alpha).sum()) \
      + contract('mk,k->', ss, self.alpha - 1, backend='torch')
    
    second_term = contract(
      'mkv,mk,mv->', 
      phi, ss, W_batch, 
      backend='torch'
    ) 

    third_term = contract(
      'mkv,mv,kv->', 
      phi, W_batch, beta.log(), 
      backend='torch'
    ) 
    
    fourth_term = torch.lgamma(gamma_batch.sum(dim=1)).sum() \
      - torch.lgamma(gamma_batch).sum() \
      + contract('mk,mk->', ss, gamma_batch - 1, backend='torch')
    
    fifth_term = contract(
      'mkv,mkv,mv->', 
      phi, phi.log(), W_batch, 
      backend='torch'
    )

    if version=='real':
      s_term = s_term_normal(y_batch, gamma_batch, self.eta, delta, M)
    else:
      s_term = s_term_bernoulli(y_batch, gamma_batch, self.eta)

    return first_term + second_term + third_term - \
      fourth_term - fifth_term + s_term

  
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
      requires_grad=True, 
      device=self.device
    )
    
    opt = torch.optim.Adam([theta_maps], lr=lr)
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
    out1 = contract('mv,mv->m', W, bl.log(), backend='torch') 
    out2 = torch.mv(theta.softmax(dim=1).log(), self.alpha - 1)
    return out1 + out2


