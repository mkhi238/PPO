import numpy as np
import gymnasium as gym
import torchrl
import torch
import torch.optim as opt

class PPO:
  '''
  def init: initalize/define the PPO object itself (make actor network, make critic network)
  def objective: define the clipped surrogate objective
  def loss: calculate the loss function value of PPO
  def rollout: collect a batch of data to update our PPO parameters
  def train: train our PPO model using our rollouts, objective, and loss
  '''
  def __init__(self, env, actor_network, critic_network, **hyperparameters):
    self.env = env
    self.action_space = env.action_space
    self.obs_space = env.observation_space
    
    self.actor_network = actor_network
    self.critic_network = critic_network
    self._init_hyperparameters(hyperparameters)
    
    self.actor_optimizer = opt.AdamW()
    
    
  def rollout(self):
    pass
    
  
  def objective(self):
    pass
  
  def loss(self):
    pass
  
  def train(self):
    #I want to do the training here, I want the training loop to pass into here, this file does not 
    #need an ifname=main, so i can call this method directly from the train in the main file, and then this 
    #execute my training.
    
    #because of this, I will need all the hypers into the init to pull down to here
    #implement minibatch SGD with AdamW optimizer
    pass
  
  def _init_hyperparameters(self, **hyperparameters):
    self.lam = hyperparameters['lambda']
    self.gamma = hyperparameters['gamma']
    self.eps = hyperparameters['eps']
    self.c1 = hyperparameters['c1'] #weight on critic
    self.c2 = hyperparameters['c2'] #KL weight
  
  
  
  
  
    
  