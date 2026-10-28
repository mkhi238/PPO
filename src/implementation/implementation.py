import numpy as np
import gymnasium as gym
import torchrl
import torch
import torch.optim as opt
from network import FeedForwardNetwork
from torch.distributions import Categorical, Normal, MultivariateNormal, Bernoulli, Beta, Multinomial

class PPO:
  '''
  def init: initalize/define the PPO object itself (make actor network, make critic network)
  def objective: define the clipped surrogate objective
  def loss: calculate the loss function value of PPO
  def rollout: collect a batch of data to update our PPO parameters
  def train: train our PPO model using our rollouts, objective, and loss
  
  PPO(env, FeedForwardNetwork, FeedForwardNetwork, **hypers)
  
  A note on policy distribtions, action spaces, and probability distributions:
  
  For a PPO benchmark, say Cartpole, we need to understand the type of task it is. Cartpole is discrete, 2 outcomes, so we want to use a Bernouli distribtution to map our choices 
  We would NOT use a gaussian/multivariate gaussian, since if I wanted to sample 1 cts number, I would do gaussian, and If I wanted to sample multiple cts numbers, mutlivariate gaussian
  Therefore, we must make the distribution choice itself a hyperparameter somehow, mapping:
  the action space (multiple continuous actions to select) -> Multivariate Gaussian probability distributions -> probability_distributions['multivariate'] = MultivariateNormal()
  
  In other words, the distribution we use is dependant on the task itself. We cannot default to one particular distribution, nor can we train the parameters of one distribution to have it work for all
    -No Free Lunch Theorm fundamentals
  '''
  def __init__(self, env, actor_network, critic_network, **hyperparameters):
    #get env
    self.env = env
    self.action_space = env.action_space #returns int dim
    self.obs_space = env.observation_space #returns np.array
    
    #hyperparams init
    self._init_hyperparameters(hyperparameters)
    
    #we will need to check the probability distribution we use here 
    
    #obs_space.shape[0] is input dims
    self.actor_network = actor_network(self.obs_space.shape[0], self.hidden_size, self.action_space.n) #.n to go from discrete(2) to 2
    self.critic_network = critic_network(self.obs_space.shape[0], 1)

    #AdamW is considered the best optimizer currently for PPO
    self.actor_optimizer = opt.AdamW(self.actor.parameters(), learning_rate = self.lr_actor)
    self.critic_optimizer = opt.AdamW(self.actor.parameters(), learning_rate = self.lr_critic)
    
    
  def rollout(self):
    '''
    Here, we collect the actual real rollouts/samples
    on-policy, so we run this everytime we want to oupdate pi_theta_old after our epochs*batch_size minibatch updates
    need to actually run the actor model here, get actions, and then continue
    '''
    t = 0
    while t < self.max_rollout_size:
      
    
    
      
      t += 1
    pass
    
  
  def objective(self):
    
    #in the actual PPO code, to avoid underflow the PPO is log(pi_theta(at|st)) - log(pi_theta_old(at|st))
    
    #torch can automatically compute the computation graph for the PPO clip loss function for the actor
    #no need to define the gradient, loss.backward() makes the computation graph of L_Clip (Eqn 9) and will handle that for me
    
    pass
  
  def loss(self):

    pass
  
  def train_model(self):
    #I want to do the training here, I want the training loop to pass into here, this file does not 
    #need an ifname=main, so i can call this method directly from the train in the main file, and then this 
    #execute my training.
    
    #because of this, I will need all the hypers into the init to pull down to here
    #implement minibatch SGD with AdamW optimizer
    pass
  
  def _init_hyperparameters(self, **hyperparameters):
    #env hypers
    #2048 or 4096
    self.max_rollout_size = hyperparameters['horizon'] #horizon = rollout_size = how many steps to collect in one env run
    self.max_timesteps_per_episode = hyperparameters['episode_cap'] #a cap on the number of steps we can take in one given episode
    #it limits how long a SINGLE episode can go
     
    #GAE & PPO hypers
    self.lam = hyperparameters['lambda'] #at lam=0, it is TD(0), at lam =1, it is pure MC
    #this controls how far into the future we see/care about
    #this is the EMA discount parameter (how much we value the current estimate wrt all the other estimates weve seen)
    #it can also be considered the bias/variance tradeoff parameter (lam = 1 = high variance, no bias)
    
    self.gamma = hyperparameters['gamma'] #standard geometric series discounting factor (how much in the future we care about)

    self.eps = hyperparameters['eps'] #clip
    self.c1 = hyperparameters['c1'] #weight on critic
    self.c2 = hyperparameters['c2'] #KL weight
    self.epochs = hyperparameters['epochs'] #how many times the algorithm loops over the entire 4,800-step rollout dataset during a single training phase
    self.minibatch_size = self.epochs/16 #if 4096, then 256 minibatch size, 128 if 2048
    
    #actor critic NN hypers
    self.lr_actor = hyperparameters['learning_rate_actor']
    self.lr_critic = hyperparameters['learning_rate_critic']
    self.hidden_size = hyperparameters['hidden_size']
    
  
  
  
  
  
    
  