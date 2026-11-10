import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical, Normal, Independent, Bernoulli, Beta
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torchrl.objectives.value import GAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPO:
  '''
  def init: initalize/define the PPO object itself (make actor network, make critic network)
  def objective: define the clipped surrogate objective
  def loss: calculate the loss function value of PPO
  def rollout: collect a batch of data to update our PPO parameters
  def train: train our PPO model using our rollouts, objective, and loss
  
  PPO(env, FeedForwardNetwork, FeedForwardNetwork, **hypers)
  '''
  
  def __init__(self, env, actor_network, critic_network, **hyperparameters):
    '''
    Initalize the PPO 
    '''
    
    #get env
    self.env = env
    self.action_space = env.action_space #returns int dim
    self.obs_space = env.observation_space #returns np.array
    
    #hyperparams init
    self._init_hyperparameters(**hyperparameters)
    #we will need to check the probability distribution we use here 
    
    #obs_space.shape[0] is input dims
    self.actor_network = actor_network(self.obs_space.shape[0], self.hidden_size, self.output_size).to(device) #.n to go from discrete(2) to 2 (can just do actor_network(obs_))
    self.critic_network = critic_network(self.obs_space.shape[0], self.hidden_size, 1).to(device) #maybe double descent this? add a multiplier to overparamterize

    #AdamW is considered the best optimizer currently for PPO
    self.actor_optimizer = opt.AdamW(self.actor_network.parameters(), lr = self.lr_actor)
    self.critic_optimizer = opt.AdamW(self.critic_network.parameters(), lr = self.lr_critic)
    self.gae_func = self.compute_GAE(self.critic_network)
    
    
    
  def rollout(self):
    '''
    Here, we collect the actual real rollouts/samples
    on-policy, so we run this everytime we want to oupdate pi_theta_old after our epochs*batch_size minibatch updates
    need to actually run the actor model here, get actions, and then continue
    
    For early stopping purposes (say theta_pi deviates too far away from theta_pi_old) we will want to use a KL-divergence 
    
    Rollout is for collecting a SINGLE episode's worth of data. We reset the rollout every time we finish an episode, or the first time we interact with an environment
    
    value target is just the rtgs (conceptually), advantage is the main GAE output 
    in practice, torchrl's computation of the GAE outputs both a advantage and value function
    
    so all these policy based methods, they define set distributions and only use the actor to select for these dists (ANNs with fixed weights are not stochastic, 
    we only adjust the weigths sparsely in PPO, so it will always prodcue the same answer for our current on polcy batch)
    
    ROLLOUT IS THE FORWARD PASS OF THE MODEL ONLY. WE DO NOT COMPUTE THE LOSS HERE, ONLY THE VALUES NEEDED
    '''
    t = 0
    rewards = [] #episode rewards, used for the GAE function ()
    returns = []
    observations = []
    next_observations = []
    actions = []
    logprobs = [] #track logprobs (log(pi) - log(pi_old))
    dones = [] #if we are done (either got too long or got terminated)
    end_episode = [] #only if we got terminated
    
    while t < self.max_rollout_size:
      
      #reset at the end of an episode
      obs, info = self.env.reset() #obs is observation, first observation seen by the env. Info is a dict of info
      #obs = [T=horizon, obs_dim]; for cartpole, obs = [4] obs_tensor = [2048, 4]
      episode_len = 0
      episode_return = [] #this is summing the cumulative reward of the episode
      while episode_len < self.max_timesteps_per_episode and t < self.max_rollout_size:
        terminated = False #end of the episode naturally
        with torch.no_grad():
          observations.append(obs)
          actn, sampled_probs = self.get_action(obs, self.dist_name)
          obs, reward, terminated, truncated, info = self.env.step(self.to_env(actn)) #truncated is ending the episode due to timestep reasons
          rewards.append(reward)
          episode_return.append(reward)
          next_observations.append(obs)
          logprobs.append(sampled_probs) #[logprobs1, logprobs2, ..., logprobsn-1, logprobsn], and just pull out logprobsn-1 and logprobsn (n-1 is old, n is new)
          actions.append(actn)
          dones.append(terminated or truncated) #dones appends only if the episode got too long or ended 
          end_episode.append(terminated) #end_epidoe appends if the episode got too long
          t += 1
          episode_len += 1
          if truncated or terminated:
            break
          
      #append the unnormalized rewards to compare against CleanRL benchmarks
      #we are literally only tracking this as another metric to use to see our success other than loss
      #simply summing up the rewards of the episode (so the return of a set of actions, the total accumulated reward)
      #recall return = sigma(reward)_T (summed reward over all timesteps)
      if "episode" in info:
        returns.append(info["episode"]["r"])
      else: 
        for i in episode_return:
          returns.append(sum(episode_return))
          
      dones[-1] = True #since our episode is done
      
    if not (terminated or truncated) and episode_len < self.max_timesteps_per_episode:
      returns.pop()
      
    #convert to tensors (first to np arrays then to tensors for efficiency over straight python lists)
    actions_tensor = torch.stack(actions, dim = 0) #(2048,)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1) #(2048,) -> (2048, 1)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device) #(#of episodes,)
    obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32, device=device) #(2048, 4)
    next_obs_tensor = torch.tensor(np.array(next_observations), dtype=torch.float32, device=device) #(2048,4)
    logprobs_tensor = torch.stack(logprobs, dim=0) #(2048,)
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device).unsqueeze(-1) #(2048,) -> (2048,1)
    end_episode_tensor = torch.tensor(end_episode, dtype = torch.bool, device=device).unsqueeze(-1) #(2048,) -> (2048,1)
    
    with torch.no_grad():
      tnsrdct = TensorDict({"obs" : obs_tensor, 
                                  "next" : {"obs": next_obs_tensor,
                                            "reward": rewards_tensor,
                                            "done": dones_tensor,
                                            'terminated': end_episode_tensor
                                            },
                                  
                            }, batch_size=[self.max_rollout_size])
      
      self.gae_func(tnsrdct)
      advtg = tnsrdct['advantage'] #type = tensor, (2048,1)
      vf = tnsrdct['value_target'] #type = tensor, (2048,1)

    return advtg, vf, logprobs_tensor, rewards_tensor, obs_tensor, actions_tensor, dones_tensor, end_episode_tensor, returns_tensor #need to return these so that when we do loss.backwards() it works

  
  def get_action(self, observation_state, distribution, beta_clip = 0.01):

      logits = self.actor_network(observation_state) #returns a tensor, so probability dist should be able to accept it
      _ , prob_dist = self.make_dist(self.dist_name, logits)
      sampled_action = prob_dist.sample() #this returns an idx of the actual sampled action.
      if self.dist_name == "Beta":
        sampled_action = sampled_action.clip(beta_clip, 1-beta_clip)
      log_probs = prob_dist.log_prob(sampled_action)
      return sampled_action, log_probs
    
  def compute_GAE(self, value_network):
    value_dict = TensorDictModule(module = value_network, in_keys=['obs'], out_keys=['state_value'])
    gae = GAE(
      gamma = self.gamma,
      lmbda = self.lam,
      value_network = value_dict,
      device=device
    )
    return gae
  
  def objective(self, old_policy, curr_policy, advantage, curr_dist, value_network, vf):
    '''
    #in the actual PPO code, to avoid underflow the PPO is log(pi_theta(at|st)) - log(pi_theta_old(at|st))
    
    #torch can automatically compute the computation graph for the PPO clip loss function for the actor
    #no need to define the gradient, loss.backward() makes the computation graph of L_Clip (Eqn 9) and will handle that for me
    
    Ok so here we combine the 3 losses. 
    
    E[L_CLIP - c_1*L_VF + c2*(Entropy of dist bonus)]
    '''
    L_clip = self.actor_loss(old_policy, curr_policy, advantage)
    L_vf = self.critic_loss(value_network, vf)
    L_h = torch.mean(self.c2 * curr_dist.entropy())
    return -L_clip + L_vf - L_h
  
  def actor_loss(self, old, curr, advantage):
    '''
    Q - V = A; this is ONLY defined for the actor
    
    E[L_CLIP]
    Take L_CLIP - c_1*L_VF + c2*(Entropy of dist bonus) in expectation, which is done emperically (1/N)*sum(L_CLIP - c_1*L_VF + c2*(Entropy of dist bonus) for all minibatches
    Law of Large Numbers miu -> E[L_CLIP - c_1*L_VF + c2*(Entropy of dist bonus]
    
    c_1*L_VF does use V, but it's SOLE purpose is to be a variance reduction mechanism (for free) against the actor.
    '''
    r_t = curr.unsqueeze(-1) - old.unsqueeze(-1) #makes curr (minibatch,1) and same for old
    r_t = torch.exp(r_t)
    clipped = torch.clip(r_t, 1-self.eps, 1+self.eps)
    L_clip = torch.mean(torch.min(r_t * advantage, clipped * advantage)) #broadcasting works, adv will be sliced into minibatches from (2048,1) to make it work
    return L_clip

  
  def critic_loss(self, value_network, vf):
    '''
    c_1*L_VF
    '''
    L_vf = torch.mean(self.c1 *torch.square(value_network - vf)) #needs to be (minibatch,1)
    return L_vf

  
  def train_model(self):
    #I want to do the training here, I want the training loop to pass into here, this file does not 
    #need an ifname=main, so i can call this method directly from the train in the main file, and then this 
    #execute my training.
    
    #because of this, I will need all the hypers into the init to pull down to here
    #implement minibatch SGD with AdamW optimizer
    total_timesteps = 0
    losses, mean_episode_return, approx_KL = [], [], [] #approx KL is the mean KL between old and new policies
    while total_timesteps < self.num_iterations * self.max_rollout_size:

      advtg, vf, logprobs_tensor, _, obs_tensor, actions_tensor, _, _, returns_tensor = self.rollout()
      mean_episode_return.append(torch.mean(returns_tensor).item())
      print(f"Iter return: {mean_episode_return[-1]:.1f}")
      data = TensorDataset(logprobs_tensor, advtg, vf, actions_tensor, obs_tensor) #make the data needed into a tensor dataset
      train_dataloader = DataLoader(data, batch_size=self.minibatch_size, shuffle = True) #make it into a dataloader
      #logprobs_tensor is old (2048,)
      for idx in range(self.epochs):
        summed_epoch_loss = 0
        print(f"Epoch: {idx}")
        for minibatch in train_dataloader:
          #unpack the train_dataloader (logprobs_tensor, advtg, vf, actions_tensor, obs_tensor)
          old_lps, advs, vfs, acts, observes = minibatch #unpack
          self.actor_optimizer.zero_grad()
          self.critic_optimizer.zero_grad()
          output = self.actor_network(observes) #forward pass
          _, curr_dist = self.make_dist(self.dist_name, output) #this is curr dist (logprob of curr)
          curr_lps = curr_dist.log_prob(acts)
          loss = self.objective(old_lps, curr_lps, advs, curr_dist, self.critic_network(observes), vfs)
          with torch.no_grad():
            losses.append(loss.item())
            summed_epoch_loss += loss.item()
            #Schulman approximate KL divergence: http://joschu.net/blog/kl-approx.html
            #forward KL divergence: KL = E[log(old) - log(curr)] since we are doing KL-divergence of the logprobs
            #let p(x) be the curr, let q(x) be the old. log_ratio = log(p(x)/q(x)) = curr_lps - old_lps 
            #we do this since we only stored the old policy’s log prob for the taken action so the sample estimate is all we can get 
            #(unless we store the full dist, but thats heavy for some tasks)
            log_ratio = curr_lps - old_lps
            ratio = torch.exp(log_ratio)
            approx_forward_kl_div = torch.mean((ratio-1) - log_ratio) #recalling torch.log(ratio) is back to log_ratio
            approx_KL.append(approx_forward_kl_div.item())
          
          loss.backward()
          self.actor_optimizer.step()
          self.critic_optimizer.step()
        #stop early if our KL divergence between the old and new policy is too larget
        if self.target_kl is not None and approx_forward_kl_div > self.target_kl:
          break
        print(f"Avg Loss: {round(summed_epoch_loss/len(train_dataloader),4)}")
      total_timesteps += self.max_rollout_size
      
    #return as arrays so i can save and graph them
    return np.array(losses), np.array(mean_episode_return), np.array(approx_KL)
  
  def _init_hyperparameters(self, **hyperparameters):
    #env hypers
    #2048 or 4096
    self.max_rollout_size = hyperparameters['horizon'] #horizon = rollout_size = how many steps to collect in one env run
    self.max_timesteps_per_episode = hyperparameters['episode_cap'] #a cap on the number of steps we can take in one given episode
    self.num_iterations = hyperparameters['num_iterations']
    #it limits how long a SINGLE episode can go
    self.dist_name = hyperparameters['distribution']
    self.output_size, _ = self.make_dist(self.dist_name)
         
    #GAE & PPO hypers
    self.lam = hyperparameters['lambda'] #at lam=0, it is TD(0), at lam =1, it is pure MC
    #this controls how far into the future we see/care about
    #this is the EMA discount parameter (how much we value the current estimate wrt all the other estimates weve seen)
    #it can also be considered the bias/variance tradeoff parameter (lam = 1 = high variance, no bias)
    
    self.gamma = hyperparameters['gamma'] #standard geometric series discounting factor (how much in the future we care about)
    self.eps = hyperparameters['eps'] #clip
    self.c1 = hyperparameters['c1'] #weight on critic
    self.c2 = hyperparameters['c2'] #entropy weight
    self.epochs = hyperparameters['epochs'] #how many times the algorithm loops over the entire 4,800-step rollout dataset during a single training phase
    self.minibatch_size = int(self.max_rollout_size/16) #if 4096, then 256 minibatch size, 128 if 2048
    self.target_kl = hyperparameters['target_kl']
    
    #actor critic NN hypers
    #may get some use in scheduling these (should actually schedule these, decay to 0)
    self.lr_actor = hyperparameters['learning_rate_actor']
    self.lr_critic = hyperparameters['learning_rate_critic']
    self.hidden_size = hyperparameters['hidden_size']
    
    
  def make_dist(self, dist_name, output_head = None): 
    dists = {
      #lambda prevents eager evaluation
      "Bernoulli": lambda: 1,
      "Categorical": lambda: self.action_space.n,
      "Normal": lambda: 2 * self.action_space.shape[0],
      "Beta": lambda: 2 * self.action_space.shape[0],
    }
    size_lam = dists[dist_name]

    if output_head is None:
      return size_lam(), None
    
    match dist_name:
      case "Categorical": return size_lam(), Categorical(logits = output_head)
      case "Bernoulli": return size_lam(), Bernoulli(logits = output_head.squeeze(-1))
      case "Normal": 
        mu, log_std = output_head.chunk(2, dim=-1)
        return size_lam(), Independent(Normal(mu, log_std.clamp(-5, 2).exp()), 1)
      case "Beta":
          a, b = (F.softplus(output_head) + 1).chunk(2, dim=-1) #a = alpha, b = beta (pseudocounts)
          return size_lam(), Independent(Beta(a, b), 1)
  
  def to_env(self, actn):
    match self.dist_name:
      case "Categorical" : return int(actn.item())
      case "Bernoulli": return int(actn.item())
      case "Normal": return np.clip(actn.cpu().numpy(), self.action_space.low, self.action_space.high) #.cpu() needed since numpy only works on CPU tensors
      case "Beta":   return self.action_space.low + actn.cpu().numpy() * (self.action_space.high - self.action_space.low)


    
  
'''

  A note on policy distribtions, action spaces, and probability distributions:
  
  TLDR: A bunch of notes on Probability theory below talking to how the dists for the policies are chosen
  
  For a PPO benchmark, say Cartpole, we need to understand the type of task it is in order to choose an appropriate probability distribution.
  Cartpole is discrete, 2 outcomes, so we want to use a Bernouli distribtution to map our choices (Categorical works here too).
  We would NOT want to use a gaussian/multivariate gaussian, since these are continuous. 
  If I wanted to sample 1 cts number, I would do gaussian, and If I wanted to sample multiple cts numbers, mutlivariate gaussian
  Therefore, we must make the distribution choice itself a hyperparameter somehow, mapping:
  the action space (multiple continuous actions to select) -> Multivariate Gaussian probability distributions -> probability_distributions['multivariate'] = MultivariateNormal()
  
  Discrete Tasks PMF supports: Bernouli, Categorical
  Continuous Tasks PDF supports: Gaussian (Normal), MultivariateNormal, Beta
  
  Recall as well that we have an Entropy bonus within our objective function anyways, so we dont even need heavy-tailed distributions to promote exploration. 
  Our Entropy bonus for our probability distribution within the loss function appropriatley promotes heavy-tailed distribution implicity.
  This is also true given that the Gaussian dist is the maximum entropy distribution for infinite unconstrained (hence cts) spaces (like in some MuJoCo tasks)
  Thus, by maximizing the entropy bonus, PPO naturally forces the Gaussian dist to widen its variance and flatten out, 
  providing the aggressive tail exploration we need without changing the underlying distribution. Since we are doing multiobjective optimization and are weighting 
  c2 to be our entropy bonus, we promote exploration to however much we want c2 to enable this exploration w/o needed a different distribution
  
  Unfortunatley, with constrained continuous spaces (in some other MuJoCo tasks), we cannot simply use a Uniform distribution even though it is the maximum entropy distribution
  This is becuase it is a flat prob density function, so flat probability density function yields gradients of 0 with respect to the action choice (recall uniform is flat so gradident = 0), 
  This makes it mathematically impossible for for backpropagation to occur: recall the computational graph in torch is taking the derivative over the ENTIRE loss function, which includes the entropy term
  Instead, we compromise with the Gaussian, which enables us to parameterize the mean and variance continuously, giving us a differentiable probability dist function f(x)
  while still allowing the entropy bonus to flatten it out into a uniform-like shape during early exploration.
  
  We also dont need to schedule c1, c2 to decay since the loss function of the importance weighted L_t_Clip term will eventually dominate, promoting explotative, better actions
  over the explore promoting c2 entropy bonus

  Finally, an obvious but important point is that the distribution we use is dependant on the task itself. 
  We cannot default to one particular distribution, nor can we train the parameters of one distribution to have it work for all.
  Hence, we need to do the mapping to the appropriate dist beforehand.
  This is in line with No Free Lunch Theorm fundamentals and Probability theory fundamentals
'''
    
  