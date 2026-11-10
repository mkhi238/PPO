from implementation import PPO
from network import FeedForwardNetwork
import gymnasium as gym
import numpy as np
import torch

#this is for our MinAtar tasks
from minatar.gym import register_envs
register_envs()

def init_agent(env, actor_network, critic_network,**hyperparameters):
  '''
  Initialzie the agent
  '''
  #create agent
  return PPO(env, actor_network=actor_network, critic_network=critic_network, **hyperparameters)

#this function is for our MuJoCo tasks
def make_env(env_id, gamma):
  env = gym.make(env_id)
  env = gym.wrappers.RecordEpisodeStatistics(env)   # true returns, before reward normalization
  env = gym.wrappers.ClipAction(env)
  env = gym.wrappers.NormalizeObservation(env)
  env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)
  env = gym.wrappers.NormalizeReward(env, gamma=gamma)
  env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
  return env

if __name__ == '__main__':
  #define hyperparameters
  params = {
    'horizon' : 2048,
    'episode_cap' : 1000,
    'num_iterations': 5000,
    'distribution': "Categorical",
    'lambda' : 0.95,
    'gamma' : 0.99, #for LunarLander gamma = 0.999 since its a long run reward task (value future outcomes more)
    'eps' : 0.2,
    'c1' : 0.525,
    'c2' : 0.01, #no entropy bonus for continuous control PPO (set to 0.0)
    'epochs' : 10,
    'target_kl': 0.02, #no target_kl for cts actns (set to None)
    'learning_rate_actor' : 3e-4,
    'learning_rate_critic': 3e-4,
    'hidden_size': 256 #256 for MinAtar (Space Invaders, Breakout)
  }
  
  #For Cartpole/Pendulum
  #env = gym.make("Pendulum-v1")
  
  #For MuJoCo tasks
  #env = make_env("HalfCheetah-v5", params['gamma'])
  
  #for MinAtar/Box2D tasks
  #env = gym.make("LunarLander-v3")
  env = gym.make("MinAtar/Breakout-v1")
  env = gym.wrappers.RecordEpisodeStatistics(env)
  
  #For MinAtar
  env = gym.wrappers.FlattenObservation(env)   # (10, 10, 6)  = 10*10*6 = 600 (flatten to 600), breakout flattens similarly to 400
  
  PPO_breakout = init_agent(env, FeedForwardNetwork, FeedForwardNetwork, **params)
  losses, mean_episode_rets, approx_KL = PPO_breakout.train_model()
  np.savez("breakout_run_5000_iters.npz", losses=losses, returns=mean_episode_rets, kl=approx_KL)
  
  #Mujoco needs to save mean and variance (theta_miu, theta_sigma)
  # theta = env.get_wrapper_attr("obs_rms")
  # np.savez("breakout_obs_rms.npz", mean=theta.mean, var=theta.var)
  
  print("training completed")
  torch.save(PPO_breakout.actor_network.state_dict(), "breakout_actor_5000_iters.pt")
