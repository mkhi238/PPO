from implementation import PPO
from network import FeedForwardNetwork
import gymnasium as gym
import numpy as np
import torch

def init_agent(env, actor_network, critic_network,**hyperparameters):
  '''
  Initialzie the agent
  '''
  #create agent
  return PPO(env, actor_network=actor_network, critic_network=critic_network, **hyperparameters)


if __name__ == '__main__':
  #define hyperparameters
  params = {
    'horizon' : 2048,
    'episode_cap' : 512,
    'num_iterations': 125,
    'distribution': "Categorical",
    'lambda' : 0.95,
    'gamma' : 0.99,
    'eps' : 0.2,
    'c1' : 0.525,
    'c2' : 0.0125,
    'epochs' : 10,
    'target_kl': 0.015,
    'learning_rate_actor' : 2.5e-4,
    'learning_rate_critic': 2.5e-4,
    'hidden_size': 64
  }
  env = gym.make("CartPole-v1")
  PPO_Cartpole = init_agent(env, FeedForwardNetwork, FeedForwardNetwork, **params)
  losses, mean_episode_rets, approx_KL = PPO_Cartpole.train_model()
  print("training completed")
  np.savez("cartpole_run.npz", losses=losses, returns=mean_episode_rets, kl=approx_KL)
  torch.save(PPO_Cartpole.actor_network.state_dict(), "cartpole_actor.pt")
