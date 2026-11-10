import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import gymnasium as gym
import torch
from implementation.network import FeedForwardNetwork
from gymnasium.wrappers import RecordVideo

#for MinAtar
from minatar.gym import register_envs
register_envs()
class UpscaleRender(gym.Wrapper):
    def render(self):
        f = self.env.render()                                   # (10, 10, 3) floats in [0, 1]
        f = (np.clip(f, 0, 1) * 255).astype(np.uint8)           # to 8-bit colour
        return np.kron(f, np.ones((40, 40, 1), dtype=np.uint8)) # each cell -> 40x40 block, so 400x400

#env = gym.make("CartPole-v1", render_mode = "human")
#env = gym.make("Pendulum-v1", render_mode = "rgb_array") #to record the video
 
name_prefix = "breakout"
#for box2d
#env = gym.make("LunarLander-v3", render_mode = "rgb_array") #to record the video

#for MinAtar
env = UpscaleRender(gym.make("MinAtar/Breakout-v1", render_mode="rgb_array"))
env = gym.wrappers.FlattenObservation(env)

#for MuJoCo
#env = gym.make("HalfCheetah-v5", render_mode="rgb_array", camera_name="track")

env = RecordVideo(env, video_folder=str(Path.home() / "PPO" / "graphs"), name_prefix=name_prefix, episode_trigger=lambda ep: True)
if name_prefix == "hopper" or name_prefix == "cheetah":
   env = gym.wrappers.NormalizeObservation(env)
   rms = np.load(f"{name_prefix}_obs_rms.npz")
   env.get_wrapper_attr("obs_rms").mean = rms["mean"]
   env.get_wrapper_attr("obs_rms").var = rms["var"]
   env.update_running_mean = False   # freeze the stats during eval
   env = gym.wrappers.TransformObservation(env, lambda o: np.clip(o, -10, 10), env.observation_space)
actor = FeedForwardNetwork(400, 256, 3) # obs dim, hidden_size, n actions (must match training) (Cartpole is Bernouli (L or R), Pendulum input is theta_dot (anglular velocity), sin(theta), cos(theta) (pendulum angle))
#output dim for cartpole is 2, input dim for cartpole is 4. pendulum output is mean, std of normal dist
#hopper is 11, 64, 6
#space invaders is 600, 256, 4
#lunar lander is 8, 64, 4
#cheetah is 17, 64, 12
#breakout is 400, 256, 3
actor.load_state_dict(torch.load("breakout_actor.pt",  weights_only = True))
actor.eval() #network is in evaluation mode

for i in range(5):
  obs, info = env.reset()
  done = False
  ret = 0
  while not done:
    with torch.no_grad():
      #action = int(actor(obs).argmax()) #for MinAtar, Pendulum, Cartpole
      action = int(torch.distributions.Categorical(logits=actor(obs)).sample()) #sampling from categorical for breakout instead of greedy argmax
      #action = actor(obs)[:6].numpy() #[:3] for Hopper, [:6] for Cheetah
    obs, reward, terminated, truncated, _ = env.step(action)
    ret += reward
    if terminated or truncated:
      done = True
  print(f"Episode {i}: {ret}")

env.close()