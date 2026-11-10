import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

d = np.load(Path.home() / "breakout_run.npz")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(d["returns"]); ax[0].set(title="Mean episode return", xlabel="Iteration")
ax[1].plot(d["losses"]); ax[1].set(title="Loss", xlabel="Minibatch update")
ax[2].plot(d["kl"]); ax[2].set(title="Approx KL", xlabel="Minibatch update")
plt.tight_layout()
plt.savefig(Path.home() / "PPO" / "graphs" / "breakout_curves.png", dpi=150)