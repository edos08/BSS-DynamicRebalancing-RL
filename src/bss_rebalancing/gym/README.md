# bss-gym

Gymnasium environments on the BSS simulator v2 (`docs/sim-v2/SPEC.md` §14).

`GridZoneEnv` (`bss_gym/GridZone-v0`) reproduces the legacy grid environment on `bss_sim`:
- 300 m zones and the legacy 8 actions (STAY, UP, DOWN, LEFT, RIGHT, DROP, PICK_UP, CHARGE) mapped to
  truck commands;
- a graph observation per zone (truck, critic, eligibility, bikes) plus the legacy agent vector;
- the legacy shaped reward plus a failure cost per 30 s step;
- **semi-MDP stepping**: an action lasts as long as it really takes, in 30 s steps. `info["steps"]`
  is its duration τ and `info["step_rewards"]` the reward of each step. Discount as
  `R = Σ γ^i r_i` and the next state by `γ^τ` (`rl_training`'s `DQNAgent.action_return`,
  `smdp_target`).

```python
import gymnasium as gym
import bss_gym  # registers the env

env = gym.make("bss_gym/GridZone-v0", bundle="data/scenarios/cambridge-2024", episode_days=7)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
info["steps"], info["step_rewards"]
```
