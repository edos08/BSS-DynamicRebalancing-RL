"""Train the DQN on ``bss_gym.GridZoneEnv`` (the new simulator) (issue #166).

    python -m rl_training.train_grid --bundle data/scenarios/cambridge-se-2024 --episodes 30 \
        --device cuda:0 --results-path results/ --run-id 2

Same agent, network and hyper-parameters as the legacy ``train.py`` (gamma 0.95 per 30 s step,
SGD lr 1e-4, batch 64, buffer 1e5, hard target update every 8 three-hour slots, epsilon decaying
per slot so that it reaches ~0.1 after ``exploration_time`` of the training), with the
**semi-MDP update** (issue #164). Each transition stores:

* the discounted reward of the action over its 30 s steps, ``R = sum_i gamma^i r_i``;
* its own duration ``tau``: the target discounts the next state by ``gamma^tau``;
* ``terminal`` only for true terminal states. Episodes end by time limit, so the last state still
  bootstraps.

Results use the legacy layout, so ``bss-results-webapp`` shows them next to the legacy runs:
``<results>/run_NNN/training/episode_NNN/`` (scalars, per-3 h timeslot metrics, step data, zone
graph with per-zone means), the greedy evaluation on held-out weeks as
``validation/episode_NNN/episode_000/``, and models in ``models/``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from shapely.geometry import box
from torch_geometric.data import Data

from rl_training.agents import DQNAgent
from rl_training.memory import ReplayBuffer
from rl_training.results import EpisodeResults, ResultsManager
from rl_training.utils import set_seed

PARAMS = {"gamma": 0.95, "lr": 1e-4, "batch_size": 64, "buffer": int(1e5), "epsilon_start": 1.0,
          "epsilon_end": 0.01, "exploration_time": 0.6, "target_every_slots": 8}


class GraphState:
    """What the replay buffer stores for one observation."""

    def __init__(self, obs: dict, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        self.x = torch.as_tensor(obs["graph"], dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.agent_state = obs["agent"]

    def single(self, device) -> Data:
        return Data(x=self.x.to(device), edge_index=self.edge_index.to(device), edge_attr=self.edge_attr.to(device),
                    agent_state=torch.as_tensor(self.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
                    batch=torch.zeros(self.x.size(0), dtype=torch.long, device=device))


def make_env(bundle: str, start_dates: list[str], days: float, sim_config: dict | None = None, cell_m: float = 300.0):
    from bss_gym import GridZoneEnv
    return GridZoneEnv(bundle, sim_config=sim_config, start_dates=start_dates, episode_days=days, cell_m=cell_m)


def zone_graph(env, stats: dict, cell_m: float) -> nx.MultiDiGraph:
    """Zone graph in the legacy ``cell_subgraph`` format (lon/lat, square boundary, per-zone metrics)."""
    sc = env.sim.scenario
    root = sc.root
    with np.load(root / "network" / "bike_nodes.npz") as z:
        lat, lon = z["lat"], z["lon"]
    # local projection is linear at city scale: x -> lon, y -> lat
    ax, bx = np.polyfit(sc.bike_x, lon, 1)
    ay, by = np.polyfit(sc.bike_y, lat, 1)
    x0, y0 = sc.bike_x.min(), sc.bike_y.min()
    g = nx.MultiDiGraph()
    g.graph["crs"] = "EPSG:4326"
    for k, zone in enumerate(env._grid):
        xa, ya = x0 + zone.ix * cell_m, y0 + zone.iy * cell_m
        poly = box(ax * xa + bx, ay * ya + by, ax * (xa + cell_m) + bx, ay * (ya + cell_m) + by)
        cx, cy = env.zones.center[k]
        g.add_node(k, cell_id=k, x=float(ax * cx + bx), y=float(ay * cy + by), boundary=poly,
                   **{name: float(v[k]) for name, v in stats.items()})
    for a, b in env.edge_index.T.tolist():
        g.add_edge(a, b)
    return g


def run_episode(env, agent: DQNAgent, seed: int, device, learn: bool, params: dict, on_slot=None,
                cell_m: float = 300.0, episode: int = 0, mode: str = "train") -> tuple[EpisodeResults, dict]:
    obs, info = env.reset(seed=seed)
    ei = torch.as_tensor(env.edge_index, dtype=torch.long)
    ea = torch.as_tensor(env.edge_attr, dtype=torch.float32)
    state = GraphState(obs, ei, ea)
    Z = env.n_zones
    sums = {"critic": np.zeros(Z), "eligibility": np.zeros(Z), "bikes": np.zeros(Z)}
    visits, ops = np.zeros(Z), np.zeros(Z)
    n_obs = 0
    series = {k: [] for k in ("reward", "failures", "epsilon", "demand", "deployed", "truck", "depot", "outside",
                              "riding", "q")}
    slot_reward, slot_fail, req0 = 0.0, 0, env.sim.kpis_.requests
    actions, rewards_by_action, critic_steps = [], {a: [] for a in range(8)}, []
    ret, invalid, losses, done = 0.0, 0, [], False
    t_wall = time.perf_counter()
    while not done:
        single = state.single(device)
        action = agent.select_action(single, epsilon_greedy=learn, greedy=not learn)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        R = DQNAgent.action_return(info["step_rewards"], params["gamma"])
        next_state = GraphState(obs, ei, ea)
        if learn:
            agent.replay_buffer.push(state, action, R, next_state, done, steps=info["steps"], terminal=terminated)
            loss = agent.train_step(params["batch_size"])
            if loss:
                losses.append(loss)
        # bookkeeping
        ret += reward
        slot_reward += reward
        slot_fail += sum(info["failures"])
        invalid += int(info["invalid"])
        actions.append(action)
        rewards_by_action[action].append(reward)
        critic_steps.append(info["global_critic"])
        z = info["truck_zone"]
        if action in (1, 2, 3, 4) and not info["invalid"]:
            visits[z] += 1
        if action in (5, 6, 7) and not info["invalid"]:
            ops[z] += 1
        sums["critic"] += env.critic
        sums["eligibility"] += env.eligibility
        sums["bikes"] += env.bikes
        n_obs += 1
        if info.get("slot_end") or done:
            f = env.sim.fleet
            series["reward"].append(slot_reward)
            series["failures"].append(slot_fail)
            series["epsilon"].append(agent.epsilon)
            series["demand"].append(env.sim.kpis_.requests - req0)
            series["deployed"].append(f.status_count[0])
            series["truck"].append(len(env.sim.trucks[0].load))
            series["depot"].append(f.status_count[4])
            series["outside"].append(f.status_count[5])
            series["riding"].append(f.status_count[2])
            with torch.no_grad():
                series["q"].append(agent.get_q_values(single)[0].squeeze().cpu().numpy())
            slot_reward, slot_fail, req0 = 0.0, 0, env.sim.kpis_.requests
            if info.get("slot_end") and on_slot is not None:
                on_slot()
        state = next_state
    k = env.kpis()
    days = env.episode_s / 86400.0
    failures_zone = env.sim.partition(env.zones.name).failures.sum(axis=1)
    stats = {"critic_mean": sums["critic"] / n_obs, "eligibility_mean": sums["eligibility"] / n_obs,
             "bikes_mean": sums["bikes"] / n_obs, "failure_sum": failures_zone, "visits_sum": visits, "ops_sum": ops}
    res = EpisodeResults(
        episode=episode, mode=mode, seed=seed, total_reward=ret,
        mean_daily_failures=k["failed"] / days, total_failures=int(k["failed"]), total_invalid_actions=invalid,
        mean_q_values=float(np.mean([q.mean() for q in series["q"]])), epsilon=agent.epsilon,
        rewards_per_timeslot=series["reward"], failures_per_timeslot=series["failures"],
        epsilon_per_timeslot=series["epsilon"], demand_per_timeslot=series["demand"],
        deployed_bikes=series["deployed"], truck_load=series["truck"], depot_load=series["depot"],
        outside_system_bikes=series["outside"], traveling_bikes=series["riding"],
        q_values_per_timeslot=series["q"], action_per_step=actions, reward_tracking_per_action=rewards_by_action,
        global_critic_scores=critic_steps, cell_subgraph=zone_graph(env, stats, cell_m))
    summary = {"return": ret, "decisions": len(actions), "invalid": invalid, "epsilon": agent.epsilon,
               "loss": float(np.mean(losses)) if losses else 0.0, "wall_s": time.perf_counter() - t_wall,
               "city_service_level": k["city_service_level"], "failed": k["failed"], "requests": k["requests"],
               "bikes_moved": k["bikes_moved"], "truck_km": k["truck_km"]}
    return res, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="data/scenarios/cambridge-se-2024")
    ap.add_argument("--results-path", default="results/")
    ap.add_argument("--run-id", type=int, required=True)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run with the same id")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--days", type=float, default=7.0, help="episode length in days")
    ap.add_argument("--fleet", type=int, default=340, help="bikes in the zone (Cambridge ratio: 800 per 1,824 docks)")
    ap.add_argument("--train-dates", nargs="+", default=["2024-09-09", "2024-09-16", "2024-09-23", "2024-10-14",
                                                           "2024-10-21"])
    ap.add_argument("--eval-dates", nargs="+", default=["2024-09-30", "2024-10-07"])
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--mode", default="dockless", choices=["dockless", "docked"])
    ap.add_argument("--cell-m", type=float, default=300.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=PARAMS["gamma"])
    ap.add_argument("--save-every", type=int, default=5, help="save the model every N episodes")
    args = ap.parse_args(argv)
    params = {**PARAMS, "gamma": args.gamma}
    set_seed(args.seed)
    sim_config = {"mode": args.mode, "fleet": {"size": args.fleet}}
    if args.mode == "docked":
        sim_config["fleet"]["initial_distribution"] = "uniform"
    env = make_env(args.bundle, args.train_dates, args.days, sim_config, args.cell_m)
    slots_per_episode = int(round(args.days * 8))
    T = params["exploration_time"] * args.episodes * slots_per_episode
    agent = DQNAgent(num_actions=8, observation_space_len=env.agent_len, replay_buffer=ReplayBuffer(params["buffer"]),
                     gamma=params["gamma"], epsilon_start=params["epsilon_start"], epsilon_end=params["epsilon_end"],
                     epsilon_decay=T * T / math.log(10.0), lr=params["lr"], device=args.device, seed=args.seed)
    rm = ResultsManager(args.results_path, args.run_id, overwrite=args.overwrite, interactive=False)
    rm.save_hyperparameters(
        params={**params, **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "env": "bss_gym.GridZoneEnv", "zones": env.n_zones, "agent_len": env.agent_len, "smdp": True},
        reward_params=vars(env.reward_cfg))
    slots = {"n": 0}

    def on_slot():
        slots["n"] += 1
        agent.update_epsilon()
        if slots["n"] % params["target_every_slots"] == 0:
            agent.update_target_network()

    for ep in range(args.episodes):
        res, s = run_episode(env, agent, seed=args.seed * 1000 + ep, device=args.device, learn=True, params=params,
                             on_slot=on_slot, cell_m=args.cell_m, episode=ep)
        rm.save_episode(res)
        rm.save_run_summary()
        print(f"ep {ep:3d} return {s['return']:9.1f} served {s['city_service_level']:.3f} moved {s['bikes_moved']:5d} "
              f"invalid {s['invalid']:5d}/{s['decisions']:5d} eps {s['epsilon']:.3f} loss {s['loss']:.4f} "
              f"wall {s['wall_s']:.0f}s", flush=True)
        if (ep + 1) % args.save_every == 0 or ep == args.episodes - 1:
            agent.save_model(rm.models_path / f"model_episode_{ep:03d}.pt")
    agent.save_model(rm.models_path / "model_final.pt")

    # greedy evaluation on held-out weeks, as the webapp's validation view
    eval_env = make_env(args.bundle, args.eval_dates, args.days, sim_config, args.cell_m)
    val = ResultsManager(args.results_path, args.run_id, overwrite=True, interactive=False, mode="validation",
                         val_tag=f"episode_{args.episodes - 1:03d}")
    i = 0
    for d, day in enumerate(args.eval_dates):
        eval_env.start_dates = (day,)
        for s_ in range(args.eval_seeds):
            res, s = run_episode(eval_env, agent, seed=10_000 + 100 * d + s_, device=args.device, learn=False,
                                 params=params, cell_m=args.cell_m, episode=i, mode="validation")
            val.save_episode(res)
            print(f"eval {day} seed {s_}: served {s['city_service_level']:.3f} moved {s['bikes_moved']} "
                  f"failures {s['failed']}", flush=True)
            i += 1
    val.save_run_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
