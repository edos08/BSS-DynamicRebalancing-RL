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

**Checkpoints and best model** (issue #170).
* Every ``--checkpoint-every`` episodes the whole training state is saved to ``checkpoints/``:
  networks, optimizer, epsilon, RNG states and replay buffer. ``--resume`` continues an interrupted
  run exactly where it stopped.
* Every ``--val-every`` episodes the greedy policy is evaluated on ``--val-date`` (a held-out week).
  The best one by mean daily failures is kept in ``models/best/``: DQN performance
  oscillates during training, so the final weights are often not the best.

**Weights & Biases** (issue #171, optional): ``--wandb`` logs the configuration, every episode and
every validation, and uploads **only the best model** of the run (one artifact, alias ``best``) at
the end of training (issue #174).

Results use the legacy layout, so ``bss-results-webapp`` shows them next to the legacy runs:
``<results>/run_NNN/training/episode_NNN/`` (scalars, per-3 h timeslot metrics, step data, zone
graph with per-zone means), the greedy evaluation on held-out weeks as
``validation/episode_NNN/episode_000/`` (with its ``config.json``), models as in ``train.py``
(``models/episodes/episode_NNN/``, ``models/best/`` with ``metadata.json``, ``models/final/``) and the log
in ``training/logs/train.log``. ``config.json`` uses the legacy hyper-parameter names (``num_episodes``,
``maximum_number_of_bikes``, ...), and the legacy flags ``--num-episodes``, ``--max-num-bikes``,
``--min-num-bikes`` and ``--critic-mode`` are accepted. The best model is, as in ``train.py``, the one with
the lowest validation mean daily failures.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from shapely.geometry import box
from torch_geometric.data import Data
from tqdm import tqdm

import pandas as pd

from rl_training.agents import DQNAgent
from rl_training.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from rl_training.logging_config import LoggingConfig, get_logger, init_logging
from rl_training.memory import ReplayBuffer
from rl_training.results import EpisodeResults, ResultsManager, prepare_run_dir
from rl_training.tracking import Tracker
from rl_training.utils import set_seed

PARAMS = {"gamma": 0.95, "lr": 1e-4, "batch_size": 64, "buffer": int(1e5), "epsilon_start": 1.0,
          "epsilon_end": 0.01, "exploration_time": 0.6, "target_every_slots": 8,
          # as in train.py: stored and passed to the agent, but both trainers copy the target network
          # every 8 slots (the agent's soft update is disabled)
          "soft_update": True, "tau": 0.005}


def legacy_hyperparameters(args, params: dict, env, epsilon_decay: float, slots_per_episode: int) -> dict:
    """``config.json`` hyper-parameters with the keys of the legacy ``train.py``, so that runs of the two
    trainers can be compared field by field (the results webapp reads these keys), then the new ones."""
    try:
        depot_zone = int(env.zone_of_node[env.sim.depot_stop])
    except Exception:   # noqa: BLE001 - informative only
        depot_zone = None
    legacy = {
        "seed": args.seed,
        "num_episodes": args.episodes,
        "batch_size": params["batch_size"],
        "replay_buffer_capacity": params["buffer"],
        "gamma": params["gamma"],
        "epsilon_start": params["epsilon_start"],
        "epsilon_end": params["epsilon_end"],
        "epsilon_decay": epsilon_decay,
        "exploration_time": params["exploration_time"],
        "lr": params["lr"],
        "soft_update": params["soft_update"],
        "tau": params["tau"],
        "total_timeslots": slots_per_episode,
        "maximum_number_of_bikes": args.fleet,
        "minimum_number_of_bikes": args.min_num_bikes,
        "enable_repositioning": False,
        "use_net_flow": False,
        "critic_mode": args.critic_mode,
        "depot_position_id": depot_zone,
        "initial_cell_id": depot_zone,
        "validation_every": args.val_every,
    }
    extra = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    extra.update({"target_every_slots": params["target_every_slots"], "env": "bss_gym.GridZoneEnv",
                  "zones": env.n_zones, "agent_len": env.agent_len, "smdp": True})
    return {**legacy, **{k: v for k, v in extra.items() if k not in legacy}}


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
    ap.add_argument("--overwrite", action="store_true", help="replace an existing run with the same id without asking")
    ap.add_argument("--episodes", "--num-episodes", dest="episodes", type=int, default=30)
    ap.add_argument("--days", type=float, default=7.0, help="episode length in days")
    ap.add_argument("--fleet", "--max-num-bikes", dest="fleet", type=int, default=340,
                    help="bikes in the zone (Cambridge ratio: 800 per 1,824 docks); --max-num-bikes as in bss-train")
    ap.add_argument("--min-num-bikes", type=int, default=0, choices=[0],
                    help="as in bss-train; the new simulator has no per-zone minimum yet, so only 0")
    ap.add_argument("--critic-mode", default="poisson", choices=["poisson"],
                    help="as in bss-train; the new env only forecasts (no oracle of future trips)")
    ap.add_argument("--train-dates", nargs="+", default=["2024-09-09", "2024-09-16", "2024-09-23", "2024-10-14",
                                                           "2024-10-21"])
    ap.add_argument("--eval-dates", nargs="+", default=["2024-09-30", "2024-10-07"])
    ap.add_argument("--eval-seeds", type=int, default=2)
    ap.add_argument("--mode", default="dockless", choices=["dockless", "docked"])
    ap.add_argument("--cell-m", type=float, default=300.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=PARAMS["gamma"])
    ap.add_argument("--exploration-time", type=float, default=PARAMS["exploration_time"],
                    help="fraction of the training spent exploring (epsilon ~0.1 at its end), as in bss-train")
    ap.add_argument("--lr", type=float, default=PARAMS["lr"])
    ap.add_argument("--save-every", type=int, default=5, help="save the model every N episodes")
    ap.add_argument("--checkpoint-every", type=int, default=1, help="save a full training checkpoint every N episodes")
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--no-buffer-checkpoint", action="store_true",
                    help="do not save the replay buffer in checkpoints (smaller; a resume starts with an empty buffer)")
    ap.add_argument("--resume", action="store_true", help="continue the run from its last checkpoint")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="stop after N episodes in this invocation (resume later with --resume)")
    ap.add_argument("--val-every", type=int, default=5, help="validate the greedy policy every N episodes (0: never)")
    ap.add_argument("--val-date", default=None, help="week used for validation (default: the first --eval-dates)")
    ap.add_argument("--wandb", action="store_true", help="track the run with Weights & Biases")
    ap.add_argument("--wandb-project", default="bss-rebalancing")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline"])
    args = ap.parse_args(argv)
    params = {**PARAMS, "gamma": args.gamma, "exploration_time": args.exploration_time, "lr": args.lr}
    set_seed(args.seed)
    sim_config = {"mode": args.mode, "fleet": {"size": args.fleet}}
    if args.mode == "docked":
        sim_config["fleet"]["initial_distribution"] = "uniform"
    env = make_env(args.bundle, args.train_dates, args.days, sim_config, args.cell_m)
    slots_per_episode = int(round(args.days * 8))
    T = params["exploration_time"] * args.episodes * slots_per_episode
    epsilon_decay = T * T / math.log(10.0)
    agent = DQNAgent(num_actions=8, observation_space_len=env.agent_len, replay_buffer=ReplayBuffer(params["buffer"]),
                     gamma=params["gamma"], epsilon_start=params["epsilon_start"], epsilon_end=params["epsilon_end"],
                     epsilon_decay=epsilon_decay, lr=params["lr"], device=args.device, seed=args.seed,
                     tau=params["tau"], soft_update=params["soft_update"])
    prepare_run_dir(args.results_path, args.run_id, resume=args.resume, overwrite=args.overwrite)   # asks (#178)
    rm = ResultsManager(args.results_path, args.run_id, overwrite=True, interactive=False)
    hyper = legacy_hyperparameters(args, params, env, epsilon_decay, slots_per_episode)
    reward_params = vars(env.reward_cfg)
    # same log files as train.py: training/logs/train.log
    init_logging(LoggingConfig(level=logging.INFO, log_dir=os.path.join(rm.training_path, "logs"),
                               run_id=args.run_id, console=False, logger_name="train"))
    logger = get_logger("train", logger_name="train")
    logger.info("Starting training loop")
    ckpt_dir = rm.run_dir / "checkpoints"
    slots = {"n": 0}
    # best model: lowest validation mean daily failures, as in train.py
    start_episode, best, best_ep = 0, np.inf, None
    if args.resume:
        path = latest_checkpoint(ckpt_dir)
        if path is None:
            raise SystemExit(f"--resume: no checkpoint in {ckpt_dir}")
        state = load_checkpoint(path, agent)
        start_episode = state["episode"] + 1
        slots["n"] = state["extra"].get("slots", 0)
        if "best_val_failures" in state["extra"]:
            best = state["extra"]["best_val_failures"]
            best_ep = state["extra"].get("best_episode")
        else:   # checkpoint from before the legacy-compatible best model: start the best tracker again
            best, best_ep = np.inf, None
        rm.best_models = [] if best_ep is None else [(best_ep, best)]
        summary = rm.training_path / "train_summary.csv"
        if summary.exists():   # keep the episodes already saved in the run summary
            rm.training_summary = pd.read_csv(summary)
            rm.training_summary = rm.training_summary[rm.training_summary["episode"] < start_episode]
        print(f"resumed from {path.name}: next episode {start_episode}, epsilon {agent.epsilon:.3f}, "
              f"buffer {len(agent.replay_buffer)}, best validation {best:.4f}", flush=True)
        logger.info(f"Resumed from {path}: next episode {start_episode}")
    else:
        rm.save_hyperparameters(params=hyper, reward_params=reward_params)

    config = {**hyper, "reward": reward_params}
    tracker = Tracker(args.wandb, project=args.wandb_project, entity=args.wandb_entity,
                      name=f"grid-run{args.run_id:03d}", config=config, run_dir=rm.run_dir,
                      run_id=state["extra"].get("wandb_id") if args.resume else None, mode=args.wandb_mode,
                      tags=("GridZoneEnv", args.mode), group="GridZoneEnv")

    val_date = args.val_date or args.eval_dates[0]
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Env: GridZoneEnv on {args.bundle}: {env.n_zones} zones of {args.cell_m:g} m, "
          f"{args.days:g}-day episodes x {args.episodes}, validation every {args.val_every} on {val_date}")
    print(f"Params: {hyper}")
    print("=" * 80, flush=True)
    # progress bar as in the legacy trainer: one tick per 3-hour slot, simulated clock in the description
    bar = tqdm(total=slots_per_episode * args.episodes, initial=slots_per_episode * start_episode,
               desc="Training computation is starting ", position=0, leave=True, dynamic_ncols=True)
    out = bar.write                                     # prints above the bar without breaking it
    current = {"ep": start_episode}

    def on_slot():
        d = env.sim.datetime
        bar.set_description(f"[TRAIN] Run {args.run_id}. Epis {current['ep']}, Week {d.isocalendar().week}, "
                            f"{d:%A} at {d:%H:%M:%S}")
        bar.set_postfix({"eps": round(agent.epsilon, 3)})
        bar.update(1)
        slots["n"] += 1
        agent.update_epsilon()
        if slots["n"] % params["target_every_slots"] == 0:
            agent.update_target_network()

    val_env = None
    stop_at = args.episodes if args.stop_after is None else min(args.episodes, start_episode + args.stop_after)
    for ep in range(start_episode, stop_at):
        current["ep"] = ep
        res, s = run_episode(env, agent, seed=args.seed * 1000 + ep, device=args.device, learn=True, params=params,
                             on_slot=on_slot, cell_m=args.cell_m, episode=ep)
        bar.n = slots_per_episode * (ep + 1)            # an episode may end between slot boundaries
        bar.refresh()
        rm.save_episode(res)
        rm.save_run_summary()
        line = (f"ep {ep:3d} return {s['return']:9.1f} served {s['city_service_level']:.3f} "
                f"failures/day {res.mean_daily_failures:7.1f} moved {s['bikes_moved']:5d} "
                f"invalid {s['invalid']:5d}/{s['decisions']:5d} eps {s['epsilon']:.3f} loss {s['loss']:.4f} "
                f"wall {s['wall_s']:.0f}s")
        out(line)
        logger.info(line)
        tracker.log({f"train/{k}": v for k, v in s.items()} | {"train/mean_daily_failures": res.mean_daily_failures,
                    "train/buffer": len(agent.replay_buffer)}, step=ep)
        if (ep + 1) % args.save_every == 0 or ep == args.episodes - 1:   # models/episodes/episode_NNN/
            rm.save_model(agent, ep, res.mean_daily_failures, model_type="episode")
        if args.val_every and (ep + 1) % args.val_every == 0:
            if val_env is None:
                val_env = make_env(args.bundle, [val_date], args.days, sim_config, args.cell_m)
            bar.set_description(f"[VAL] Run {args.run_id}. After epis {ep}, greedy week from {val_date}")
            saved_eps = agent.epsilon
            vres, vs = run_episode(val_env, agent, seed=20_000, device=args.device, learn=False, params=params,
                                   cell_m=args.cell_m, episode=0, mode="validation")
            agent.epsilon = saved_eps
            vrm = ResultsManager(args.results_path, args.run_id, overwrite=True, interactive=False, mode="validation",
                                 val_tag=f"episode_{ep:03d}")
            vrm.save_hyperparameters(params={**hyper, "validation_date": val_date, "validation_seed": 20_000},
                                     reward_params=reward_params)
            vrm.save_episode(vres)
            score = vres.mean_daily_failures
            improved = score < best
            tracker.log({f"val/{k}": vs[k] for k in ("city_service_level", "failed", "return", "bikes_moved",
                                                       "invalid", "truck_km")}
                        | {"val/mean_daily_failures": score}, step=ep)
            if improved:
                best, best_ep = score, ep
                rm.save_model(agent, ep, score, model_type="best")   # models/best/ + best_models_summary.csv
                tracker.summary({"best/val_mean_daily_failures": best, "best/episode": ep,
                                 "best/val_city_service_level": vs["city_service_level"]})
            line = (f"val ep {ep:3d} ({val_date}): failures/day {score:.2f} served {vs['city_service_level']:.4f} "
                    f"{'NEW BEST' if improved else f'best {best:.2f} (ep {best_ep})'}")
            out(line)
            logger.info(f"[val] {line}")
        if args.checkpoint_every and (ep + 1) % args.checkpoint_every == 0:
            save_checkpoint(ckpt_dir, agent, {"slots": slots["n"], "best_val_failures": best, "best_episode": best_ep,
                                              "wandb_id": tracker.id},
                            episode=ep, buffer=not args.no_buffer_checkpoint, keep=args.keep_checkpoints)
    bar.close()
    if stop_at < args.episodes:
        print(f"stopped after episode {stop_at - 1}; continue with --resume", flush=True)
        tracker.finish()
        return 0
    rm.save_run_summary()
    last = rm.training_summary["mean_daily_failures"].iloc[-1] if len(rm.training_summary) else float("nan")
    final_path = rm.save_model(agent, args.episodes - 1, float(last), model_type="final")
    logger.info("Training finished")
    # W&B keeps one model per run: the best by validation, uploaded once (issue #174)
    best_path = rm.get_best_model_path()
    if best_path is not None:
        tracker.log_model(best_path, ["best"],
                          {"episode": best_ep, "val_mean_daily_failures": best, "val_date": val_date})
    else:   # no validation ran: the final model is the best we have
        tracker.log_model(final_path, ["best"],
                          {"episode": args.episodes - 1, "note": "no validation; final model"})

    # greedy evaluation on held-out weeks, as the webapp's validation view
    eval_env = make_env(args.bundle, args.eval_dates, args.days, sim_config, args.cell_m)
    val = ResultsManager(args.results_path, args.run_id, overwrite=True, interactive=False, mode="validation",
                         val_tag=f"final_episode_{args.episodes - 1:03d}")
    val.save_hyperparameters(params={**hyper, "eval_dates": args.eval_dates, "eval_seeds": args.eval_seeds},
                             reward_params=reward_params)
    i = 0
    final_eval = []
    ebar = tqdm(total=len(args.eval_dates) * args.eval_seeds, desc=f"[EVAL] Run {args.run_id}", leave=True,
                dynamic_ncols=True)
    for d, day in enumerate(args.eval_dates):
        eval_env.start_dates = (day,)
        for s_ in range(args.eval_seeds):
            ebar.set_description(f"[EVAL] Run {args.run_id}. Week from {day}, seed {s_}")
            res, s = run_episode(eval_env, agent, seed=10_000 + 100 * d + s_, device=args.device, learn=False,
                                 params=params, cell_m=args.cell_m, episode=i, mode="validation")
            val.save_episode(res)
            final_eval.append(s)
            ebar.write(f"eval {day} seed {s_}: served {s['city_service_level']:.3f} moved {s['bikes_moved']} "
                       f"failures {s['failed']}")
            ebar.update(1)
            i += 1
    ebar.close()
    val.save_run_summary()
    tracker.summary({"eval/city_service_level": float(np.mean([e["city_service_level"] for e in final_eval])),
                     "eval/failed": float(np.mean([e["failed"] for e in final_eval]))})
    tracker.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
