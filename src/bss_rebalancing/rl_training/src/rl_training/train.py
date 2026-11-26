import os
import sys
import json
import subprocess
import gymnasium
from dataclasses import dataclass
import torch
import argparse
import gc
import warnings
import logging

import gymnasium_env  # noqa: F401 — registers the gym environment

import gymnasium as gym
import numpy as np
import multiprocessing as mp

from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions
from gymnasium_env.envs.fully_dynamic_env import EnvDefaults, RewardComponents

from rl_training.agents import DQNAgent
from rl_training.memory import ReplayBuffer
from rl_training.results import ResultsManager, EpisodeResults
from rl_training.logging_config import init_logging, LoggingConfig, get_logger
from rl_training.utils import (
    convert_graph_to_data,
    convert_seconds_to_hours_minutes,
    set_seed,
    setup_device,
    build_cell_graph_from_cells,
    update_cell_graph_features
)

# ------------------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------------------

devices = ["cpu"]
if torch.cuda.is_available():
    num_cuda = torch.cuda.device_count()
    for i in range(num_cuda):
        devices.append(f"cuda:{i}")
if torch.backends.mps.is_available():
    devices.append("mps")

# Only print in the main process, not in spawned children
if mp.current_process().name == "MainProcess":
    print(f"Devices available: {devices}\n")

# ------------------------------------------------------------------------------
# Default params
# ------------------------------------------------------------------------------

params = {
    "seed": int(42),                        # Random seed for reproducibility
    "num_episodes": 140,                    # Total number of training episodes
    "batch_size": int(64),                  # Batch size for replay buffer sampling
    "replay_buffer_capacity": int(1e5),     # Capacity of replay buffer: 0.1 million transitions
    "gamma": 0.95,                          # Discount factor
    "epsilon_start": 1.0,                   # Starting exploration rate
    "epsilon_delta": 0.05,                  # Epsilon decay rate
    "epsilon_end": 0.01,                    # Minimum exploration rate
    "epsilon_decay": 1e-5,                  # Epsilon decay constant
    "exploration_time": 0.6,                # Fraction of total training time for exploration
    "lr": 1e-4,                             # Learning rate
    "soft_update": True,                    # Use soft update for target network
    "tau": 0.005,                           # Tau parameter for soft update

    "total_timeslots": 56,                  # Total number of time slots in one episode (1 month)
    "maximum_number_of_bikes": 500,         # Maximum number of bikes in the system
    "minimum_number_of_bikes": 1,           # Minimum number of bikes per cell
    "enable_repositioning": False,          # Use base repositioning strategy at the start of each episode
    "use_net_flow": False,                  # Use net flow repositioning strategy at the start of each episode
    "critic_mode": "poisson",               # Critic forecast: "poisson" (sampled from rates) or "oracle" (realised future trips)
    "depot_position_id": 18,                # ID (cell) of the depot position
    "initial_cell_id": 18,                  # Initial cell where the truck starts

    "validation_epsilon_threshold": 0.1,
    "validation_timeout": 600,
}

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the preprocessing CLI."""
    parser = argparse.ArgumentParser(
        description="BSS Train Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Run full preprocessing pipeline
            bss-train --data-path data/

            # Specify run ID and results path
            bss-train --run-id 1 --data-path data/ --results-path results/

            # Use GPU device
            bss-train --data-path data/ --device cuda:0

            # Set random seed and number of episodes
            bss-train --data-path data/ --seed 123 --num-episodes 150

            # Enable logging
            bss-train --data-path data/ --enable-logging 

            # Perform only one validation at the end of training
            bss-train --data-path data/ --one-validation  

            # Perform training with number of bikes and exploration time
            bss-train --data-path data/ --num-bikes 300 --exploration-time 0.8

            # Use a separate GPU for validation subprocesses
            bss-train --data-path data/ --device cuda:0 --val-device cuda:1
        """,
    )

    parser.add_argument(
        '--run-id',
        type=int,
        default=0,
        help='Run ID for the experiment.'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/',
        help='Path to the data folder.'
    )
    parser.add_argument(
        '--results-path',
        type=str,
        default='results/',
        help='Path to the results folder.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        help=f'Hardware device to use. Available options: {devices}.'
    )
    parser.add_argument(
        '--val-device',
        type=str,
        default=None,
        help=f'Hardware device to use for validation subprocesses. Falls back to --device if not specified. Available options: {devices}.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=params["seed"],
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=params['num_episodes'],
        help='Number of episodes to train.'
    )
    parser.add_argument(
        '--max-num-bikes',
        type=int,
        default=params['maximum_number_of_bikes'],
        help='Number of bikes of the system.'
    )
    parser.add_argument(
        '--min-num-bikes',
        type=int,
        default=params['minimum_number_of_bikes'],
        help='Minimum number of bikes per cell.'
    )
    parser.add_argument(
        '--enable-repositioning',
        action='store_true',
        help='Enable repositioning beyond minimum bikes per cell at the start of each episode.'
    )
    parser.add_argument(
        '--use-net-flow',
        action='store_true',
        help='Use net-flow-based repositioning instead of random at the start of each episode.'
    )
    parser.add_argument(
        '--critic-mode',
        choices=['poisson', 'oracle'],
        default=params['critic_mode'],
        help='How the critic score anticipates future demand: "poisson" = forecast sampled from the demand '
             'rates, independent of the simulated trips (default); "oracle" = reads the realised future trips.'
    )
    parser.add_argument(
        '--exploration-time',
        type=float,
        default=params['exploration_time'],
        help='Number of episodes to explore.'
    )
    parser.add_argument(
        '--log',
        action='store_true',
        help='Enable logging.'
    )
    parser.add_argument(
        '--one-validation',
        action='store_true',
        help='Performs only one validation at the end of the training.'
    )

    return parser


# ------------------------------------------------------------------------------
# Subprocess-based validation helpers
# ------------------------------------------------------------------------------

def _get_validate_script_path() -> str:
    """
    Resolve the absolute path to validate.py.

    Strategy (in order):
      1. Same directory as this train.py file  ← works for src-layout packages
      2. `bss-validate` console-script on PATH  ← works if installed via pip/setup.py
      3. Raises immediately so you know rather than silently failing.
    """
    candidate = Path(__file__).parent / "validate.py"
    if candidate.exists():
        return str(candidate)

    import shutil
    entry = shutil.which("bss-validate")
    if entry:
        return str(entry)

    raise FileNotFoundError(
        "Cannot locate validate.py. Expected it next to train.py, "
        "or 'bss-validate' on PATH (installed via pip)."
    )


def _build_validate_cmd(
        run_id: int,
        data_path: str,
        results_path: str,
        episode: int,
        val_device: str,
        seed: int,
        max_num_bikes: int,
        min_num_bikes: int,
        total_timeslots: int,
        enable_repositioning: bool,
        use_net_flow: bool,
        critic_mode: str,
) -> list[str]:
    """
    Build the argv list to invoke validate.py as a completely independent subprocess
    — exactly as if you typed it in your terminal.
    Uses sys.executable so the subprocess runs in the same venv as training.
    """
    validate_script = str(_get_validate_script_path())
    cmd = [
        sys.executable, validate_script,
        "--run-id", str(run_id),
        "--data-path", data_path,
        "--results-path", results_path,
        "--model-type", "episode",
        "--model-episode", str(episode),
        "--device", val_device,
        "--seed", str(seed),
        "--max-num-bikes", str(max_num_bikes),
        "--min-num-bikes", str(min_num_bikes),
        "--total-timeslots", str(total_timeslots),
        "--critic-mode", critic_mode,
        "--non-interactive",
    ]
    if enable_repositioning:
        cmd.append("--enable-repositioning")
    if use_net_flow:
        cmd.append("--use-net-flow")
    return cmd


@dataclass
class _PendingVal:
    """Tracks a validation subprocess running in parallel with training."""
    episode: int
    proc: subprocess.Popen


def _launch_validation_subprocess(
        cmd: list[str],
        episode: int,
        logger: logging.Logger,
) -> '_PendingVal | None':
    """
    Spawn validate.py as fire-and-forget — training continues immediately.
    Returns a _PendingVal handle to be collected later.
    stdout/stderr are inherited so the validation tqdm bar prints inline.
    """
    logger.info(f"[val] Launching validation subprocess for episode {episode}: {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        return _PendingVal(episode=episode, proc=proc)
    except Exception as e:
        logger.error(f"[val] Failed to launch validation subprocess for episode {episode}: {e}")
        print(f"[VAL] Could not launch validation for episode {episode}: {e}")
        return None


def _collect_pending_val(
        pending: '_PendingVal',
        logger: logging.Logger,
        timeout: int = int(params['validation_timeout']),
) -> bool:
    """
    Wait (up to `timeout` seconds) for a previously launched validation subprocess
    to finish. Returns True on clean exit, False on timeout or non-zero exit.
    Called at the next validation gate, so training is never stalled mid-episode.
    """
    logger.info(f"[val] Waiting for validation of episode {pending.episode} to finish...")

    try:
        pending.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pending.proc.kill()
        pending.proc.wait()
        logger.warning(
            f"[val] Validation subprocess for episode {pending.episode} timed out "
            f"after {timeout}s — skipping best-model check."
        )
        print(f"[VAL] Validation timed out for episode {pending.episode}, skipping.")
        return False

    if pending.proc.returncode != 0:
        logger.warning(
            f"[val] Validation subprocess for episode {pending.episode} "
            f"exited with code {pending.proc.returncode}."
        )
        print(f"[VAL] Validation exited non-zero ({pending.proc.returncode}) for episode {pending.episode}.")
        return False

    logger.info(f"[val] Validation for episode {pending.episode} completed successfully.")
    return True


def _read_validation_score(
        results_path: str,
        run_id: int,
        episode: int,
        logger: logging.Logger,
) -> float | None:
    """
    After the validation subprocess finishes, read the scalar JSON it wrote and
    return `mean_daily_failures` (lower is better).  Returns None on any I/O error.
    """
    val_tag = ResultsManager.build_val_tag("episode", episode)
    scalars_path = (
            Path(results_path) / f"run_{run_id:03d}" / "validation" / val_tag
            / "episode_000" / "scalars.json"
    )

    try:
        with open(scalars_path, "r") as f:
            scalars = json.load(f)
        score = float(scalars["mean_daily_failures"])
        logger.info(f"[val] Episode {episode} validation score (mean_daily_failures): {score:.4f}")
        return score
    except FileNotFoundError:
        logger.warning(f"[val] Scalars file not found at {scalars_path}. Skipping best-model update.")
        return None
    except Exception as e:
        logger.error(f"[val] Failed to read validation score for episode {episode}: {e}")
        return None


# ------------------------------------------------------------------------------
# train_dqn
# ------------------------------------------------------------------------------

def train_dqn(
        env: gymnasium.Env,
        agent: DQNAgent,
        batch_size: int,
        episode: int,
        device: torch.device,
        run_id: int,
        logging_enabled: bool,
        logger: logging.Logger,
        tbar=None,
        episode_results_path: str | None = None,
        seed: int = None,
) -> dict:
    # ============================================================================
    # Initialize metrics tracking
    # ============================================================================
    # Per-timeslot metrics
    rewards = []
    failures = []
    epsilons = []
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    traveling_bikes = []
    demand_per_timeslot = []
    q_values = []

    # Per-step metrics
    action_per_step = []
    global_critic_scores = []
    reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}

    # Accumulators (reset each timeslot)
    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    last_cumulative_demand = 0
    iterations = 0

    # ============================================================================
    # Environment setup and reset
    # ============================================================================
    reset_options = {
        'total_timeslots': params["total_timeslots"],
        'maximum_number_of_bikes': params["maximum_number_of_bikes"],
        'minimum_number_of_bikes': params["minimum_number_of_bikes"],
        'enable_repositioning': params["enable_repositioning"],
        'use_net_flow': params["use_net_flow"],
        'critic_mode': params["critic_mode"],
        'discount_factor': params["gamma"],
        'depot_id': params['depot_position_id'],
        # 'initial_cell': params['initial_cell_id'],
    }
    if episode_results_path is not None:
        reset_options['results_path'] = episode_results_path

    agent_state, info = env.reset(seed=seed, options=reset_options)

    # Extract static environment info
    cell_dict = info['cell_dict']
    nodes_dict = info['nodes_dict']
    distance_lookup = info['distance_lookup']

    # Build initial graph from cells (reads metrics automatically)
    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict,
        nodes_dict=nodes_dict,
        distance_lookup=distance_lookup
    )

    # Define which metrics to use as GNN features: THEY SHOULD MATCH THE FEATURES USED IN THE CELLS
    gnn_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'total_bikes',
    ]

    # Initialize state
    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # ============================================================================
    # Main training loop
    # ============================================================================
    episode_cell_stats = {
        cell_id: {
            'critic_sum': 0.0,
            'eligibility_sum': 0.0,
            'bikes_sum': 0.0,
            'bikes_dead_sum': 0.0
        }
        for cell_id in cell_dict.keys()
    }

    done = False
    while not done:
        # ── State (S) → device ──────────────────────────────────────────────────
        single_state = Data(
            x=state.x.to(device),
            edge_index=state.edge_index.to(device),
            edge_attr=state.edge_attr.to(device),
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0).to(device),
            batch=torch.zeros(state.x.size(0), dtype=torch.long).to(device),
        )

        # ── Action (A) selection ────────────────────────────────────────────────
        action = agent.select_action(single_state, epsilon_greedy=True)

        # ── Environment step ────────────────────────────────────────────────────
        agent_state, reward, done, timeslot_terminated, info = env.step(action)

        # ── Graph update ────────────────────────────────────────────────────────
        cell_dict = info['cell_dict']
        update_cell_graph_features(cell_graph, cell_dict)

        # ── Cell stats accumulation ─────────────────────────────────────────────
        for cell_id, cell in cell_dict.items():
            stats = episode_cell_stats[cell_id]
            stats['critic_sum'] += cell.get_critic_score()
            stats['eligibility_sum'] += cell.get_eligibility_score()
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] += cell.get_dead_bikes()

        # ── Build next state ────────────────────────────────────────────────────
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # ── Replay push ─────────────────────────────────────────────────────────
        # Semi-MDP transition (issue #164): the action just taken lasted info['steps'] steps of
        # 30 s; the legacy env gives one reward per action, received when it is chosen. Episodes
        # end on a time limit (truncation), so the next state is never terminal.
        agent.replay_buffer.push(state, action, reward, next_state, done, steps=info['steps'], terminal=False)

        # ── Train step ──────────────────────────────────────────────────────────
        agent.train_step(batch_size)

        # ── Scalar bookkeeping ──────────────────────────────────────────────────
        action_per_step.append(action)
        reward_tracking_per_action[action].append(reward)
        global_critic_scores.append(info['global_critic_score'])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info['failures'])
        iterations += 1

        # ── Timeslot boundary ───────────────────────────────────────────────────
        if timeslot_terminated:
            timeslots_completed += 1

            # Update networks
            if timeslots_completed % 8 == 0:
                agent.update_target_network()
            agent.update_epsilon()

            # Record Q-values (expensive - only once per timeslot)
            with torch.no_grad():
                q_val_tensor = agent.get_q_values(single_state)
                q_values.append(q_val_tensor[0].squeeze().cpu().numpy())

            # Record timeslot metrics
            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            epsilons.append(agent.epsilon)
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])

            current = sum(cell.get_total_demand() for cell in cell_dict.values())
            demand_per_timeslot.append(current - last_cumulative_demand)
            last_cumulative_demand = current

            # Reset accumulators
            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0

            # Update progress bar
            if tbar is not None:
                tbar.set_description(
                    f"[TRAIN] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.set_postfix({'eps': agent.epsilon})
                tbar.update(1)

        # Move to next state
        state = next_state
        del single_state

    # Cleanup
    torch.cuda.empty_cache()

    # ============================================================================
    # Post-episode cell stats
    # ============================================================================
    steps_in_episode = iterations
    for cell_id, stats in episode_cell_stats.items():
        center_node = cell_dict[cell_id].get_center_node()
        if center_node not in cell_graph.nodes:
            continue

        if steps_in_episode > 0:
            critic_mean = stats.get('critic_sum', 0.0) / steps_in_episode
            eligibility_mean = stats.get('eligibility_sum', 0.0) / steps_in_episode
            bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
            dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
        else:
            critic_mean = eligibility_mean = bikes_mean = dead_bikes_mean = 0.0

        nx_attrs = cell_graph.nodes[center_node]

        nx_attrs['critic_mean'] = critic_mean
        nx_attrs['eligibility_mean'] = eligibility_mean
        nx_attrs['failure_sum'] = cell_dict[cell_id].get_failures()
        nx_attrs['failure_rate'] = cell_dict[cell_id].get_failure_rate()
        nx_attrs['visits_sum'] = cell_dict[cell_id].get_visits()
        nx_attrs['ops_sum'] = cell_dict[cell_id].get_ops()
        nx_attrs['pick_ups_sum'] = cell_dict[cell_id].get_pick_ups()
        nx_attrs['drops_sum'] = cell_dict[cell_id].get_drops()
        nx_attrs['success_rebalancing'] = cell_dict[cell_id].get_total_rebalanced()
        nx_attrs['bikes_mean'] = bikes_mean
        nx_attrs['dead_bikes_mean'] = dead_bikes_mean

    # ============================================================================
    # Return results
    # ============================================================================
    return {
        "rewards_per_timeslot": rewards,
        "failures_per_timeslot": failures,
        "total_invalid_actions": info["total_invalid_actions"],
        "q_values_per_timeslot": q_values,
        "action_per_step": action_per_step,
        "global_critic_scores": global_critic_scores,
        "reward_tracking_per_action": reward_tracking_per_action,
        "epsilon_per_timeslot": epsilons,
        "deployed_bikes": system_bikes,
        "truck_load": truck_load,
        "depot_load": depot_load,
        "outside_system_bikes": outside_system_bikes,
        'traveling_bikes': traveling_bikes,
        "demand_per_timeslot": demand_per_timeslot,
        "cell_subgraph": cell_graph,
    }


# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main():
    # spawn is required before any CUDA context is created
    mp.set_start_method('spawn', force=True)
    warnings.filterwarnings("ignore")

    args = create_parser().parse_args()

    device = setup_device(args.device.lower(), devices)
    val_device = setup_device(args.val_device.lower(), devices) if args.val_device else device

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------
    run_id = args.run_id
    data_path = args.data_path
    results_path = args.results_path
    logging_enabled = args.log

    params['seed'] = args.seed
    params['num_episodes'] = args.num_episodes
    params['maximum_number_of_bikes'] = args.max_num_bikes
    params['minimum_number_of_bikes'] = args.min_num_bikes
    params['enable_repositioning'] = args.enable_repositioning
    params['use_net_flow'] = args.use_net_flow
    params['critic_mode'] = args.critic_mode
    params['exploration_time'] = args.exploration_time

    print(f"Setting seed: {params['seed']}")
    set_seed(params['seed'])

    # Ensure the data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data path does not exist: {data_path}")

    # epsilon(t) = eps_min + (eps_max - eps_min) * exp(-t^2 / decay), with t in timeslots.
    # With decay = T^2 / ln(10) and T = exploration_time * total timeslots, the exponential
    # term equals 0.1 at t = T, i.e. epsilon ~= 0.109 after `exploration_time` of the training.
    params["epsilon_decay"] = ((params["exploration_time"] * params["num_episodes"] * params[
        "total_timeslots"]) ** 2) / np.log(10)

    # ------------------------------------------------------------------
    # ResultsManager
    # ------------------------------------------------------------------
    results_manager = ResultsManager(
        results_path=results_path,
        run_id=run_id,
        overwrite=False,
        interactive=True
    )

    # Init logging
    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(results_manager.training_path, "logs"),
        run_id=run_id,
        console=False,
        logger_name="train",
    ))
    logger = get_logger("train", logger_name="train")
    logger.info("Starting training loop")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = gym.make(
        'gymnasium_env/FullyDynamicEnv-v0',
        data_path=data_path,
        results_path=f"{str(results_manager.training_path)}/",
        seed=params['seed'],
        logging_enabled=logging_enabled
    )

    # Save hyperparameters
    results_manager.save_hyperparameters(
        params={
            **params,
            **{k: v for k, v in vars(EnvDefaults).items() if not k.startswith('_')},
        },
        reward_params={k: v for k, v in vars(RewardComponents).items() if not k.startswith('_')}
    )

    print("=" * 80)
    print(f"Device: {device}")
    print(f"Validation device: {val_device}")
    print(f"Params: {params}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Agent with replay buffer
    # ------------------------------------------------------------------
    # Set up replay buffer
    replay_buffer = ReplayBuffer(params["replay_buffer_capacity"])

    # Initialize the DQN agent
    agent = DQNAgent(
        seed=int(params['seed']),
        replay_buffer=replay_buffer,
        num_actions=env.action_space.n,
        observation_space_len=env.observation_space.shape[0],
        gamma=params["gamma"],
        epsilon_start=params["epsilon_start"],
        epsilon_end=params["epsilon_end"],
        epsilon_decay=params["epsilon_decay"],
        lr=params["lr"],
        device=device,
        tau=params["tau"],
        soft_update=params["soft_update"],
    )
    print("Model initialized successfully.\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    num_days = params["total_timeslots"] / 8  # float: stays valid for episodes shorter than a day

    # Best validation score tracker (lower mean_daily_failures = better)
    best_val_score = float("inf")

    # Holds the currently running validation subprocess (if any).
    # There is at most one pending validation at a time; we collect it
    # before launching the next one so saving is always serial.
    pending_val: _PendingVal | None = None

    try:
        tbar = tqdm(
            range(int(params["total_timeslots"] * params["num_episodes"])),
            desc="Training computation is starting ",
            position=0,
            leave=True,
            dynamic_ncols=True
        )

        logger.info(f"Training started with the following parameters: {params}")

        # Train loop
        for episode in range(int(params["num_episodes"])):
            current_seed = int(params['seed'] + episode)
            # ------------------------------------------------------------------
            # Train one episode
            # ------------------------------------------------------------------
            training_dict = train_dqn(
                env=env,
                agent=agent,
                batch_size=int(params["batch_size"]),
                episode=episode,
                device=device,
                run_id=run_id,
                logging_enabled=logging_enabled,
                logger=logger,
                tbar=tbar,
                episode_results_path=os.path.join(f"{str(results_manager.training_path)}", f"episode_{episode:03d}"),
                seed=current_seed
            )

            # Build EpisodeResults
            training_results = EpisodeResults(
                episode=episode,
                mode='train',
                seed=current_seed,
                epsilon=agent.epsilon,
                epsilon_per_timeslot=training_dict['epsilon_per_timeslot'],
                rewards_per_timeslot=training_dict['rewards_per_timeslot'],
                demand_per_timeslot=training_dict['demand_per_timeslot'],
                total_reward=sum(training_dict['rewards_per_timeslot']),
                failures_per_timeslot=training_dict['failures_per_timeslot'],
                total_failures=sum(training_dict['failures_per_timeslot']),
                mean_daily_failures=sum(training_dict['failures_per_timeslot']) / num_days,
                q_values_per_timeslot=training_dict['q_values_per_timeslot'],
                mean_q_values=float(np.mean(training_dict['q_values_per_timeslot'])) if training_dict[
                    'q_values_per_timeslot'] else 0.0,
                deployed_bikes=training_dict['deployed_bikes'],
                truck_load=training_dict['truck_load'],
                depot_load=training_dict['depot_load'],
                outside_system_bikes=training_dict['outside_system_bikes'],
                action_per_step=training_dict['action_per_step'],
                total_invalid_actions=training_dict['total_invalid_actions'],
                reward_tracking_per_action=training_dict['reward_tracking_per_action'],
                global_critic_scores=training_dict['global_critic_scores'],
                cell_subgraph=training_dict['cell_subgraph'],
                traveling_bikes=training_dict['traveling_bikes'],
            )

            # Save training episode results
            results_manager.save_episode(training_results)

            if (
                    not args.one_validation
                    and agent.epsilon < params['validation_epsilon_threshold']
            ):

                # ── Step A: collect the previous val subprocess (if any) ──────
                # This is the only point where training may briefly wait, and
                # only if val_N-1 hasn't finished by the time train_N is done.
                if pending_val is not None:
                    val_ok = _collect_pending_val(pending_val, logger, int(params['validation_timeout']))
                    if val_ok:
                        val_score = _read_validation_score(
                            results_path=results_path,
                            run_id=run_id,
                            episode=pending_val.episode,
                            logger=logger,
                        )
                        if val_score is not None and val_score < best_val_score:
                            prev_best = best_val_score
                            best_val_score = val_score
                            # Promote the already-saved episode snapshot — do NOT
                            # use the live agent weights (we are now one episode ahead).
                            results_manager.promote_episode_to_best(
                                episode=pending_val.episode,
                                score=val_score,
                            )
                            logger.info(
                                f"[val] Episode {pending_val.episode}: NEW BEST promoted! "
                                f"val_score={val_score:.4f} (prev best={prev_best:.4f})"
                            )
                        elif val_score is not None:
                            logger.info(
                                f"[val] Episode {pending_val.episode}: "
                                f"val_score={val_score:.4f} did not beat best={best_val_score:.4f}"
                            )
                    pending_val = None

                # ── Step B: save this episode's model snapshot ────────────────
                # Uses this episode's own training score as metadata.
                # The snapshot is what the validator will load.
                results_manager.save_model(
                    agent=agent,
                    episode=episode,
                    score=training_results.mean_daily_failures,
                    model_type='episode'
                )
                logger.info(f"Episode {episode}: model snapshot saved (epsilon={agent.epsilon:.4f})")

                # ── Step C: launch the new val subprocess (non-blocking) ──────
                val_cmd = _build_validate_cmd(
                    run_id=run_id,
                    data_path=data_path,
                    results_path=results_path,
                    episode=episode,
                    val_device=str(val_device),
                    seed=int(params['seed']),
                    max_num_bikes=int(params['maximum_number_of_bikes']),
                    min_num_bikes=int(params['minimum_number_of_bikes']),
                    total_timeslots=int(params['total_timeslots']),
                    enable_repositioning=bool(params['enable_repositioning']),
                    use_net_flow=bool(params['use_net_flow']),
                    critic_mode=str(params['critic_mode']),
                )
                pending_val = _launch_validation_subprocess(val_cmd, episode, logger)

            logger.info(
                f"Episode {episode}: Seed = {current_seed}, "
                f"Mean Failures = {training_results.mean_daily_failures:.2f}, "
                f"Total Failures = {training_results.total_failures}, "
                f"Invalid Actions = {training_results.total_invalid_actions}, "
                f"Epsilon = {agent.epsilon:.4f}"
            )

            gc.collect()

        # ------------------------------------------------------------------
        # --one-validation: validate only the final model
        # ------------------------------------------------------------------
        if args.one_validation:
            last_episode = int(params["num_episodes"]) - 1
            results_manager.save_model(
                agent=agent,
                episode=last_episode,
                score=training_results.mean_daily_failures,
                model_type='episode'
            )
            logger.info(f"Episode {last_episode}: final model snapshot saved for single validation")
            val_cmd = _build_validate_cmd(
                run_id=run_id,
                data_path=data_path,
                results_path=results_path,
                episode=last_episode,
                val_device=str(val_device),
                seed=int(params['seed']),
                max_num_bikes=int(params['maximum_number_of_bikes']),
                min_num_bikes=int(params['minimum_number_of_bikes']),
                total_timeslots=int(params['total_timeslots']),
                enable_repositioning=bool(params['enable_repositioning']),
                use_net_flow=bool(params['use_net_flow']),
                critic_mode=str(params['critic_mode']),
            )
            pending_val = _launch_validation_subprocess(val_cmd, last_episode, logger)

        # ------------------------------------------------------------------
        # End of training — collect any still-running validation
        # ------------------------------------------------------------------
        if pending_val is not None:
            print(f"\n[VAL] Training finished. Waiting for last validation (episode {pending_val.episode})...")
            val_ok = _collect_pending_val(pending_val, logger, int(params['validation_timeout']))
            if val_ok:
                val_score = _read_validation_score(
                    results_path=results_path,
                    run_id=run_id,
                    episode=pending_val.episode,
                    logger=logger,
                )
                if val_score is not None and val_score < best_val_score:
                    prev_best = best_val_score
                    best_val_score = val_score
                    results_manager.promote_episode_to_best(
                        episode=pending_val.episode,
                        score=val_score,
                    )
                    logger.info(
                        f"[val] Final best: episode {pending_val.episode}, "
                        f"val_score={val_score:.4f} (prev best={prev_best:.4f})"
                    )
                    print(
                        f"\n[VAL] ✓ Final best model: episode {pending_val.episode} "
                        f"— mean_daily_failures={val_score:.4f}"
                    )

        # Save aggregated summaries
        results_manager.save_run_summary()
        logger.info("Training completed successfully")
        tbar.close()
        env.close()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        if pending_val is not None and pending_val.proc.poll() is None:
            print(f"[VAL] Terminating background validation for episode {pending_val.episode}...")
            pending_val.proc.terminate()
        env.close()
        return
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if pending_val is not None and pending_val.proc.poll() is None:
            pending_val.proc.terminate()
        env.close()
        raise

    print(f"\nTraining {run_id} completed.")
    print(f"Best validation score (mean_daily_failures): {best_val_score:.4f}")


if __name__ == "__main__":
    main()