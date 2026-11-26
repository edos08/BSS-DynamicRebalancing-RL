"""
Standalone validation script for trained DQN agents.
Mirrors the validate_dqn() / _validation_worker() logic from train.py exactly.
"""

import os
import argparse
import warnings
import logging
import torch

import gymnasium_env  # noqa: F401 — registers the gym environment
import gymnasium as gym

from tqdm import tqdm
from torch_geometric.data import Data
from gymnasium_env.simulator.utils import Actions
from gymnasium_env.envs.fully_dynamic_env import EnvDefaults, RewardComponents

from rl_training.agents import DQNAgent
from rl_training.results import ResultsManager, EpisodeResults
from rl_training.logging_config import init_logging, LoggingConfig, get_logger
from rl_training.utils import (
    convert_graph_to_data,
    convert_seconds_to_hours_minutes,
    set_seed,
    setup_device,
    build_cell_graph_from_cells,
    update_cell_graph_features,
)

# ------------------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------------------

devices = ["cpu"]
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")
if torch.backends.mps.is_available():
    devices.append("mps")

# print(f"Devices available: {devices}\n")

# ------------------------------------------------------------------------------
# Default params
# ------------------------------------------------------------------------------

params = {
    "seed": 42,
    "total_timeslots": 56,
    "maximum_number_of_bikes": 500,
    "minimum_number_of_bikes": 1,
    "gamma": 0.95,
    "lr": 1e-4,
    "tau": 0.005,
    "enable_repositioning": False,
    "use_net_flow": False,
    "critic_mode": "poisson",
    "depot_position_id": 18,
    "initial_cell_id": 18,
}

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BSS Validation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate the best model from run 0
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type best

  # Validate a specific episode
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-type episode --model-episode 42

  # Validate an arbitrary .pt file
  bss-validate --run-id 0 --data-path data/ --results-path results/ --model-path results/run_000/models/final/trained_agent.pt

  # Override number of bikes and use GPU
  bss-validate --run-id 0 --data-path data/ --max-num-bikes 300 --device cuda:0

  # Non-interactive mode (used by training subprocess — no stdin prompts, fail fast)
  bss-validate --run-id 0 --data-path data/ --model-type episode --model-episode 42 --non-interactive
        """,
    )

    # --- run / paths ---
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help="Run ID whose models/results to use."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/",
        help="Path to the data folder."
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="results/",
        help="Path to the results folder (same root used during training)."
    )

    # --- model source (mutually exclusive-ish: either --model-type or --model-path) ---
    parser.add_argument(
        "--model-type",
        type=str,
        default="best",
        choices=["best", "episode", "final"],
        help="Which saved model to load from the run directory (default: best).",
    )
    parser.add_argument(
        "--model-episode",
        type=int,
        default=None,
        help="Episode number to load when --model-type=episode.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Direct path to a trained_agent.pt file. Takes priority over --model-type.",
    )

    # --- environment overrides ---
    parser.add_argument(
        "--max-num-bikes",
        type=int,
        default=params["maximum_number_of_bikes"],
        help="Maximum number of bikes in the system."
    )
    parser.add_argument(
        "--min-num-bikes",
        type=int,
        default=params["minimum_number_of_bikes"],
        help="Minimum number of bikes per cell."
    )
    parser.add_argument(
        "--total-timeslots",
        type=int,
        default=params["total_timeslots"],
        help="Total timeslots for the validation episode (default: 56 = 1 week)."
    )
    parser.add_argument(
        "--enable-repositioning",
        action="store_true",
        help="Enable repositioning at the start of the episode."
    )
    parser.add_argument(
        "--use-net-flow",
        action="store_true",
        help="Use net-flow repositioning strategy."
    )
    parser.add_argument(
        "--critic-mode",
        choices=["poisson", "oracle"],
        default=params["critic_mode"],
        help="Critic forecast: 'poisson' (sampled from the demand rates, default) or 'oracle' "
             "(realised future trips). Use the mode the model was trained with."
    )

    # --- misc ---
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help=f"Hardware device. Available: {devices}."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=params["seed"],
        help="Random seed."
    )
    parser.add_argument(
        "--num-seed-runs",
        type=int,
        default=1,
        help="Number of validation runs with incremented seeds (default: 1)."
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable environment logging."
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help=(
            "Disable interactive stdin prompts. "
            "Used when validate.py is spawned as a subprocess by train.py. "
            "Causes ResultsManager to raise immediately if the output directory already "
            "exists (instead of asking the user), so the training loop can detect the "
            "failure rather than blocking forever waiting for input."
        ),
    )

    return parser

# ------------------------------------------------------------------------------
# validate_dqn
# ------------------------------------------------------------------------------

def validate_dqn(
        env,
        agent: DQNAgent,
        episode: int,
        device: torch.device,
        run_id: int,
        logging_enabled: bool,
        logger: logging.Logger,
        tbar: tqdm,
        episode_results_path: str | None = None,
        seed: int = None,
) -> dict:
    # -------------------------------------------------------------------------
    # Metrics tracking
    # -------------------------------------------------------------------------
    rewards = []
    failures = []
    system_bikes = []
    truck_load = []
    depot_load = []
    outside_system_bikes = []
    demand_per_timeslot = []
    traveling_bikes = []

    action_per_step = []
    global_critic_scores = []
    reward_tracking_per_action = {idx: [] for idx in range(len(Actions))}

    total_reward_per_timeslot = 0.0
    total_failures_per_timeslot = 0
    timeslots_completed = 0
    last_cumulative_demand = 0
    iterations = 0

    # -------------------------------------------------------------------------
    # Environment reset
    # -------------------------------------------------------------------------
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

    # Build initial graph from cells
    cell_graph = build_cell_graph_from_cells(
        cells=cell_dict,
        nodes_dict=nodes_dict,
        distance_lookup=distance_lookup
    )

    gnn_features = [
        'truck_cell',
        'critic_score',
        'eligibility_score',
        'total_bikes',
    ]

    state = convert_graph_to_data(cell_graph, node_features=gnn_features)
    state.agent_state = agent_state
    state.steps = info['steps']

    # ============================================================================
    # Main validation loop
    # ============================================================================
    episode_cell_stats = {
        cell_id: {
            'critic_sum': 0.0,
            'bikes_sum': 0.0,
            'bikes_dead_sum': 0.0,
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
            stats['bikes_sum'] += cell.get_total_bikes()
            stats['bikes_dead_sum'] += cell.get_dead_bikes()

        # ── Build next state ────────────────────────────────────────────────────
        next_state = convert_graph_to_data(cell_graph, node_features=gnn_features)
        next_state.agent_state = agent_state
        next_state.steps = info['steps']

        # ── Scalar bookkeeping ──────────────────────────────────────────────────
        action_per_step.append(action)
        reward_tracking_per_action[action].append(reward)
        global_critic_scores.append(info['global_critic_score'])
        total_reward_per_timeslot += reward
        total_failures_per_timeslot += sum(info['failures'])
        iterations += 1

        if timeslot_terminated:
            timeslots_completed += 1

            rewards.append(total_reward_per_timeslot)
            failures.append(total_failures_per_timeslot)
            system_bikes.append(info['number_of_system_bikes'])
            truck_load.append(info['truck_bikes'])
            depot_load.append(info['depot_bikes'])
            outside_system_bikes.append(info['number_of_outside_bikes'])
            traveling_bikes.append(info['number_of_traveling_bikes'])

            current = sum(cell.get_total_demand() for cell in cell_dict.values())
            demand_per_timeslot.append(current - last_cumulative_demand)
            last_cumulative_demand = current

            total_reward_per_timeslot = 0.0
            total_failures_per_timeslot = 0

            if tbar is not None:
                tbar.set_description(
                    f"[VAL] Run {run_id}. Epis {episode}, Week {info['week'] % 52}, "
                    f"{info['day'].capitalize()} at {convert_seconds_to_hours_minutes(info['time'])}"
                )
                tbar.update(1)

        state = next_state
        del single_state

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
            bikes_mean = stats.get('bikes_sum', 0.0) / steps_in_episode
            dead_bikes_mean = stats.get('bikes_dead_sum', 0.0) / steps_in_episode
        else:
            critic_mean = bikes_mean = dead_bikes_mean = 0.0

        nx_attrs = cell_graph.nodes[center_node]

        nx_attrs['critic_mean'] = critic_mean
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
        "action_per_step": action_per_step,
        "global_critic_scores": global_critic_scores,
        "reward_tracking_per_action": reward_tracking_per_action,
        "deployed_bikes": system_bikes,
        "truck_load": truck_load,
        "depot_load": depot_load,
        "outside_system_bikes": outside_system_bikes,
        "traveling_bikes": traveling_bikes,
        "demand_per_timeslot": demand_per_timeslot,
        "cell_subgraph": cell_graph,
        # Not tracked during validation — kept for EpisodeResults compatibility
        "q_values_per_timeslot": [],
        "epsilon_per_timeslot": [],
    }

# ------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    args = create_parser().parse_args()

    # ------------------------------------------------------------------
    # Params
    # ------------------------------------------------------------------
    run_id = args.run_id
    data_path = args.data_path
    results_path = args.results_path
    logging_enabled = args.log
    non_interactive = args.non_interactive

    device = setup_device(args.device.lower(), devices, non_interactive=non_interactive)

    params["seed"] = args.seed
    params["num_seed_runs"] = args.num_seed_runs
    params["total_timeslots"] = args.total_timeslots
    params["maximum_number_of_bikes"] = args.max_num_bikes
    params["minimum_number_of_bikes"] = args.min_num_bikes
    params["enable_repositioning"] = args.enable_repositioning
    params["use_net_flow"] = args.use_net_flow
    params["critic_mode"] = args.critic_mode

    set_seed(params["seed"])

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    # ------------------------------------------------------------------
    # ResultsManager
    # ------------------------------------------------------------------
    if args.model_path is None:
        val_tag = ResultsManager.build_val_tag(args.model_type, args.model_episode)
    else:
        val_tag = "custom"

    results_manager = ResultsManager(
        results_path=results_path,
        run_id=run_id,
        overwrite=non_interactive,
        interactive=not non_interactive,
        val_tag=val_tag,
        mode='validation',
    )

    # Init logging
    init_logging(LoggingConfig(
        level=logging.INFO,
        log_dir=os.path.join(str(results_manager.validation_path), "logs"),
        run_id=run_id,
        console=False,
        logger_name="validate",
    ))
    logger = get_logger("validate", logger_name="validate")
    logger.info("Starting validation loop")

    # ------------------------------------------------------------------
    # Resolve model_path from CLI args
    # ------------------------------------------------------------------
    if args.model_path is not None:
        # Direct path takes priority
        model_path = args.model_path
        logger.info(f"Using model from direct path: {model_path}")
    else:
        # Derive from run directory via ResultsManager
        if args.model_type == "best":
            model_path = results_manager.get_best_model_path()
            if model_path is None:
                raise FileNotFoundError(f"No best model found in {results_manager.models_path}")
            logger.info(f"Using best model from {model_path}")
        elif args.model_type == "final":
            model_path = results_manager.models_path / "final" / "trained_agent.pt"
            logger.info(f"Using final model from {model_path}")
        elif args.model_type == "episode":
            if args.model_episode is None:
                raise ValueError("--model-episode must be specified when --model-type=episode")
            model_path = (
                    results_manager.models_path
                    / "episodes"
                    / f"episode_{args.model_episode:03d}"
                    / "trained_agent.pt"
            )
            logger.info(f"Using episode model from {model_path}")
        else:
            raise ValueError(f"Unknown --model-type: {args.model_type}")
        model_path = str(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = gym.make(
        "gymnasium_env/FullyDynamicEnv-v0",
        data_path=data_path,
        results_path=f"{str(results_manager.validation_path)}/",
        seed=params["seed"],
        logging_enabled=logging_enabled,
    )

    # Save hyperparameters
    results_manager.save_hyperparameters(
        params={
            **params,
            **{k: v for k, v in vars(EnvDefaults).items() if not k.startswith('_')},
        },
        reward_params={k: v for k, v in vars(RewardComponents).items() if not k.startswith('_')}
    )

    if not non_interactive:
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Params: {params}")
        print("=" * 80)

    # ------------------------------------------------------------------
    # Agent — frozen, no replay buffer
    # ------------------------------------------------------------------
    agent = DQNAgent(
        seed=int(params['seed']),
        num_actions=env.action_space.n,
        observation_space_len=env.observation_space.shape[0],
        gamma=params["gamma"],
        epsilon_start=0.01,
        epsilon_end=0.01,
        epsilon_decay=1,  # no decay
        lr=params["lr"],
        device=device,
        tau=params["tau"],
        soft_update=False,
        replay_buffer=None,
    )
    agent.load_model(model_path)
    agent.train_model.eval()
    if not non_interactive:
        print("Model loaded successfully.\n")

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------
    best_score = float("inf")
    num_days = params["total_timeslots"] / 8  # float: stays valid for episodes shorter than a day

    try:
        tbar = tqdm(
            range(int(params["total_timeslots"] * params["num_seed_runs"])),
            desc="Validation computation is starting",
            position=1 if non_interactive else 0,
            leave=False if non_interactive else True,
            dynamic_ncols=True,
        )

        logger.info(f"Validation started with the following parameters: {params}")

        for episode in range(int(params["num_seed_runs"])):
            current_seed = int(params["seed"] + episode)
            set_seed(current_seed)

            validation_dict = validate_dqn(
                seed=current_seed,
                env=env,
                agent=agent,
                episode=episode,
                device=device,
                run_id=run_id,
                logging_enabled=logging_enabled,
                logger=logger,
                tbar=tbar,
                episode_results_path=os.path.join(str(results_manager.validation_path), f"episode_{episode:03d}"),
            )

            validation_results = EpisodeResults(
                episode=episode,
                mode="validation",
                seed=current_seed,
                epsilon=agent.epsilon_min,
                epsilon_per_timeslot=validation_dict.get("epsilon_per_timeslot", []),
                rewards_per_timeslot=validation_dict["rewards_per_timeslot"],
                total_reward=sum(validation_dict["rewards_per_timeslot"]),
                failures_per_timeslot=validation_dict["failures_per_timeslot"],
                total_failures=sum(validation_dict["failures_per_timeslot"]),
                mean_daily_failures=sum(validation_dict["failures_per_timeslot"]) / num_days,
                action_per_step=validation_dict["action_per_step"],
                total_invalid_actions=validation_dict["total_invalid_actions"],
                reward_tracking_per_action=validation_dict["reward_tracking_per_action"],
                q_values_per_timeslot=validation_dict.get("q_values_per_timeslot", []),
                mean_q_values=0.0,
                traveling_bikes=validation_dict['traveling_bikes'],
                deployed_bikes=validation_dict["deployed_bikes"],
                demand_per_timeslot=validation_dict['demand_per_timeslot'],
                truck_load=validation_dict["truck_load"],
                depot_load=validation_dict["depot_load"],
                outside_system_bikes=validation_dict["outside_system_bikes"],
                global_critic_scores=validation_dict.get("global_critic_scores", []),
                cell_subgraph=validation_dict["cell_subgraph"],
            )

            results_manager.save_episode(validation_results)

            is_best = validation_results.total_failures < best_score
            if is_best:
                best_score = validation_results.total_failures

            logger.info(
                f"Episode {episode} (seed={current_seed}): "
                f"failures={validation_results.total_failures} total / "
                f"{validation_results.mean_daily_failures:.2f} mean daily | "
                f"invalid={validation_results.total_invalid_actions}"
            )

        results_manager.save_run_summary()
        logger.info("Validation completed successfully")
        tbar.close()
        env.close()
    except KeyboardInterrupt:
        print("\nValidation interrupted.")
        logger.info("Validation interrupted.")
        env.close()
        return
    except Exception as e:
        logger.error(f"Validation failed: {e}.")
        env.close()
        raise

    if not non_interactive:
        print(f"\nValidation {run_id} completed.")


if __name__ == "__main__":
    main()