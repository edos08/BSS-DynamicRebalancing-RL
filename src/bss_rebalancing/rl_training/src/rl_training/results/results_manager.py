import json
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pickle
import networkx as nx


class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):   # catches float16, float32, float64
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class EpisodeResults:
    """Container for all metrics from a single episode."""
    episode: int
    mode: str  # 'train' or 'validation'
    seed: int

    # Episode-level scalars
    total_reward: float = 0.0
    mean_daily_failures: float = 0.0
    total_failures: int = 0
    total_invalid_actions: int = 0
    mean_q_values: float = 0.0
    epsilon: float = 0.0

    # Time-series data (per timeslot)
    rewards_per_timeslot: List[float] = field(default_factory=list)
    failures_per_timeslot: List[int] = field(default_factory=list)
    epsilon_per_timeslot: List[float] = field(default_factory=list)
    demand_per_timeslot: List[float] = field(default_factory=list)
    deployed_bikes: List[int] = field(default_factory=list)
    truck_load: List[int] = field(default_factory=list)
    depot_load: List[int] = field(default_factory=list)
    outside_system_bikes: List[int] = field(default_factory=list)
    traveling_bikes: List[int] = field(default_factory=list)
    q_values_per_timeslot: List[np.ndarray] = field(default_factory=list)

    # Action-level data (per step)
    action_per_step: List[int] = field(default_factory=list)
    reward_tracking_per_action: Dict[int, List[float]] = field(default_factory=dict)
    global_critic_scores: List[float] = field(default_factory=list)

    # Spatial data (cell subgraph)
    cell_subgraph: Optional[nx.Graph] = None


def prepare_run_dir(results_path: str | Path, run_id: int, resume: bool = False, overwrite: bool = False,
                    ask=None, interactive: bool | None = None) -> Path:
    """Make ``results_path/run_NNN`` ready for a new training run (issue #178).

    * ``resume``: an existing run is kept as is.
    * The run exists and is not empty: ask ``Overwrite it? [y/N]``. Yes (or ``overwrite``) deletes
      the whole run directory (training, validation, models, checkpoints, local W&B files); anything
      else stops with :class:`SystemExit` and a hint.
    * No terminal to ask in (``interactive`` false, e.g. ``nohup``): stop unless ``overwrite``.
    """
    import sys
    run_dir = Path(results_path) / f"run_{run_id:03d}"
    if resume or not run_dir.exists() or not any(run_dir.iterdir()):
        return run_dir
    if not overwrite:
        episodes = len(list((run_dir / "training").glob("episode_*")))
        parts = sorted(d.name for d in run_dir.iterdir() if d.is_dir())
        summary = f"{episodes} training episodes; " if episodes else ""
        print(f"Run {run_id} already exists: {run_dir} ({summary}{', '.join(parts) or 'files'}).")
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            raise SystemExit(f"Run {run_id} exists and there is no terminal to ask: pass --overwrite to replace it, "
                             f"--resume to continue it, or another --run-id.")
        try:
            answer = (ask or input)(f"Overwrite run {run_id}? Everything in {run_dir} will be deleted. [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() not in ("y", "yes"):
            raise SystemExit(f"Kept run {run_id}. Use another --run-id, or --resume to continue it.")
    shutil.rmtree(run_dir)
    print(f"Deleted {run_dir}.")
    return run_dir


class ResultsManager:
    """Centralized manager for training and validation results."""

    def __init__(
            self,
            results_path: str,
            run_id: int,
            overwrite: bool = False,
            interactive: bool = True,
            val_tag: str | None = None,
            mode: str = 'train',
    ):
        self.results_path = Path(results_path)
        self.run_id = run_id
        self.run_dir = self.results_path / f"run_{run_id:03d}"
        self.training_path = self.run_dir / "training"
        self.validation_path = self.run_dir / "validation"
        self.models_path = self.run_dir / "models"
        self.mode = mode

        # Determine which directories this instance actually owns
        _val_base = self.run_dir / "validation"
        _val_path = _val_base / val_tag if val_tag else _val_base

        if mode == 'train':
            self._owned_dirs = [self.training_path, self.models_path]
        else:
            self._owned_dirs = [_val_path]

        # Check if results already exist
        if self.run_dir.exists() and not overwrite:
            if interactive:
                self._handle_existing_results()
            else:
                # Only raise if the owned directories actually exist
                existing = [d for d in self._owned_dirs if d.exists()]
                if existing:
                    raise Exception(
                        f"Results already exist: {existing}. "
                        f"Use overwrite=True or change run_id."
                    )

        # Create only the directories this instance owns
        for d in self._owned_dirs:
            d.mkdir(parents=True, exist_ok=True)

        self.validation_path = _val_path

        self._active_path = self.training_path if mode == 'train' else self.validation_path

        # Initialize aggregated DataFrames
        self.training_summary = pd.DataFrame()
        self.validation_summary = pd.DataFrame()
        self.best_models = []


    def _handle_existing_results(self):
        """Handle case where results directory already exists — only touches owned subdirs."""
        existing = [d for d in self._owned_dirs if d.exists()]
        if not existing:
            return  # nothing to overwrite, proceed silently

        print(f"⚠️  WARNING: Results already exist for run {self.run_id}.")
        for d in existing:
            print(f"   {d}")
        print("   These directories will be deleted and recreated.")

        try:
            proceed = input("Are you sure you want to continue? (y/n) ").strip().lower()
        except (ValueError, EOFError):
            proceed = ""
            print("Invalid input! Please enter 'y' or 'n'.")

        if proceed in ['y', 'yes']:
            for d in existing:
                print(f"Removing {d}...")
                shutil.rmtree(d)
        else:
            raise Exception(
                f"Aborted: Change the 'run_id' (current: {self.run_id}) "
                f"or the 'results_path' (current: {self.results_path})."
            )


    @classmethod
    def create_with_auto_increment(cls, results_path: str) -> 'ResultsManager':
        """Create a ResultsManager with auto-incremented run_id to avoid conflicts."""
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)

        # Find next available run_id
        existing_runs = [
            int(d.name.split('_')[1])
            for d in results_path.iterdir()
            if d.is_dir() and d.name.startswith('run_')
        ]
        next_run_id = max(existing_runs, default=-1) + 1

        print(f"Auto-assigned run_id: {next_run_id}")
        return cls(str(results_path), next_run_id, overwrite=False)


    @staticmethod
    def build_val_tag(model_type: str, model_episode: int | None = None) -> str:
        """
        Build a human-readable validation subdirectory tag.

        Examples:
            build_val_tag("best")          → "best"
            build_val_tag("final")         → "final"
            build_val_tag("episode", 42)   → "episode_042"
        """
        if model_type == "episode":
            if model_episode is None:
                raise ValueError("model_episode must be provided when model_type='episode'")
            return f"episode_{model_episode:03d}"
        return model_type


    def save_episode(self, results: EpisodeResults):
        """Save results for a single episode."""
        # Validate and fix results
        if results.mode != self.mode:   # Optionally guard against misuse
            raise ValueError(...)
        results = self._validate_episode_results(results)
        episode_dir = self._active_path / f"episode_{results.episode:03d}"
        episode_dir.mkdir(exist_ok=True)

        # 1. Save scalars as JSON (human-readable)
        scalars = {
            'episode': results.episode,
            'seed': results.seed,
            'total_reward': results.total_reward,
            'mean_daily_failures': results.mean_daily_failures,
            'total_failures': results.total_failures,
            'total_invalid_actions': results.total_invalid_actions,
            'epsilon': results.epsilon,
        }
        with open(episode_dir / 'scalars.json', 'w') as f:
            json.dump(scalars, f, indent=2, cls=_NumpyEncoder)

        # 2. Save time-series as CSV (easy analysis)
        timeslot_df = pd.DataFrame({
            'timeslot': range(len(results.rewards_per_timeslot)),
            'reward': results.rewards_per_timeslot,
            'failures': results.failures_per_timeslot,
            'epsilon': results.epsilon_per_timeslot,
            'demand': results.demand_per_timeslot,
            'deployed_bikes': results.deployed_bikes,
            'truck_load': results.truck_load,
            'depot_load': results.depot_load,
            'outside_system_bikes': results.outside_system_bikes,
            'traveling_bikes': results.traveling_bikes,
        })
        timeslot_df.to_csv(episode_dir / 'timeslot_metrics.csv', index=False)

        # 3. Save step-level data as compressed pickle (large data)
        step_data = {
            'actions': results.action_per_step,
            'reward_tracking_per_action': results.reward_tracking_per_action,
            'global_critic_scores': results.global_critic_scores,
            'q_values': results.q_values_per_timeslot,
        }
        with open(episode_dir / 'step_data.pkl.gz', 'wb') as f:
            pickle.dump(step_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. Save cell subgraph separately (spatial data)
        if results.cell_subgraph is not None:
            with open(episode_dir / 'cell_subgraph.gpickle', 'wb') as f:
                pickle.dump(results.cell_subgraph, f)

        # 5. Update aggregated summary
        self._update_summary(results)


    def _update_summary(self, results: EpisodeResults):
        """Append episode summary to aggregated DataFrame."""
        summary_row = {
            'episode': results.episode,
            'total_reward': results.total_reward,
            'mean_daily_failures': results.mean_daily_failures,
            'total_failures': results.total_failures,
            'total_invalid_actions': results.total_invalid_actions,
            'epsilon': results.epsilon,
        }

        if results.mode == 'train':
            self.training_summary = pd.concat([
                self.training_summary,
                pd.DataFrame([summary_row])
            ], ignore_index=True)
        else:
            self.validation_summary = pd.concat([
                self.validation_summary,
                pd.DataFrame([summary_row])
            ], ignore_index=True)


    def save_run_summary(self):
        """Save aggregated summaries for the entire run."""
        summary = self.training_summary if self.mode == 'train' else self.validation_summary
        if not summary.empty:
            summary.to_csv(self._active_path / f'{self.mode}_summary.csv', index=False)


    def save_hyperparameters(self, params: dict, reward_params: dict):
        """Save hyperparameters and configuration."""
        config = {
            'run_id': self.run_id,
            'hyperparameters': params,
            'reward_parameters': reward_params,
        }
        path = self._active_path / 'config.json'
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, cls=_NumpyEncoder)


    def load_episode(self, episode: int) -> EpisodeResults:
        """Load results for a specific episode."""
        episode_dir = self._active_path / f"episode_{episode:03d}"

        # Load scalars
        with open(episode_dir / 'scalars.json', 'r') as f:
            scalars = json.load(f)

        # Load time-series
        timeslot_df = pd.read_csv(episode_dir / 'timeslot_metrics.csv')

        # Load step data
        with open(episode_dir / 'step_data.pkl.gz', 'rb') as f:
            step_data = pickle.load(f)

        # Load cell subgraph if exists
        subgraph_path = episode_dir / 'cell_subgraph.gpickle'
        if subgraph_path.exists():
            with open(episode_dir / 'cell_subgraph.gpickle', 'rb') as f:
                cell_subgraph = pickle.load(f)
        else:
            cell_subgraph = None

        # Reconstruct EpisodeResults
        return EpisodeResults(
            episode=scalars['episode'],
            mode=self.mode,
            seed=scalars['seed'],
            total_reward=scalars['total_reward'],
            mean_daily_failures=scalars['mean_daily_failures'],
            total_failures=scalars['total_failures'],
            total_invalid_actions=scalars['total_invalid_actions'],
            epsilon=scalars['epsilon'],
            rewards_per_timeslot=timeslot_df['reward'].tolist(),
            failures_per_timeslot=timeslot_df['failures'].tolist(),
            epsilon_per_timeslot=timeslot_df['epsilon'].tolist(),
            demand_per_timeslot=timeslot_df['demand'].tolist(),
            deployed_bikes=timeslot_df['deployed_bikes'].tolist(),
            truck_load=timeslot_df['truck_load'].tolist(),
            depot_load=timeslot_df['depot_load'].tolist(),
            outside_system_bikes=timeslot_df['outside_system_bikes'].tolist(),
            traveling_bikes=timeslot_df['traveling_bikes'].tolist(),
            action_per_step=step_data['actions'],
            reward_tracking_per_action=step_data['reward_tracking_per_action'],
            global_critic_scores=step_data['global_critic_scores'],
            q_values_per_timeslot=step_data['q_values'],
            cell_subgraph=cell_subgraph,
        )


    def save_model(self, agent, episode: int, score: float, model_type: str = 'episode'):
        """
        model_type: 'episode' | 'best' | 'final'
        Passing model_type='best' or model_type='final' does NOT also save an episode snapshot.
        If you want both, call this twice or use save_episode_and_maybe_best().
        """
        if self.mode != 'train':
            raise RuntimeError("save_model called on a validation-mode ResultsManager.")

        if model_type == 'episode':
            model_dir = self.models_path / 'episodes' / f'episode_{episode:03d}'
        elif model_type == 'best':
            model_dir = self.models_path / 'best'
        elif model_type == 'final':
            model_dir = self.models_path / 'final'
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model_dir.mkdir(parents=True, exist_ok=True)
        agent.save_model(str(model_dir / 'trained_agent.pt'))

        metadata = {
            'episode': episode,
            'score': score,
            'model_type': model_type,
            'saved_at': pd.Timestamp.now().isoformat()
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

        if model_type == 'best':
            self._update_best_summary(episode, score)

        return model_dir / 'trained_agent.pt'


    def _update_best_summary(self, episode: int, score: float):
        self.best_models.append((episode, score))
        self.best_models.sort(key=lambda x: x[1])
        pd.DataFrame(self.best_models, columns=['episode', 'score']).to_csv(
            self.models_path / 'best_models_summary.csv', index=False
        )


    def promote_episode_to_best(self, episode: int, score: float) -> Path:
        """
        Copy an already-saved episode snapshot to models/best/ without touching
        the live agent.  Used when validation runs in parallel and the decision
        to promote is made one episode after the model was saved.

        Raises FileNotFoundError if the episode snapshot does not exist.
        """
        if self.mode != 'train':
            raise RuntimeError("promote_episode_to_best called on a validation-mode ResultsManager.")

        src = self.models_path / 'episodes' / f'episode_{episode:03d}' / 'trained_agent.pt'
        if not src.exists():
            raise FileNotFoundError(
                f"Episode snapshot not found: {src}. "
                "Make sure save_model(..., model_type='episode') was called before promoting."
            )

        best_dir = self.models_path / 'best'
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(best_dir / 'trained_agent.pt'))

        metadata = {
            'episode': episode,
            'score': score,
            'model_type': 'best',
            'promoted_from': str(src),
            'saved_at': pd.Timestamp.now().isoformat(),
        }
        with open(best_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=_NumpyEncoder)

        self._update_best_summary(episode, score)
        return best_dir / 'trained_agent.pt'


    def get_best_model_path(self) -> Optional[Path]:
        p = self.models_path / 'best' / 'trained_agent.pt'
        return p if p.exists() else None


    def load_model(self, agent, model_type: str = 'best', episode: int = None):
        """
        Load a saved model.

        Args:
            agent: The agent to load weights into
            episode: Specific episode to load (None for best/final)
            model_type: Type of model to load ('checkpoint', 'best', 'final')
        """
        if model_type == 'best' and episode is None:
            model_path = self.get_best_model_path()
            if model_path is None:
                raise FileNotFoundError("No best model found")
        elif model_type == 'final':
            model_path = self.models_path / 'final' / 'trained_agent.pt'
        elif model_type == 'episode' and episode is not None:
            model_path = self.models_path / 'episodes' / f'episode_{episode:03d}' / 'trained_agent.pt'
        else:
            raise ValueError("Invalid combination of model_type and episode")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        agent.load_model(str(model_path))
        return model_path


    @staticmethod
    def _validate_episode_results(results: EpisodeResults) -> EpisodeResults:
        """
        Validate and fix EpisodeResults to ensure data consistency.

        Returns a corrected copy of the results.
        """
        # Determine expected length from rewards (primary metric)
        expected_length = len(results.rewards_per_timeslot)

        if expected_length == 0:
            raise ValueError("Episode results must have at least one timeslot of data")

        # Helper to pad arrays
        def pad_to_length(data, length, fill_value = 0.0):
            if len(data) == length:
                return data
            elif len(data) < length:
                return data + [fill_value] * (length - len(data))
            else:
                return data[:length]

        # Create corrected copy
        return EpisodeResults(
            episode=results.episode,
            mode=results.mode,
            seed=results.seed,
            total_reward=results.total_reward,
            mean_daily_failures=results.mean_daily_failures,
            total_failures=results.total_failures,
            total_invalid_actions=results.total_invalid_actions,
            epsilon=results.epsilon,

            # Ensure consistent lengths
            rewards_per_timeslot=pad_to_length(results.rewards_per_timeslot, expected_length, 0.0),
            failures_per_timeslot=pad_to_length(results.failures_per_timeslot, expected_length, 0),
            epsilon_per_timeslot=pad_to_length(results.epsilon_per_timeslot, expected_length, results.epsilon),
            demand_per_timeslot=pad_to_length(results.demand_per_timeslot, expected_length, 0),
            deployed_bikes=pad_to_length(results.deployed_bikes, expected_length, 0),
            truck_load=pad_to_length(results.truck_load, expected_length, 0),
            depot_load=pad_to_length(results.depot_load, expected_length, 0),
            outside_system_bikes=pad_to_length(results.outside_system_bikes, expected_length, 0),
            traveling_bikes=pad_to_length(results.traveling_bikes, expected_length, 0),
            q_values_per_timeslot=results.q_values_per_timeslot,

            # Step-level data (no length requirements)
            action_per_step=results.action_per_step,
            reward_tracking_per_action=results.reward_tracking_per_action,
            global_critic_scores=results.global_critic_scores,
            cell_subgraph=results.cell_subgraph,
        )