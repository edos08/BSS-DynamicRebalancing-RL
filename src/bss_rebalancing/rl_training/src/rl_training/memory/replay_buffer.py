import torch
import numpy as np
import os
import pickle

from torch_geometric.data import Data, Batch

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        """
        Custom increment logic for edge indices to handle source (`_s`) and target (`_t`) data.

        Parameters:
            - key: Key of the attribute.
            - value: Value of the attribute.
            - args: Additional arguments.
            - kwargs: Additional keyword arguments.
        """
        if key == 'edge_index_s':
            return self.x_s.size(0)
        elif key == 'edge_index_t':
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        """
        Replay buffer for storing transitions with support for PairData objects.

        Parameters:
            - max_size: Maximum number of transitions to store.
            - device: Device to store the transitions.
        """
        self.buffer = []
        self.buffer_size = max_size

    def push(self, state, action, reward, next_state, done, steps=None, terminal=None):
        """
        Adds a transition to the buffer (semi-MDP: an action can last several steps, issue #164).

        Parameters:
            - state (PairData): Current state.
            - action (int): Action taken.
            - reward (float): Discounted reward collected **during this action**,
              ``R = sum_{i < steps} gamma^i r_i`` (see ``DQNAgent.action_return``).
            - next_state (PairData): State when the action ended.
            - done (bool): Whether the episode is over (terminated or truncated), kept for logging.
            - steps (int): Duration of **this** action in environment steps (``tau``); the target
              discounts the next state by ``gamma ** steps``. Required.
            - terminal (bool): Whether ``next_state`` is a true terminal state (no bootstrap).
              Defaults to ``False``: an episode that ends on a time limit is truncated, and its
              target still bootstraps.
        """
        if steps is None:
            raise ValueError("push() needs the duration of the action taken (steps); see issue #164")
        if int(steps) < 1:
            raise ValueError("an action lasts at least one step")
        # Create a PairData transition
        transition = PairData(
            x_s=state.x,
            edge_index_s=state.edge_index,
            edge_attr_s=state.edge_attr,

            x_t=next_state.x,
            edge_index_t=next_state.edge_index,
            edge_attr_t=next_state.edge_attr
        )

        # Add additional attributes to the transition
        transition.agent_state = torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(dim=0)
        transition.actions = torch.tensor([action]).unsqueeze(dim=1)
        transition.reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(dim=1)
        transition.agent_next_state = torch.tensor(next_state.agent_state, dtype=torch.float32).unsqueeze(dim=0)
        transition.done = torch.tensor([bool(done)]).unsqueeze(dim=1)
        transition.terminal = torch.tensor([bool(terminal) if terminal is not None else False]).unsqueeze(dim=1)
        transition.steps = torch.tensor([int(steps)]).unsqueeze(dim=1)

        # Maintain buffer size by removing oldest transition if full
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer and moves them to the device.

        Parameters:
            - batch_size (int): Number of transitions to sample.

        Returns:
            - List of transitions sampled from the buffer (on the specified device).
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch_list = [self.buffer[i] for i in indices]
        batch = Batch.from_data_list(batch_list, follow_batch=['x_s', 'x_t'])

        return batch

    def __len__(self):
        """
        Returns the current size of the buffer.

        Returns:
            - Size of the buffer.
        """
        return len(self.buffer)

    def save_to_files(self, folder_path, chunk_size=10000):
        """
        Saves the replay buffer to multiple files in chunks.

        Parameters:
            - folder_path: Directory to save the buffer files.
            - chunk_size: Number of transitions per file.
        """
        os.makedirs(folder_path, exist_ok=True)
        num_chunks = (len(self.buffer) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            chunk = self.buffer[i * chunk_size : (i + 1) * chunk_size]
            file_path = os.path.join(folder_path, f"buffer_chunk_{i}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(chunk, f)

    def load_from_files(self, folder_path):
        """
        Loads the replay buffer from files in a folder.

        Parameters:
            - folder_path: Directory containing the buffer files.
        """
        from tqdm import tqdm
        self.buffer = []
        buffer_files = [f for f in os.listdir(folder_path)
                        if f.startswith("buffer_chunk_") and f.endswith(".pkl")]
        buffer_files = sorted(buffer_files)

        tbar = tqdm(total=len(buffer_files), desc="Loading buffer files")
        for file_name in buffer_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "rb") as f:
                    chunk = pickle.load(f)
                    self.buffer.extend(chunk)
            except EOFError:
                print(f"Warning: Skipping file {file_name} due to EOFError.")
            except Exception as e:
                print(f"Error: Failed to load file {file_name} due to {e}.")
            tbar.update(1)
