import torch
import numpy as np
import random

from torch.nn import functional as F

from rl_training.networks import DQN

class DQNAgent:
    def __init__(self, num_actions, observation_space_len = 256, replay_buffer =  None, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=500, lr=0.1,
                 device='cpu', tau=0.3, beta=0.01, soft_update=False, seed: int = None):
        """
        Initializes the DQNAgent.
        """
        self.train_model = DQN(num_actions, observation_space_len).to(device)
        self.target_model = DQN(num_actions, observation_space_len).to(device)
        self.target_model.load_state_dict(self.train_model.state_dict())
        self.optimizer = torch.optim.SGD(self.train_model.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.train_model.parameters(), lr=lr, momentum=0.9)
        # self.loss_function = torch.nn.HuberLoss()
        self.loss_function = torch.nn.SmoothL1Loss()
        self.replay_buffer = replay_buffer
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_max = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.device = device
        self.tau = tau
        self.beta = beta
        self.soft_update = soft_update

        if seed is not None:
            self.set_seed(seed)


    @staticmethod
    def set_seed(seed: int) -> None:
        """Seed all RNG sources used by this agent."""
        from rl_training.utils import set_seed
        set_seed(seed)


    def select_action(self, state, epsilon_greedy=False, greedy=False, avoid_action: list = None):
        """
        Selects an action using an epsilon-greedy strategy.

        """
        if avoid_action is None:
            avoid_action = []

        with torch.no_grad():
            # Get Q-values for the state (assumed shape [1, num_actions]) and squeeze to [num_actions]
            q_values = self.get_q_values(state).squeeze(0)

            # Create a boolean mask for allowed actions
            mask = torch.ones_like(q_values, dtype=torch.bool)
            for a in avoid_action:
                mask[a] = False

            if not mask.any():
                raise ValueError("No valid actions available to select.")

            # Mask out invalid actions by setting their Q-value to -infinity
            q_values_masked = q_values.clone()
            q_values_masked[~mask] = -float('inf')

        # Select a random action with probability epsilon
        if epsilon_greedy:
            valid_actions = [action for action in range(self.num_actions) if action not in avoid_action]
            if not valid_actions:
                raise ValueError("No valid actions available to select.")
            if random.random() < self.epsilon:
                return random.choice(valid_actions)
            else:
                return torch.argmax(q_values_masked).item()

        if greedy:
            # Greedy: select the allowed action with the highest Q-value.
            return torch.argmax(q_values_masked).item()
        # else:
        #     # Softmax: compute probabilities from the masked Q-values.
        #     probabilities = torch.softmax(q_values_masked, dim=0)
        #     # Sample from the softmax distribution.
        #     return torch.multinomial(probabilities, 1).item()

        # If cannot find a valid action
        raise ValueError("No valid greedy action could be selected.")


    def get_q_values(self, state, key=None):
        """
        Returns the Q-values of the train model (not the targhet model) for the given state.
        """
        with torch.no_grad():
            return self.train_model(state, key)


    def update_epsilon(self, steps_in_action=1, delta_epsilon = None):
        """
        Updates epsilon for the epsilon-greedy strategy using exponential decay.
        """
        self.steps_done += steps_in_action

        if delta_epsilon is None:
            self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-1.0 * (self.steps_done**2) / self.epsilon_decay)
        else:
            self.epsilon = max(self.epsilon - delta_epsilon, self.epsilon_min)


    def update_target_network(self):
        """
        Updates the target model by copying the weights from the training model.
        """
        self.target_model.load_state_dict(self.train_model.state_dict())

    @staticmethod
    def compute_loss(q_values, expected_q, actions, beta, tau):
        # Gather Q(s, a) for the taken actions using the provided action indices.
        q_action = q_values.gather(1, actions)

        # TD error and MSE loss (detach expected_q to prevent gradients flowing back)
        td_error = q_action - expected_q.detach()
        loss = F.mse_loss(q_action, expected_q.detach())

        # Compute softmax probabilities for the full Q-value distribution.
        probs = F.softmax(q_values / tau, dim=-1)

        # Compute entropy: -sum(p(a|s)*log(p(a|s))) averaged over samples.
        entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        entropy_term = beta * entropy_loss.mean()

        # Final loss: standard loss minus the entropy bonus.
        total_loss = loss - entropy_term

        return total_loss, td_error


    @staticmethod
    def action_return(step_rewards, gamma: float) -> float:
        """Discounted reward of an action that lasted ``len(step_rewards)`` steps:
        ``R = sum_i gamma^i r_i`` (semi-MDP, issue #164)."""
        r = 0.0
        for i, x in enumerate(step_rewards):
            r += (gamma ** i) * float(x)
        return r

    @staticmethod
    def smdp_target(reward, steps, terminal, next_q, gamma: float):
        """Semi-MDP Q-learning target ``y = R + gamma^tau * Q(s', a*) * (1 - terminal)``.

        ``steps`` is the duration ``tau`` of the action taken in ``s`` (not of the one that led to
        ``s``), ``terminal`` is true only for real terminal states: a time-limit truncation still
        bootstraps (issue #164)."""
        return reward + (gamma ** steps.float()) * next_q * (1.0 - terminal.float())

    def soft_update_target_network(self, tau=0.005):
        for target_param, train_param in zip(self.target_model.parameters(), self.train_model.parameters()):
            target_param.data.copy_(tau * train_param.data + (1 - tau) * target_param.data)


    def train_step(self, batch_size) -> float:
        """
        Performs a single training step using a batch sampled from the replay buffer.
        """
        if len(self.replay_buffer) < batch_size:
            return 0

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(batch_size).to(self.device)

        # Compute the train values Q(S, a)
        train_q_values = self.train_model(batch, 's').gather(1, batch.actions)

        with torch.no_grad():
            # Compute the target values of next step Q(S', pi(S'))
            next_actions = self.train_model(batch, 't').argmax(dim=1, keepdim=True)
            next_q_values = self.target_model(batch, 't').gather(1, next_actions)

            # Semi-MDP Bellman target: discount by the duration of the action taken, bootstrap
            # unless the next state is truly terminal (time-limit truncation still bootstraps)
            target_q_values = self.smdp_target(batch.reward, batch.steps, batch.terminal, next_q_values, self.gamma)

        # Compute loss
        loss = F.smooth_l1_loss(train_q_values, target_q_values)
        # loss, _ = self.compute_loss(
        #     train_q_values,
        #     target_q_values,
        #     batch.actions,
        #     beta=self.beta,
        #     tau=self.tau
        # )

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), max_norm=10.0)
        self.optimizer.step()

        # self.soft_update_target_network(tau=self.tau)

        return loss.item()


    def save_model(self, file_path):
        """
        Save the model to a file.
        """
        torch.save({
            'train_model': self.train_model.state_dict(),
            'target_model': self.target_model.state_dict(),
        }, file_path)


    def load_model(self, file_path):
        """
        Load the model from a file.
        """
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and 'train_model' in checkpoint:
            self.train_model.load_state_dict(checkpoint['train_model'])
            self.target_model.load_state_dict(checkpoint['target_model'])
        else:
            # backwards compatibility with old single-state-dict saves
            self.train_model.load_state_dict(checkpoint)
            self.target_model.load_state_dict(checkpoint)
