"""The custom training loop from rl_training/README.md, run for real on synthetic data."""
import gymnasium as gym
import torch
from torch_geometric.data import Data

import gymnasium_env  # noqa: F401
from rl_training import (
    DQNAgent, ReplayBuffer, set_seed,
    build_cell_graph_from_cells, update_cell_graph_features, convert_graph_to_data,
)


def test_readme_custom_training_loop(data_path, tmp_path):
    set_seed(42)
    env = gym.make("gymnasium_env/FullyDynamicEnv-v0", data_path=data_path, results_path=str(tmp_path))
    agent = DQNAgent(
        num_actions=env.action_space.n,
        observation_space_len=env.observation_space.shape[0],
        replay_buffer=ReplayBuffer(100_000),
        gamma=0.95, lr=1e-4, device="cpu",
    )

    FEATURES = ["truck_cell", "critic_score", "eligibility_score", "total_bikes"]
    obs, info = env.reset(options={"total_timeslots": 2, "maximum_number_of_bikes": 40, "depot_id": 18})
    graph = build_cell_graph_from_cells(info["cell_dict"], info["nodes_dict"], info["distance_lookup"])
    state = convert_graph_to_data(graph, node_features=FEATURES)
    state.agent_state, state.steps = obs, info["steps"]

    losses, done = [], False
    while not done:
        single = Data(
            x=state.x, edge_index=state.edge_index, edge_attr=state.edge_attr,
            agent_state=torch.tensor(state.agent_state, dtype=torch.float32).unsqueeze(0),
            batch=torch.zeros(state.x.size(0), dtype=torch.long),
        )
        action = agent.select_action(single, epsilon_greedy=True)
        obs, reward, done, timeslot_done, info = env.step(action)

        update_cell_graph_features(graph, info["cell_dict"])
        next_state = convert_graph_to_data(graph, node_features=FEATURES)
        next_state.agent_state, next_state.steps = obs, info["steps"]

        agent.replay_buffer.push(state, action, reward, next_state, done, steps=info["steps"], terminal=False)
        assert agent.replay_buffer.buffer[-1].steps.item() == info["steps"]    # this action's duration
        losses.append(agent.train_step(batch_size=64))
        if timeslot_done:
            agent.update_epsilon()
        state = next_state
    env.close()

    assert len(agent.replay_buffer) > 64
    assert any(l > 0 for l in losses)          # training actually happened
    assert agent.epsilon < 1.0

    # greedy action + masking on the last state
    a = agent.select_action(single, greedy=True, avoid_action=[0])
    assert a != 0

    # save / load round trip
    path = tmp_path / "agent.pt"
    agent.save_model(path)
    agent2 = DQNAgent(num_actions=8, observation_space_len=obs.shape[0], device="cpu")
    agent2.load_model(path)
    q1, q2 = agent.get_q_values(single), agent2.get_q_values(single)
    assert torch.allclose(q1, q2)


def test_smdp_target_and_action_return():
    import pytest

    g = 0.9
    assert DQNAgent.action_return([1.0, 1.0, 1.0], g) == pytest.approx(1 + 0.9 + 0.81)
    assert DQNAgent.action_return([], g) == 0.0
    reward = torch.tensor([[1.0], [1.0], [1.0]])
    steps = torch.tensor([[1], [3], [3]])
    terminal = torch.tensor([[False], [False], [True]])
    next_q = torch.tensor([[10.0], [10.0], [10.0]])
    y = DQNAgent.smdp_target(reward, steps, terminal, next_q, g)
    assert torch.allclose(y, torch.tensor([[1 + 0.9 * 10], [1 + 0.729 * 10], [1.0]]))


def test_push_requires_the_action_duration():
    import pytest

    s = Data(x=torch.zeros(2, 4), edge_index=torch.zeros(2, 0, dtype=torch.long), edge_attr=torch.zeros(0, 1))
    s.agent_state = [0.0, 1.0]
    buf = ReplayBuffer(10)
    with pytest.raises(ValueError, match="duration"):
        buf.push(s, 0, 1.0, s, False)
    with pytest.raises(ValueError):
        buf.push(s, 0, 1.0, s, False, steps=0)
    buf.push(s, 0, 1.0, s, True, steps=4)
    t = buf.buffer[-1]
    assert t.steps.item() == 4 and t.done.item() and not t.terminal.item()
