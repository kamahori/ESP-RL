import random

import numpy as np
import torch


def choose_action(state, F_network, C_network, action_space, state_len, eps, device):
    """Choose action from sub q networks epsilon-greedily
    returns: argmax_a \Sum_i Q_i(state, a)
    """

    if np.random.random() < eps:
        return int(torch.tensor(np.random.randint(0, action_space))), None

    state_ft = torch.tensor(state).view(-1, state_len).to(device)
    com_state = state_ft.repeat((1, action_space)).view((-1, state_len)).to(device)
    action = torch.zeros((action_space, action_space)).to(device)
    for i in range(action_space):
        action[i][i] = 1.0
    state_action = torch.cat((com_state, action), 1).to(torch.float32).to(device)
    # [[s1, s2, s3, s4, 1, 0], [s1, s2, s3, s4, 0, 1]]

    feature_vector = F_network(state_action)
    q_values = C_network(feature_vector)
    best_action = q_values.max(0)[1].item()
    feature_vector = feature_vector.view(action_space, -1)
    return best_action, feature_vector[best_action]


def get_feature(state):
    pos_th = 1
    cart_vel_th = 1
    angle_th = 0.07
    pole_vel_th = 0.7

    res = np.ones(8)

    (
        pos,
        cart_vel,
        angle,
        pole_vel,
    ) = state

    if pos_th < pos:
        res[0] = -1
    if -pos_th > pos:
        res[1] = -1

    if cart_vel_th < cart_vel:
        res[2] = -1
    if -cart_vel_th > cart_vel:
        res[3] = -1

    if angle_th < angle:
        res[4] = -1
    if -angle_th > angle:
        res[5] = -1

    if pole_vel_th < pole_vel:
        res[6] = -1
    if -pole_vel_th > pole_vel:
        res[7] = -1

    return res


def concat_state_action(states, n_action, n_state, device):
    action_vector = torch.zeros((n_action, n_action))
    for i in range(len(action_vector)):
        action_vector[i, i] = 1.0

    com_state = states.repeat((1, n_action)).view((-1, n_state)).to(device)
    actions = action_vector.repeat((len(states), 1)).to(device)

    state_action = torch.cat((com_state, actions), 1)
    return state_action


def exp_sample(memory, batch_size, device):
    exps = random.sample(memory, k=batch_size)
    state_actions = (
        torch.from_numpy(np.vstack([e.state_action for e in exps if e is not None]))
        .float()
        .to(device)
    )
    rewards = (
        torch.from_numpy(np.vstack([e.reward for e in exps if e is not None]))
        .float()
        .to(device)
    )
    features = (
        torch.from_numpy(np.vstack([e.feature for e in exps if e is not None]))
        .float()
        .to(device)
    )
    next_states = (
        torch.from_numpy(np.vstack([e.next_state for e in exps if e is not None]))
        .float()
        .to(device)
    )
    dones = (
        torch.from_numpy(np.vstack([e.done for e in exps if e is not None]))
        .float()
        .to(device)
    )

    return state_actions, rewards, features, next_states, dones
