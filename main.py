import copy
from collections import deque, namedtuple

import gym
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pyvirtualdisplay
import torch
import torch.nn as nn
import torch.optim as optim

from model import Combiner_network, GVF_network, soft_update
from utils import choose_action, concat_state_action, exp_sample, get_feature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
env.seed(seed)

state = env.reset()
done = False

_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

n_action = env.action_space.n
n_state = len(env.reset())
n_input = n_state + n_action
n_feature = 8
n_output = 1

memory = deque(maxlen=200000)

lr_f = 1e-5
lr_c = 1e-5

reward_discount_factor = 0.99
GVF_discount_factor = 0.99

eps_start = 1.0
eps_decrease_steps = int(2e5)
eps_end = 0.05

batch_size = 128
tau = 5e-4

n_episode = 5000
n_timestep = env._max_episode_steps

freq_target_update = 1
freq_evaluation = 100
freq_update_network = 5

F_network = GVF_network(input_len=n_input, ouput_len=n_feature).to(device)
F_opt = optim.Adam(F_network.parameters(), lr=lr_f)
F_aux = GVF_network(input_len=n_input, ouput_len=n_feature).to(device)

C_network = Combiner_network(input_len=n_feature, ouput_len=n_output).to(device)
C_opt = optim.SGD(C_network.parameters(), lr=lr_c)
C_aux = Combiner_network(input_len=n_feature, ouput_len=n_output).to(device)

experience_t = namedtuple(
    "Experience",
    field_names=["state_action", "reward", "feature", "next_state", "done"],
)

F_net_name = "F_net"
C_net_name = "C_net"

video_name = "video.mp4"


def train():
    state_actions, rewards, features, next_states, dones = exp_sample(
        memory, batch_size, device
    )

    idx = torch.arange(batch_size, dtype=torch.long).to(device)

    feature_vector = F_network(state_actions)
    q_values = C_network(feature_vector)

    next_state_actions = concat_state_action(next_states, n_action, n_state, device)
    feature_vector_next = F_aux(next_state_actions)
    q_values_next = C_aux(feature_vector_next)
    feature_vector_next = feature_vector_next.view((-1, n_action, n_feature))
    q_values_next = q_values_next.view((-1, n_action))

    q_target, max_idx = q_values_next.max(1)
    q_target = (1 - dones.squeeze(1)) * q_target
    q_target = rewards.squeeze(1) + reward_discount_factor * q_target
    q_target = q_target.unsqueeze(1)

    feature_vector_target = feature_vector_next[idx, max_idx, :]
    feature_vector_target = (1 - dones.view(-1, 1)) * feature_vector_target
    feature_vector_target = features + feature_vector_target * GVF_discount_factor

    criterion = nn.MSELoss()

    loss_C = criterion(q_values, q_target)
    loss_F = criterion(feature_vector, feature_vector_target)

    C_opt.zero_grad()
    loss_C.backward(retain_graph=True)
    C_opt.step()

    F_opt.zero_grad()
    loss_F.backward()
    F_opt.step()


def evaluation():
    total_reward = 0
    total_GVF_loss = 0

    n_trial = 100

    for _ in range(n_trial):
        state = env.reset()
        done = False

        gt_feature = torch.zeros(n_timestep, n_feature)
        pred_feature = torch.zeros(n_timestep, n_feature)
        discounted_para = torch.zeros(n_timestep, 1)

        step = 0
        for t in range(1, n_timestep + 1):
            step += 1
            action, pred_feature_vector = choose_action(
                state, F_network, C_network, n_action, n_state, 0.0, device
            )

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            gt_feature_vector = torch.tensor(get_feature(state))

            pred_feature[step - 1] = pred_feature_vector

            gt_feature[:step] += (
                gt_feature_vector * (GVF_discount_factor ** discounted_para)
            )[:step]

            discounted_para[:step] += 1
            state = next_state

            if done:
                break

        with torch.no_grad():
            criterion = nn.MSELoss()
            total_GVF_loss += criterion(gt_feature[:step], pred_feature[:step]).item()

    avg_reward = total_reward / n_trial
    avg_GVF_loss = total_GVF_loss / n_trial


def main():
    eps = eps_start

    for i_episode in range(1, n_episode + 1):
        state = env.reset()
        done = False
        step = 0

        for t in range(1, n_timestep + 1):
            step += 1
            action, _ = choose_action(
                state, F_network, C_network, n_action, n_state, eps, device
            )
            next_state, reward, done, _ = env.step(action)
            feature = get_feature(state)
            action_vector = np.zeros(n_action)
            action_vector[action] = 1.0
            experience = experience_t(
                np.concatenate([state, action_vector]),
                reward,
                feature,
                next_state,
                done and t < n_timestep,
            )
            state = next_state
            memory.append(experience)

            if step % freq_update_network == 0 and len(memory) > batch_size:
                train()

            if t % freq_target_update == 0:
                soft_update(F_network, F_aux, tau)
                soft_update(C_network, C_aux, tau)

            eps = max(eps_end, eps - (eps_start - eps_end) / (eps_decrease_steps - 1))

            if done:
                break

        if i_episode % freq_evaluation == 0:
            evaluation()

    torch.save(F_network.state_dict(), F_net_name)
    torch.save(C_network.state_dict(), C_net_name)


def calc_ig(feature1, feature2):
    model = copy.deepcopy(C_network).to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr_c)
    y_baseline = model(feature2).item()
    x = feature1.view(1, -1)
    x.size()[1]

    intergated_grad = torch.zeros_like(x)

    n_iteration = 30

    for i in range(n_iteration):
        new_input = feature2 + ((i + 1) / n_iteration * (x - feature2))
        new_input = new_input.clone().detach().requires_grad_(True)

        y = model(new_input)
        loss = abs(y_baseline - y)
        opt.zero_grad()
        loss.backward()

        intergated_grad += (new_input.grad) / n_iteration

    intergated_grad *= x - feature2

    ig = np.array(intergated_grad[0].tolist())
    return ig


def explain():
    F_network.load_state_dict(torch.load(F_net_name))
    C_network.load_state_dict(torch.load(C_net_name))

    state = env.reset()
    done = False

    spec = gridspec.GridSpec(ncols=2, nrows=3, height_ratios=[3, 1, 2])
    fig = plt.figure()
    ax1 = fig.add_subplot(spec[0, :])
    ax1.axis("off")
    ax2 = fig.add_subplot(spec[1, 0])
    ax2.set_ylim(0, 500)
    ax2.set_title("Left")
    ax3 = fig.add_subplot(spec[1, 1])
    ax3.set_ylim(0, 500)
    ax3.set_title("Right")
    ax4 = fig.add_subplot(spec[2, :])
    ims = []

    for t in range(1, n_timestep + 1):
        action, _ = choose_action(
            state, F_network, C_network, n_action, n_state, 0.0, device
        )
        next_state, reward, done, _ = env.step(action)

        state_action = concat_state_action(
            torch.tensor(state).unsqueeze(0), n_action, n_state, device
        ).to(torch.float)

        feature = F_network(state_action)

        ig = calc_ig(feature[0], feature[1])
        feature = feature.to("cpu").detach().numpy()

        rend = env.render(mode="rgb_array")
        im = [ax1.imshow(rend)]
        im += ax2.bar(x=range(len(feature[0])), height=feature[0], color="b")
        im += ax3.bar(x=range(len(feature[1])), height=feature[1], color="r")
        im += ax4.bar(x=range(len(ig)), height=ig, color="g")
        ims.append(im)

        state = next_state

        if done:
            break

    ani = animation.ArtistAnimation(
        fig, ims, interval=100, blit=True, repeat_delay=1000
    )

    ani.save(video_name)


if __name__ == "__main__":
    main()
    explain()
