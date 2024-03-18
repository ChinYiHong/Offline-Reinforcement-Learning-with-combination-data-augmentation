# Adapted from the Reinforcement Learning (DQN) Tutorial by
# [Adam Paszke] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# reccomended to run on google collab (very computational heavy or just reference results.ipynb for output)
import gym
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import random
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time
from scipy.stats import mannwhitneyu
from tabulate import tabulate
import itertools
import math

warnings.filterwarnings("ignore", category=UserWarning)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA makes rewards from the uncertain far future less
# important for our agent than the ones in the near future that it can be fairly confident about
# LR is the learning rate of the ``AdamW`` optimizer
# NUM_BATCH IS number of batch retrieve for training
# TAU is the update rate of the target network
BATCH_SIZE = 256
GAMMA = 0.99
LR = 0.0001
NUM_BATCH = 200
TAU = 0.0005
# files
CSV_FILE_BASE = "/content/drive/MyDrive/data/cartpole_data.csv"
CSV_FILE_ADV = "/content/drive/MyDrive/data/augmented_data_adv.csv"
CSV_FILE_GAU = "/content/drive/MyDrive/data/augmented_data_gaussian.csv"
CSV_FILE_UNI = "/content/drive/MyDrive/data/augmented_data_uniform.csv"
CSV_STACK_ADV = "/content/drive/MyDrive/data/combined_adv.csv"
CSV_STACK_GAU = "/content/drive/MyDrive/data/combined_gauss.csv"
CSV_STACK_UNI = "/content/drive/MyDrive/data/combined_uniform.csv"
CSV_STACK_UNI_ADV = "/content/drive/MyDrive/data/combined_adv_uniform.csv"
CSV_STACK_GAU_ADV = "/content/drive/MyDrive/data/combined_adv_gauss.csv"
CSV_COM_UNI_GAU = "/content/drive/MyDrive/data/add_gauss_uniform.csv"
CSV_COM_UNI_ADV = "/content/drive/MyDrive/data/add_adv_gauss.csv"
CSV_COM_GAU_ADV = "/content/drive/MyDrive/data/add_adv_uniform.csv"
CSV_COM_ALL = "/content/drive/MyDrive/data/add_all.csv"

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward"])
# Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBufferDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(
            csv_file,
            dtype={
                "state0": float,
                "state1": float,
                "state2": float,
                "state3": float,
                "Action": int,
                "Reward": int,
                "Terminated": bool,
                "Truncated": bool,
            },
        )
        self.transitions = []

        for i in range(len(self.df)):
            state = self.df.loc[i, ["state0", "state1", "state2", "state3"]].values
            a = self.df.loc[i, "Action"]
            r = self.df.loc[i, "Reward"]
            t = self.df.loc[i, "Terminated"]
            tr = self.df.loc[i, "Truncated"]

            if t or tr:
                # Last step, termination/truncation = true
                next_state = [0.0, 0.0, 0.0, 0.0]  # Placeholder for last step
            else:
                next_state = self.df.loc[
                    i + 1, ["state0", "state1", "state2", "state3"]
                ].values

            self.transitions.append(Transition(state, a, next_state, r))

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        transition = self.transitions[idx]
        return Transition(
            transition.state,
            transition.action,
            transition.next_state,
            transition.reward,
        )


# Replay buffer
class ReplayBuffer:
    def __init__(self, dataset):
        self.buffer = dataset.transitions
        self.idx = 0

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, BATCH_SIZE, bias=True)
        self.layer3 = nn.Linear(BATCH_SIZE, BATCH_SIZE, bias=True)  # the representation layer
        self.layer4 = nn.Linear(BATCH_SIZE, n_actions, bias=True)  # the prediction layer
        '''
        # xavier_uniform Weight Initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        '''
      # during optimization. Returns prediction
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


# function for printing results
def print_results_table(condition, label, values):
    table_data = [
        ["Mean", np.mean(values)],
        ["Median", np.median(values)],
        ["Standard Deviation", np.std(values)],
        ["Minimum", np.min(values)],
        ["Maximum", np.max(values)],
    ]
    print(f"{label} for {condition}:")
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))

# function for training dqn
def train_dqn(dqn, target_net, memory, optimizer, device):
    start_time = time.time()
    q_values_list = []  # Store Q-values
    loss_list = []  # Store loss
    for transition in range(NUM_BATCH):
        # Sample batch from buffer
        batch = memory.sample(BATCH_SIZE)

        # Unpack batch
        states, actions, rewards, next_states = unpack_batch(batch)

        # Apply CUDA
        states, actions, rewards, next_states = (
            states.to(device),
            actions.to(device),
            rewards.to(device),
            next_states.to(device),
        )

        # Get Q values for current states
        q_values = dqn(states).gather(1, actions.view(-1, 1))

        # Calculate target Q values
        if next_states.size(0) > 0:
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + GAMMA * max_next_q.detach()


        # Calculate loss
        '''
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_values, target_q)
        '''
        # CQL Loss
        batch_size = BATCH_SIZE
        q_value_estimation_actions = dqn(states).max(1)[1].view(-1, 1)
        next_q_values = dqn(next_states).detach()
        next_q_values_selected = next_q_values.gather(1, q_value_estimation_actions)
        cql_temperature = 1
        cql_weight = 0.1
        loss = F.mse_loss(q_values, target_q) + cql_weight * F.mse_loss(cql_temperature * q_values, next_q_values_selected)

        # Optimize
        optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(dqn.parameters(), 100)

        optimizer.step()
        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = dqn.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        # Store Q-values and loss
        q_values_list.append(q_values.mean().item())
        loss_list.append(loss.item())
    end_time = time.time()
    elapsed_time = end_time - start_time
    return elapsed_time,q_values_list,loss_list

# function to extract batch from the buffers
def unpack_batch(batch):
    states = torch.cat(
        [
            torch.tensor(np.array(s).astype(np.float32)).float().unsqueeze(0)
            for s in batch.state
        ]
    )
    actions = torch.cat([torch.tensor(a).long().unsqueeze(0) for a in batch.action])
    rewards = torch.cat([torch.tensor(r).float().unsqueeze(0) for r in batch.reward])
    next_states = torch.cat(
        [
            torch.tensor(np.array(s).astype(np.float32)).float().unsqueeze(0)
            for s in batch.next_state
        ]
    )
    return states, actions, rewards, next_states

# function to evaluate results of testing phase
def evaluate_dqn(env, state, policy_net, target_net, device):
    episode_count = 0
    total_reward = 0
    solved = False

    while not solved and episode_count < 600:
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_steps = 0
        while True:
            action = policy_net(state).max(1).indices.view(1, 1)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            total_reward += reward
            episode_steps += 1

            if terminated or truncated:
                break
            state = next_state

        if truncated:
            solved = True

        episode_count += 1

    total_reward = total_reward / episode_count
    return episode_count, total_reward

# function to reset weights of dqn
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Reset weights for convolutional and linear layers
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# main function for experiment
def run_experiment(num_trials, condition_name, csv_file, seed):
    print(f"Running experiment for condition: {condition_name}")
    print("Extracting from dataset....")
    dataset = ReplayBufferDataset(csv_file)
    print("Creating memory buffer....")
    memory = ReplayBuffer(dataset)
    # Initialize
    env = gym.make("CartPole-v1", new_step_api=True)
    # Get number of actions from gym action space
    action_dim = env.action_space.n
    # Get the number of state observations
    state = env.reset()
    state_dim = len(state)


    # initialize arrays to store results
    condition_episodes = []
    condition_rewards = []
    condition_times = []
    condition_q_values = []
    condition_losses = []
    for x in range(num_trials):
        # Set the seed for reproducibility within the trial
        new_seed = seed + x
        env.reset(seed=new_seed)
        np.random.seed(new_seed)
        torch.manual_seed(new_seed)
        random.seed(new_seed)
        # Initialize DQN, target network, and optimizer
        dqn = DQN(state_dim, action_dim).to(device)
        target_net = DQN(state_dim, action_dim).to(device)
        target_net.load_state_dict(dqn.state_dict())
        optimizer = optim.AdamW(dqn.parameters(), lr=LR, amsgrad=True)

        # Train the DQN
        print("Training DQN: ", condition_name)
        train_time,q_values,losses = train_dqn(dqn, target_net, memory, optimizer, device)
        condition_times.append(train_time)
        condition_q_values.append(q_values)
        condition_losses.append(losses)
        state = env.reset()
        # Evaluate the DQN
        episodes, rewards = evaluate_dqn(env, state, dqn, target_net, device)
        print(
            f"Solved in {episodes} episodes for condition: {condition_name}, trial: {x}"
        )
        condition_episodes.append(episodes)
        condition_rewards.append(rewards)
        print('rewards: ', rewards)
        print('time: ', train_time)
        # reset state for gym, weights for dqn
        state = env.reset()
        dqn.apply(weight_reset)
        target_net.apply(weight_reset)

    # visualize q values and loss
    visualize_Q_results(condition_q_values, condition_losses, condition_name)


    return condition_episodes, condition_rewards, condition_times


def perform_mannwhitneyu_test(
    condition1, result1, condition2, result2, alpha, num_comparisons
):
    # Mann-Whitney U Test for Rewards
    result = mannwhitneyu(result1["mean"], result2["mean"], alternative="greater")

    print(f"\nMann-Whitney U Test for {condition1} vs {condition2}:")
    print(f"U statistic: {result.statistic}")
    print(f"p-value: {result.pvalue}")

    # Bonferroni Correction
    adjusted_alpha = alpha / num_comparisons

    # Decide if statistically significant after Bonferroni Correction
    if result.pvalue < adjusted_alpha:
        print(
            f"The mean reward for {condition1} is statistically significantly greater than the mean reward for {condition2} after Bonferroni Correction (alpha={adjusted_alpha})."
        )
    else:
        print(
            f"The mean reward for {condition1} is NOT statistically significantly greater than the mean reward for {condition2} after Bonferroni Correction (alpha={adjusted_alpha})."
        )

def main():
    # setting up different conditions for testing
    '''
        "Baseline": CSV_FILE_BASE,
        "Gaussian": CSV_STACK_GAU,
        "Adv": CSV_STACK_ADV,
        "Uniform": CSV_STACK_UNI,
        "AdvUniform": CSV_STACK_UNI_ADV,
        "AdvGaussian": CSV_STACK_GAU_ADV,
        "COMB_GaussianUniform": CSV_COM_UNI_GAU,
        "COMB_AdvUniform": CSV_COM_UNI_ADV,
        "COMB_AdvGaussian": CSV_COM_GAU_ADV,
        "COMB_ALL": CSV_COM_ALL,
    '''
    conditions = {
        "Baseline": CSV_FILE_BASE,
        "Gaussian": CSV_STACK_GAU,
        "Adv": CSV_STACK_ADV,
        "Uniform": CSV_STACK_UNI,
        "AdvUni": CSV_STACK_UNI_ADV,
        "AdvGaus": CSV_STACK_GAU_ADV,
        "C_GausUni": CSV_COM_UNI_GAU,
        "C_AdvsUni": CSV_COM_UNI_ADV,
        "C_AdvGaus": CSV_COM_GAU_ADV,
        "C_ALL": CSV_COM_ALL,
    }
    print(conditions)
    num_trials = 5
    #776 is seed used
    #seed = np.random.randint(10000)  # Set a random seed for reproducibility across different conditions
    seed = 776
    print('seed: ', seed)
    episode_results = {}
    episode_rewards = {}
    episode_times = {}

    for condition, csv_file in conditions.items():

        condition_episodes, condition_rewards, condition_times = run_experiment(
            num_trials, condition, csv_file, seed
        )

        episode_results[condition] = {
            "mean": np.mean(condition_episodes, axis=0),
            "std": np.std(condition_episodes, axis=0),
        }

        episode_rewards[condition] = {
            "mean": np.mean(condition_rewards, axis=0),
            "std": np.std(condition_rewards, axis=0),
        }

        episode_times[condition] = {
            "mean": np.mean(condition_times),
            "std": np.std(condition_times),
        }

        print_results_table(condition, "Episodes", condition_episodes)
        print_results_table(condition, "Rewards", condition_rewards)
        print_results_table(condition, "Time", condition_times)


    # Statistical test
    num_comparisons = len(episode_rewards)
    alpha = 0.05


    for condition1, result1 in episode_rewards.items():
        for condition2, result2 in episode_rewards.items():
            # Avoid redundant comparisons and ensure each pair is tested only once
            if condition1 != condition2:
                # Perform Mann-Whitney U test
                perform_mannwhitneyu_test(condition1, result1, condition2, result2, alpha, num_comparisons)



    # Visualize results
    visualize_results(episode_results, "Episodes")
    visualize_results(episode_rewards, "Rewards")
    visualize_results(episode_times, "Time")



def visualize_results(results_dict, ylabel):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(results_dict)))
    plt.figure(figsize=(15, 8))

    conditions = list(results_dict.keys())
    mean_values = [result["mean"] for result in results_dict.values()]

    plt.bar(conditions, mean_values, color=colors, alpha=0.7)

    plt.title(f"Mean {ylabel} for Different Conditions")
    plt.xlabel("Conditions")
    plt.ylabel(f"Mean {ylabel}")
    plt.show()

def visualize_Q_results(q_values_list_per_condition, losses_list_per_condition, condition_name):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for i, q_values_list in enumerate(q_values_list_per_condition):
        plt.plot(q_values_list, label=f'Trial {i + 1}')
    plt.title(f'Q-value Growth Over Iterations - {condition_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Q-value')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, losses_list in enumerate(losses_list_per_condition):
        plt.plot(losses_list, label=f'Trial {i + 1}')
    plt.title(f'Loss Over Iterations - {condition_name}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

