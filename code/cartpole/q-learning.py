# Adapted from: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947
# file to create baseline file
import gymnasium as gym
import numpy as np
import random
import math
import time
import pandas as pd

# from time import sleep
import numpy as np
import random
import math
from time import sleep

# Lists to store data
observation_columns = {"state0": [], "state1": [], "state2": [], "state3": []}
actions = []
rewards = []
terminated_list = []
truncated_list = []


## Initialize the "Cart-Pole" environment
env = gym.make("CartPole-v1")

# Initializing the random number generator
np.random.seed(int(time.time()))

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)
# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
TEST_RAND_PROB = 0.2

## Defining the simulation related constants
NUM_TRAIN_EPISODES = 300
MAX_TRAIN_STEPS = 500
SOLVED_T = 500
VERBOSE = False


def train():
    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_train_streaks = 0

    for episode in range(NUM_TRAIN_EPISODES):
        # Reset the environment
        obv, info = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)

        for t in range(MAX_TRAIN_STEPS):
            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, terminated, truncated, info = env.step(action)

            # Store data

            if episode >= 200:
                for i in range(4):
                    observation_columns[f"state{i}"].append(obv[i])
                actions.append(action)  # Record the chosen action
                rewards.append(reward)
                terminated_list.append(terminated)
                truncated_list.append(truncated)

            # Observe the result
            state = state_to_bucket(obv)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate * (
                reward + discount_factor * (best_q) - q_table[state_0 + (action,)]
            )

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if VERBOSE:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_train_streaks)

                print("")

            if terminated or truncated:
                print("Episode %d finished after %f time steps" % (episode, t))
                if t >= SOLVED_T:
                    num_train_streaks += 1
                else:
                    num_train_streaks = 0
                break

        # Update parameters
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (
                (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            )  # "index per boudns" * min_bound
            scaling = (NUM_BUCKETS[i] - 1) / bound_width  # how much index per width
            bucket_index = int(round(scaling * state[i] - offset))
            # For easier visualization of the above, you might want to use
            # pen and paper and apply some basic algebraic manipulations.
            # If you do so, you will obtaint (B-1)*[(S-MIN)]/W], where
            # B = NUM_BUCKETS, S = state, MIN = STATE_BOUNDS[i][0], and
            # W = bound_width. This simplification is very easy to
            # to visualize, i.e. num_buckets x percentage in width.
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":
    print("Training ...")
    train()
    # Create a DataFrame
    print("Creating csv files.....")
    data = {
        **observation_columns,
        "Action": actions,  # Add the actions column
        "Reward": rewards,
        "Terminated": terminated_list,
        "Truncated": truncated_list,
    }
    df = pd.DataFrame(data)

    # Save the data to a CSV file
    df.to_csv("data/cartpole_data.csv", index=False)
    print("Data saved to cartpole_data.csv")
