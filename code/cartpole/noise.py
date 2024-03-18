# file to apply gaussian and uniform noise
import pandas as pd
import numpy as np

# Load the CSV file
data = pd.read_csv("data/cartpole_data.csv")


# Define the hyperparameters for data augmentation
sigma = 0.0003  # Range for Gaussian noise
alpha = 0.0003  # Range for uniform noise
# Define state boundaries
state0_max = 2.4
state0_min = -2.4
state2_max = 0.2095
state2_min = -0.2095
# Define state boundaries with truncation/termination
state0_tmax = 4.8
state0_tmin = -4.8
state2_tmax = 0.418
state2_tmin = -0.418
print("applying Gaussian noise...")
# transformation is applied to accomadate stae values out of boundary


# Define a function for applying Gaussian noise
def apply_gaussian_noise(row):
    state = row[["state0", "state1", "state2", "state3"]].values
    noise = np.random.normal(0, sigma, size=4)
    perturbed_state = state + noise
    # Clip state values to specified boundaries
    if row["Truncated"] or row["Terminated"]:
        perturbed_state[0] = np.clip(perturbed_state[0], state0_tmin, state0_tmax)
        perturbed_state[2] = np.clip(perturbed_state[2], state2_tmin, state2_tmax)
    else:
        perturbed_state[0] = np.clip(perturbed_state[0], state0_min, state0_max)
        perturbed_state[2] = np.clip(perturbed_state[2], state2_min, state2_max)
    return pd.Series(
        {
            "state0": perturbed_state[0],
            "state1": perturbed_state[1],
            "state2": perturbed_state[2],
            "state3": perturbed_state[3],
            "Action": row["Action"],
            "Reward": row["Reward"],
            "Terminated": row["Terminated"],
            "Truncated": row["Truncated"],
        }
    )


print("applying Uniform noise...")


# Define a function for applying uniform noise
def apply_uniform_noise(row):
    state = row[["state0", "state1", "state2", "state3"]].values
    noise = np.random.uniform(-alpha, alpha, size=4)
    perturbed_state = state + noise
    # Clip state values to specified boundaries
    if row["Truncated"] or row["Terminated"]:
        perturbed_state[0] = np.clip(perturbed_state[0], state0_tmin, state0_tmax)
        perturbed_state[2] = np.clip(perturbed_state[2], state2_tmin, state2_tmax)
    else:
        perturbed_state[0] = np.clip(perturbed_state[0], state0_min, state0_max)
        perturbed_state[2] = np.clip(perturbed_state[2], state2_min, state2_max)
    return pd.Series(
        {
            "state0": perturbed_state[0],
            "state1": perturbed_state[1],
            "state2": perturbed_state[2],
            "state3": perturbed_state[3],
            "Action": row["Action"],
            "Reward": row["Reward"],
            "Terminated": row["Terminated"],
            "Truncated": row["Truncated"],
        }
    )


# Apply data augmentation methods
data_with_gaussian_noise = data.apply(apply_gaussian_noise, axis=1)
data_with_uniform_noise = data.apply(apply_uniform_noise, axis=1)


# Save augmented data to new CSV files
data_with_gaussian_noise.to_csv("data/augmented_data_gaussian.csv", index=False)
data_with_uniform_noise.to_csv("data/augmented_data_uniform.csv", index=False)

print("applying noise to adv files...")
# stacked data augmentation
data_adv = pd.read_csv("data/augmented_data_adv.csv")
data_with_gaussian_noise = data_adv.apply(apply_gaussian_noise, axis=1)
data_with_gaussian_noise.to_csv("data/gauss_adv.csv", index=False)
data_with_gaussian_noise = data_adv.apply(apply_uniform_noise, axis=1)
data_with_gaussian_noise.to_csv("data/uniform_adv.csv", index=False)
