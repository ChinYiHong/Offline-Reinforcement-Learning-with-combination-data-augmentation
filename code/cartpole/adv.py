# file to apply adverserial state training
import numpy as np
import pandas as pd
import tensorflow as tf

# Hyperparameters
epsilon = 0.0001

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


# Define a Q-value prediction model using a simple linear model
class Model:
    def __init__(self):
        # Initialize model weights as a 4x2 matrix
        self.w = tf.Variable(tf.random.normal((4, 2)))  # Use TensorFlow Variable

    def predict(self, state):
        # Reshape state for matrix multiplication
        state = tf.reshape(state, (1, 4))
        q_values = tf.matmul(state, self.w)  # Use tf.matmul for matrix multiplication
        return q_values


model = Model()

df = pd.read_csv("data/cartpole_data.csv")


def augment(state, terminated, truncated):
    state_tensor = tf.convert_to_tensor(
        state, dtype=tf.float32
    )  # Convert to TensorFlow tensor

    with tf.GradientTape() as tape:
        tape.watch(state_tensor)
        q_values = model.predict(state_tensor)
        loss = tf.reduce_sum(tf.square(q_values))  # Use tf.reduce_sum for the loss

    grad = tape.gradient(loss, state_tensor)

    state0_clip_min = state0_tmin if terminated or truncated else state0_min
    state0_clip_max = state0_tmax if terminated or truncated else state0_max
    state2_clip_min = state2_tmin if terminated or truncated else state2_min
    state2_clip_max = state2_tmax if terminated or truncated else state2_max

    # Update the state using tf.add
    augmented_state = tf.add(state_tensor, epsilon * grad)
    augmented_state = augmented_state.numpy()
    # Clip the augmented state values
    augmented_state[0] = np.clip(augmented_state[0], state0_clip_min, state0_clip_max)
    augmented_state[2] = np.clip(augmented_state[2], state2_clip_min, state2_clip_max)

    return augmented_state  # Convert the augmented state back to a NumPy array


aug_states = []

print("Clipping adv augment datasets...")
for index, row in df.iterrows():
    state = row[["state0", "state1", "state2", "state3"]].values
    terminated = row["Terminated"]
    truncated = row["Truncated"]

    aug_state = augment(state, terminated, truncated)
    aug_states.append(aug_state)

print("Preparing adv augment datasets...")

# Update the DataFrame with the augmented state values
df["state0"] = [s[0] for s in aug_states]
df["state1"] = [s[1] for s in aug_states]
df["state2"] = [s[2] for s in aug_states]
df["state3"] = [s[3] for s in aug_states]

df.to_csv("data/augmented_data_adv.csv", index=False)
print("adv augment datasets added to augmented_data_adv.csv")
