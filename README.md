# Research Proposal:  Combination Data Augmentation in Offline Reinforcement Learning for Robotics

## Steps to Run
1. use q-learning.py to generate baseline dataset.
2. use adv.py to create dataset with adverserial state training
3. use noise.py to create gaussian noise ,uniform noise , stacked adv_gaussian and stacked adv_uniform files
4. use files.py to combine all of the files, you should end up with 10 files for experiments.
5. run offline_RL.py (run on google collab, computational demanding, runtime ~ 1 hour) or just see results.ipynb for experiment setup and hyperparameter tuning


## Abstract
This research proposal is based on the paper "S4RL: Surprisingly Simple Self-Supervision for Offline Reinforcement Learning in Robotics." It explores the application of data augmentation techniques to state-based inputs in offline reinforcement learning (RL) and aims to reduce the cost of data collection in real-world scenarios. The proposal outlines research questions, motivations, and a plan for conducting experiments using classic control tasks (Cartpole and Mountain Car).

## Research Questions

### 1. Determining Optimal Augmentation Combinations
Can we identify augmentation combinations that transfer well across multiple tasks and environments? This question involves empirical evaluations across a suite of tasks and environments to identify effective combinations.

### 2. Sample Efficiency Impact
How does the sample efficiency and final performance scale with increasingly complex augmentation combinations? Are there diminishing returns and an optimal level of complexity?

### 3. Stacked or Combined
Is it better to perform stacked data augmentation on the data, or just perform data augmentations on randomly selected data?

## Motivation
The motivation for this research is threefold:
- **Data Collection Cost:** Real-world data collection can be expensive, making offline RL with pre-collected data an attractive option.
- **Sustainability:** Reducing the need for collecting large volumes of real-world data aligns with sustainability goals.
- **Practical Deployment:** Many real-world applications require RL algorithms to be trained on limited data before deployment.

## Experimental Plan

1. **Data Collection for Starting Dataset:** For this step, I will collect the starting dataset using either Q-learning , depending on the specific tasks and available resources. The number of episodes should be chosen based on the specific tasks and objectives of study.

2. **Data Augmentations:** I've chosen zero-mean Gaussian noise, zero-mean Uniform noise, and adversarial state training as potential data augmentation techniques. Combinations and stacking are important aspects to explore. I may want to conduct experiments with different combinations and stacking orders to see how they impact the learning process.

3. **Deep Q Learning (DQN):** Using the augmented datasets to train a Deep Q Network is a reasonable choice, especially for classic control problems like Cartpole and Mountain Car. I could also consider experimenting with other RL algorithms to compare their performance.

4. **Evaluation and Comparison:** This step is crucial to assess the effectiveness of the proposed augmentation techniques and combinations. I can evaluate and compare the performance of different augmentation strategies against each other and against a baseline (e.g., vanilla DQN without augmentation). Metrics to consider include training efficiency, final performance, and the ability to generalize to different tasks and environments.

# Data Augmentation Techniques 
## Adversarial Augmentation

- Generates adversarial examples by taking the gradient of the loss with respect to the inputs.
- Purpose: Exposes the model to edge cases that can fool it and improves its robustness.
- Adds small, structured perturbations that maximize the loss (where the Q-value deviates the most).
- Forces the model to learn invariances and not rely on fragile input features.

## Gaussian Noise Augmentation

- Adds random noise drawn from a Gaussian distribution to the inputs.
- The noise has a mean of 0, ensuring it doesn't bias the data in any particular direction.
- Purpose: Exposes the model to a wide variety of inputs near the original data.
- Helps improve generalization by preventing overfitting.
- Provides unstructured noise for generalization.

## Uniform Noise Augmentation

- Adds random noise uniformly sampled from a specified range [a, b].
- The noise has a zero mean, ensuring it's unbiased.
- The range [a, b] allows for controlling the intensity of the noise.
- Purpose: Enhances generalization by training the model on varied, corrupted inputs.
- Provides unstructured noise with a controlled range for diversity in training data.

In summary:

- **Adversarial Augmentation** is focused on exposing and addressing weaknesses in the model by introducing structured perturbations. It aims to improve model robustness and encourage it to be less reliant on specific input features.

- **Gaussian Noise Augmentation** introduces unstructured noise to the data, helping the model generalize better by experiencing a wide range of similar inputs. It prevents overfitting by adding variability to the training data.

- **Uniform Noise Augmentation** also introduces unstructured noise but allows for more control over the intensity of the noise. This further enhances generalization by providing a diverse set of inputs during training.

These techniques can be used individually or in combination, depending on the specific goals and challenges of reinforcement learning task. They are all aimed at improving the model's performance and robustness by exposing it to a broader spectrum of input data.


## Conclusion
This research proposal outlines a structured approach to investigate the application of data augmentation in offline reinforcement learning for robotics. By addressing the research questions and conducting experiments on classic control tasks, we aim to contribute to the understanding of how data augmentation can enhance the efficiency and practicality of RL algorithms in real-world applications.

---

Feel free to use this `README.md` as a starting point for Ir research project documentation, and tailor it to Ir specific needs and preferences.
#
