import numpy as np
import time
import matplotlib.pyplot as plt

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor


class Solver:
    def __init__(self, number_of_kernels_per_dim, number_of_actions, gamma, learning_rate):
        # Set max value for normalization of inputs
        self._max_normal = 1
        # get state \action information
        self.data_transformer = DataTransformer()
        state_mean = [-3.00283763e-01,  5.61618575e-05]
        state_std = [0.51981243, 0.04024895]
        self.data_transformer.set(state_mean, state_std)
        self._actions = number_of_actions
        # create RBF features:
        self.feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        self.number_of_features = self.feature_extractor.get_number_of_features()
        # the weights of the q learner
        self.theta = np.random.uniform(-0.001, 0, size=number_of_actions * self.number_of_features)
        # discount factor for the solver
        self.gamma = gamma
        self.learning_rate = learning_rate

    def _normalize_state(self, s):
        return self.data_transformer.transform_states(np.array([s]))[0]

    def get_features(self, state):
        normalized_state = self._normalize_state(state)
        features = self.feature_extractor.encode_states_with_radial_basis_functions([normalized_state])[0]
        return features

    def get_q_val(self, features, action):
        theta_ = self.theta[action*self.number_of_features: (1 + action)*self.number_of_features]
        return np.dot(features, theta_)

    def get_all_q_vals(self, features):
        all_vals = np.zeros(self._actions)
        for a in range(self._actions):
            all_vals[a] = solver.get_q_val(features, a)
        return all_vals

    def get_max_action(self, state):
        sparse_features = solver.get_features(state)
        q_vals = solver.get_all_q_vals(sparse_features)
        return np.argmax(q_vals)

    def get_state_action_features(self, state, action):
        state_features = self.get_features(state)
        all_features = np.zeros(len(state_features) * self._actions)
        all_features[action * len(state_features): (1 + action) * len(state_features)] = state_features
        return all_features

    def update_theta(self, state, action, reward, next_state, done):
        # compute the new weights and set in self.theta. also return the bellman error (for tracking).
        q_sa = self.get_q_val(self.get_features(state),action)
        best_action = self.get_max_action(next_state)
        q_next_s_a = self.get_q_val(self.get_features(next_state),best_action)
        if done:
            err = 0.0
        else:
            err = reward + self.gamma*q_next_s_a - q_sa
        features = self.get_state_action_features(state,action)
        self.theta = self.theta + self.learning_rate*err*features
        return err


def modify_reward(reward):
    reward -= 1
    if reward == 0.:
        reward = 100.
    return reward


def run_episode(env, solver, is_train=True, epsilon=None, max_steps=200, render=False):
    episode_gain = 0
    deltas = []
    if is_train:
        start_position = np.random.uniform(env.min_position, env.goal_position - 0.01)
        start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
    else:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
    state = env.reset_specific(start_position, start_velocity)
    step = 0
    if render:
        env.render()
        time.sleep(0.1)
    while True:
        if epsilon is not None and np.random.uniform() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            action = solver.get_max_action(state)
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        reward = modify_reward(reward)
        step += 1
        episode_gain += reward
        if is_train:
            deltas.append(solver.update_theta(state, action, reward, next_state, done))
        if done or step == max_steps:
            return episode_gain, np.mean(deltas)
        state = next_state


if __name__ == "__main__":
    run_section = 5
    env = MountainCarWithResetEnv()
    seeds = [123, 234, 345]
    epsilons = [0.1]
    if run_section == 5:
        seeds = [123]
        epsilons = [1.0,0.75,0.5,0.3,0.001]
    #seeds = [123]
    #seed = 234
    #seed = 345


    gamma = 0.999
    learning_rate = 0.01
    epsilon_current = 0.1
    epsilon_decrease = 1.
    epsilon_min = 0.05

    max_episodes = 100000


    total_rewards = [[],[],[]]
    bottom_hill_value = [[],[],[]]
    total_bellman_error = [[],[],[]]
    total_eval = [[],[],[]]
    total_reward_epsilon =[[],[],[],[],[]]
    for epsilon_idx,epsilon in enumerate(epsilons):
        epsilon_current = epsilon
        for seed_idx,seed in enumerate(seeds):
            np.random.seed(seed)
            env.seed(seed)
            last_100_bellman_err = []
            success_counter = 0
            solver = Solver(
                # learning parameters
                gamma=gamma, learning_rate=learning_rate,
                # feature extraction parameters
                number_of_kernels_per_dim=[7, 5],
                # env dependencies (DO NOT CHANGE):
                number_of_actions=env.action_space.n,
            )
            for episode_index in range(1, max_episodes + 1):
                episode_gain, mean_delta = run_episode(env, solver, is_train=True, epsilon=epsilon_current)

                # reduce epsilon if required
                epsilon_current *= epsilon_decrease
                epsilon_current = max(epsilon_current, epsilon_min)

                print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}, average error {mean_delta}')
                # total reward
                if run_section == 5:
                    total_reward_epsilon[epsilon_idx].append(episode_gain)
                if run_section == 4:
                    total_rewards[seed_idx].append(episode_gain)
                    # bottom hill value
                    bottom_hill_feature_ext = solver.get_features(np.array([-0.5,0]))
                    bottom_hill_value[seed_idx].append(np.mean(solver.get_all_q_vals(bottom_hill_feature_ext)))
                    # total_bellman_error
                    last_100_bellman_err.append(mean_delta)
                    if len(last_100_bellman_err) > 100:
                        last_100_bellman_err.pop(0)
                    total_bellman_error[seed_idx].append(np.mean(last_100_bellman_err))
                # termination condition:
                if episode_index % 10 == 9:
                    test_gains = [run_episode(env, solver, is_train=False, epsilon=0.)[0] for _ in range(10)]
                    mean_test_gain = np.mean(test_gains)
                    # total eval
                    success_rate = np.array(test_gains) > - 75
                    if run_section == 4:
                        total_eval[seed_idx].append(np.mean(success_rate))
                    print(f'tested 10 episodes: mean gain is {mean_test_gain}')
                    if mean_test_gain >= -75.:
                        if success_counter == 0:
                            print(f'solved in {episode_index} episodes')
                        success_counter += 1
                        if success_counter >= 9:
                            break

        #run_episode(env, solver, is_train=False, render=True)

        # section 3 plot
    if run_section == 4:
        for i in range(len(seeds)):
            # graph 1
            plt.figure()
            plt.plot(np.arange(1, len(total_rewards[i]) + 1,10), total_rewards[i][::10])
            plt.title("Total Reward in the Training Episode vs. Training Episodes - seed: {}".format(seeds[i]))
            plt.xlabel("Training Episodes")
            plt.ylabel("Total Reward")
            # graph 2
            plt.figure()
            plt.plot(np.arange(1, len(total_eval[i]) + 1), total_eval[i])
            plt.title("Success Rate vs. Training Episodes - seed: {}".format(seeds[i]))
            plt.xlabel("Training Episodes")
            plt.ylabel("Success Rate")
            # graph 3
            plt.figure()
            plt.plot(range(len(bottom_hill_value[i])), bottom_hill_value[i])
            plt.title("Approximated Bottom Hill State Value vs. Training Episodes - seed: {}".format(seeds[i]))
            plt.xlabel("Training Episodes")
            plt.ylabel("Bottom Hill Value")
            # graph 4
            plt.figure()
            plt.plot(range(len(total_bellman_error[i])), total_bellman_error[i])
            plt.title(
                "Total Bellman Error of The Episode (averaged over 100 episodes) vs. Training Episodes - seed: {}".format(
                    seeds[i]))
            plt.xlabel("Training Episodes")
            plt.ylabel("Bellman Error")
            plt.show()
    if run_section == 5:
        for i in range(len(epsilons)):
            # graph 1
            plt.figure()
            plt.plot(np.arange(1, len(total_reward_epsilon[i]) + 1, 10), total_reward_epsilon[i][::10])
            plt.title("Total Reward in the Training Episode vs. Training Episodes - epsilon: {}".format(epsilons[i]))
            plt.xlabel("Training Episodes")
            plt.ylabel("Total Reward")
        plt.show()