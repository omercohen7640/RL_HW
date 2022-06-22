import matplotlib.pyplot as plt
from mountain_car_with_data_collection import *
from radial_basis_function_extractor import *
import numpy as np
from data_collector import DataCollector
from data_transformer import DataTransformer
from lspi import compute_lspi_iteration
from game_player import GamePlayer
from linear_policy import LinearPolicy
from q_learn_mountain_car import *

def evaluation(env, data_transformer, feature_extractor, policy):
    max_steps_per_game = 200
    number_of_games = 50
    play = GamePlayer(env, data_transformer, feature_extractor, policy)
    start_pos = np.random.uniform(env.low[0], env.high[0], size=number_of_games)
    all_results = [play.play_game(max_steps_per_game, start_state=[start_pos[i], 0]) for i in range(number_of_games)]
    success_rate = np.mean(all_results)
    #print(f'success rate is {success_rate}')
    return success_rate


if __name__ == '__main__':

    # Question 2 section 1
    env = MountainCarWithResetEnv()
    pos = np.linspace(env.low[0]-1.5, env.high[0]-2, 100)
    vel = np.linspace(env.low[1]-2.1, env.high[1]-0.9, 100)
    P, V = np.meshgrid(pos, vel)
    states = np.concatenate((P.reshape(-1, 1), V.reshape(-1, 1)), axis=1)
    number_of_kernels_per_dim = [12, 10]
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, encoded_states[:, 0].reshape(P.shape), cmap='twilight')
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_zlabel("Feature Value")
    ax.set_title("Feature 1")
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(P, V, encoded_states[:, 1].reshape(P.shape), cmap='twilight')
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_zlabel("Feature Value")
    ax.set_title("Feature 2")
    plt.show()


    # Question 3 section 2

    samples_to_collect = 100000
    #samples_to_collect = 100
    np.random.seed(123)
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)
    print(f'data mean {state_mean}')
    print(f'data std {state_std}')


    # Question 3 section 5


    w_updates = 20
    gamma = 0.999
    data_transformer = DataTransformer()
    """successes = np.zeros((3, w_updates))
    num_of_iter = []
    for i, seed in enumerate([110, 123, 136]):
        np.random.seed(seed)
        linear_policy = LinearPolicy(120, 3, include_bias=True)
        states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
        data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
        states=data_transformer.transform_states(states)
        next_states=data_transformer.transform_states(next_states)
        encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
        encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')

            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)
            success_rate = evaluation(env, data_transformer, feature_extractor, linear_policy)
            print(success_rate)
            successes[i, lspi_iteration] = success_rate

            if norm_diff < 0.00001:
                num_of_iter += [lspi_iteration]
                break
    min_iter = min(num_of_iter)
    mean_success = np.mean(successes, axis=0)
    plt.plot(range(min_iter), mean_success[:min_iter])
    plt.title("Average Success Rate over LSPI Iterations")
    plt.xlabel("LSPI Itaration")
    plt.ylabel("Success Rate")
    plt.show()"""


    # Qusetion 3 section 6
    successes = np.zeros((3, 3, w_updates))
    num_of_iter = []
    for j, samples in enumerate([1000, 50000, 200000]):
        for i, seed in enumerate([110, 123, 136]):
            np.random.seed(seed)
            linear_policy = LinearPolicy(120, 3, include_bias=True)
            states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples)
            data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
            states = data_transformer.transform_states(states)
            next_states = data_transformer.transform_states(next_states)
            encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
            encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
            for lspi_iteration in range(w_updates):
                print(f'starting lspi iteration {lspi_iteration}')

                new_w = compute_lspi_iteration(
                    encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
                )
                norm_diff = linear_policy.set_w(new_w)
                success_rate = evaluation(env, data_transformer, feature_extractor, linear_policy)
                print(success_rate)
                successes[i, j, lspi_iteration] = success_rate

                if norm_diff < 0.00001:
                    num_of_iter += [lspi_iteration]
                    break

    min_iter = min(num_of_iter)
    mean_success = np.mean(successes, axis=0)

    fig, ax = plt.subplots()
    ax.plot(range(min_iter), mean_success[0, :min_iter])
    ax.plot(range(min_iter), mean_success[1, :min_iter])
    ax.plot(range(min_iter), mean_success[2, :min_iter])
    ax.set_xlabel("LSPI Itaration")
    ax.set_ylabel("Success rate")
    ax.legend(["1,000 samples", "50,000 samples", "200,000 samples"])
    plt.show()



