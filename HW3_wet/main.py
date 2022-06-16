import matplotlib.pyplot as plt
from mountain_car_with_data_collection import *
from radial_basis_function_extractor import *
import numpy as np
from data_collector import DataCollector
from data_transformer import DataTransformer
from lspi import compute_lspi_iteration
from game_player import GamePlayer
from linear_policy import LinearPolicy

def evaluation(env, data_transformer, feature_extractor, policy):
    max_steps_per_game = 1000
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
    for seed in [110, 123, 136]:
        np.random.seed(seed)
        linear_policy = LinearPolicy(120, 3, include_bias=False)
        states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
        data_transformer.set_using_states(states)
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
            if norm_diff < 0.00001:
                break
        success_rate = evaluation(env, data_transformer, feature_extractor, linear_policy)
