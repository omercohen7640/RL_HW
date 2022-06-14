import matplotlib.pyplot as plt
from mountain_car_with_data_collection import *
from radial_basis_function_extractor import *
import numpy as np
from data_collector import DataCollector
from data_transformer import DataTransformer
from lspi import compute_lspi_iteration

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
    np.random.seed(123)
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0)
    print(f'data mean {state_mean}')
    print(f'data std {state_std}')


    # Question 3 section 5

    w_updates = 20
    for seed in [110, 123, 136]:
        np.random.seed(seed)
        for lspi_iteration in range(w_updates):
            print(f'starting lspi iteration {lspi_iteration}')

            new_w = compute_lspi_iteration(
                encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            )
            norm_diff = linear_policy.set_w(new_w)
            if norm_diff < 0.00001:
                break
