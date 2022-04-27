import numpy as np
from cartpole_cont import CartPoleContEnv
import matplotlib.pyplot as plt
from lqr import *




if __name__ == '__main__':
    thetas = []
    for theta in [0.1 * np.pi, 0.22 * np.pi, 0.44 * np.pi]:
        env = CartPoleContEnv(initial_theta=theta)
        # the following is an example to start at a different theta
        # env = CartPoleContEnv(initial_theta=np.pi * 0.25)

        # print the matrices used in LQR
        print('A: {}'.format(get_A(env)))
        print('B: {}'.format(get_B(env)))

        # start a new episode
        actual_state = env.reset()
        env.render()
        # use LQR to plan controls
        xs, us, Ks = find_lqr_control_input(env)
        # run the episode until termination, and print the difference between planned and actual
        is_done = False
        iteration = 0
        is_stable_all = []
        episode_theta = np.empty(0)
        while not is_done:
            # print the differences between planning and execution time
            predicted_theta = xs[iteration].item(2)
            actual_theta = actual_state[2]
            episode_theta = np.append(episode_theta, actual_theta)
            predicted_action = us[iteration].item(0)
            actual_action = (Ks[iteration] * np.expand_dims(actual_state, 1).astype(np.float32)).item(0)
            print_diff(iteration, predicted_theta, actual_theta, predicted_action, actual_action)
            # apply action according to actual state visited
            # make action in range
            actual_action = max(env.action_space.low.item(0), min(env.action_space.high.item(0), actual_action))
            actual_action = np.array([actual_action]).astype(np.float32)
            actual_state, reward, is_done, _ = env.step(actual_action)
            is_stable = reward == 1.0
            is_stable_all.append(is_stable)
            env.render()
            iteration += 1
        thetas.append(episode_theta)
        env.close()
    plt.plot(thetas[0], color='r', label='$0.1*\pi$')
    plt.plot(thetas[1], color='g', label='$0.22*\pi$')
    plt.plot(thetas[2], color='b', label='$0.44*\pi$')
    plt.xlabel("time horizon")
    plt.ylabel("angle")
    plt.title("$\theta $")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()