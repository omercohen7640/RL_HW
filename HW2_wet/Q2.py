import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

mu = [0.6, 0.5, 0.3, 0.7, 0.1]
c = [1, 4, 6, 2, 9]
sample_list = [0, 1, 2, 3, 4]
final_list = []
for n in reversed(range(len(sample_list)+1)):
    final_list+=list(combinations(sample_list, n))
state_dict={}
index_dict = {}
for i,t in enumerate(final_list):
    state_dict[t] = i
    index_dict[i] = t






def solve_belman(policy):

    P_pi = np.zeros((32, 32))
    rewards = np.zeros((32, 1))
    for i in range(31):
        state = list(index_dict[i])
        for job in state:
            rewards[i] -= c[job]
        act = policy[i]
        state.remove(act)
        """try:
            tup = tuple(next_s)
        except:
            print('hi')"""
        next_idx = state_dict[tuple(state)]
        P_pi[next_idx, i] = mu[act]
        P_pi[i, i] = 1 - mu[act]

    P_pi[31,31] = 1
    V_pi = np.zeros((32, 1))
    for t in range(500):
        V_pi = rewards + np.matmul(np.transpose(P_pi), V_pi)


    #return np.matmul(np.linalg.inv(np.eye(32)-P_pi), rewards)
    return V_pi

def pi_c_policy():
    pi_c = np.zeros(32, dtype=int)
    for i in range(31):
        jobs = list(index_dict[i])
        idx = np.argmax([c[j] for j in jobs])
        pi_c[i] = jobs[idx]
    return pi_c


if __name__ == '__main__':

    pi_c = pi_c_policy()
    V_c = solve_belman(pi_c)

    plt.plot(range(32), sorted(-V_c), color='lightskyblue')
    plt.ylabel('$Value$ $Function$')
    plt.xlabel('$state$')
    plt.title("$Value$ $Function$ $V^\pi$ $of$ $Max$ $Policy$ $\pi_c$")
    plt.show()

