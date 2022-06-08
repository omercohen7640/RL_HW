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

rewards = np.zeros((32, 1))
for i in range(31):
    state = list(index_dict[i])
    for job in state:
        rewards[i] -= c[job]

a = 0




def solve_belman(policy):

    P_pi = np.zeros((32, 32))
    #rewards = np.zeros((32, 1))
    for i in range(31):
        state = list(index_dict[i])
        #for job in state:
        #    rewards[i] -= c[job]
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
    for t in range(1000):
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

def c_mu_policy():
    pi = np.zeros(32, dtype=int)
    for i in range(31):
        jobs = list(index_dict[i])
        idx = np.argmax([c[j]*mu[j] for j in jobs])
        pi[i] = jobs[idx]
    return pi


def policy_iteration():
    pi = pi_c_policy()
    V = solve_belman(pi)
    V_s0 = [V[0].copy()]
    #rewards = np.zeros((32, 1))
    #for i in range(31):
        #state = list(index_dict[i])
        #for job in state:
        #    rewards[i] -= c[job]
    for j in range(3):
        for i in reversed(range(31)):
            state = list(index_dict[i])
            for job in state:
                ref_state = state.copy()
                ref_state.remove(job)
                next_idx = state_dict[tuple(ref_state)]
                #val = rewards[i] + (1-mu[job])*V[i] +mu[job]*V[next_idx]
                val = rewards[i]/mu[job] + V[next_idx]
                if val > V[i]:
                    V[i] = val
                    pi[i] = job
        V_s0.append(V[0].copy())

    return pi, V_s0

def simulation(state_idx, action):

    #state_idx = state_dict[tuple(state)]
    state = list(index_dict[state_idx])
    next_state = state.copy()
    if state_idx == 31:
        begin_state = 0
        terminal_reward = 0
        return begin_state, terminal_reward
    if np.random.uniform() < mu[action]:
        next_state.remove(action)
    next_state_idx = state_dict[tuple(next_state)]

    return next_state_idx, rewards[state_idx]

def get_step_size(visits, method, idx):
    if method == 1:
        return 1/visits[idx]
    elif method == 2:
        return 0.01
    else:
        return 10/(100+visits[idx])

def TD_0(method):
    pi = pi_c_policy()
    V = solve_belman(pi)
    err_inf = []
    err_s0 = []
    visits = np.ones((32, 1))
    V_hat = np.zeros((32, 1))
    state_idx = 0
    for i in range(40000):
        next_state, reward = simulation(state_idx, pi[state_idx])
        step_size = get_step_size(visits, method, state_idx)
        visits[state_idx] += 1
        if state_idx == 31:
            err = np.abs(V_hat-V)
            err_s0.append(err[0].copy())
            err_inf.append(max(err.copy()))
            state_idx = next_state
            continue
        V_hat[state_idx] += step_size*(reward + V_hat[next_state]-V_hat[state_idx])
        state_idx = next_state


    return err_inf, err_s0


def TD_lambda(method, lamb):
    pi = pi_c_policy()
    V = solve_belman(pi)
    err_inf = []
    err_s0 = []
    visits = np.ones((32, 1))
    V_hat = np.zeros((32, 1))
    state_idx = 0
    e_tm1 = np.zeros((32, 1))
    e_t = np.zeros((32, 1))
    for i in range(80000):
        next_state, reward = simulation(state_idx, pi[state_idx])
        step_size = get_step_size(visits, method, state_idx)
        visits[state_idx] += 1
        if state_idx == 31:
            err = np.abs(V_hat - V)
            err_s0.append(err[0].copy())
            err_inf.append(max(err.copy()))
            state_idx = next_state
            continue
        dt = step_size * (reward + V_hat[next_state] - V_hat[state_idx])
        one_hot = np.zeros((32, 1))
        one_hot[state_idx] = 1
        e_t = lamb*e_tm1+one_hot
        V_hat += step_size*dt*e_t
        e_tm1 = e_t
        state_idx = next_state

    return err_inf, err_s0




if __name__ == '__main__':

    pi_c = pi_c_policy()
    V_c = solve_belman(pi_c)

    plt.plot(range(32), sorted(-V_c), color='lightskyblue')
    plt.ylabel('$Value$ $Function$')
    plt.xlabel('$state$')
    plt.title("$Value$ $Function$ $V^\pi$ $of$ $Max$ $Policy$ $\pi_c$")
    plt.show()


    pi_star, V_s0 = policy_iteration()

    plt.plot(range(len(V_s0)), V_s0, color='lightskyblue')
    plt.ylabel('$Value$ $Function$')
    plt.xlabel('$iteration$')
    plt.title("$Initial$ $State$ $S_0$ $Value$ $Function$ $for$ $Policy$ $Iteration$")
    plt.show()

    # section e
    V_star = solve_belman(pi_star)
    pi_c_mu = c_mu_policy()
    V_c_mu = solve_belman(pi_c_mu)

    plt.plot(range(32), V_c, color='lightskyblue')
    plt.plot(range(32), V_star, color='pink')
    plt.legend(['$V^{\pi_c}$', '$V^{\pi^*}$'])
    plt.ylabel('$Value$ $Function$')
    plt.xlabel('$state$')
    plt.title("$Value$ $Function$ $V^{\pi_c}$ $vs$ $V^{\pi^*}$")
    plt.show()

    plt.scatter(range(32), pi_c_mu, marker='x', color='lightskyblue')
    plt.scatter(range(32), pi_star, marker='x', color='pink')
    plt.legend(['$\pi_{c\mu}$', '$\pi^*$'])
    plt.yticks(range(1, 6))
    plt.ylabel('$Action$')
    plt.xlabel('$state$')
    plt.title("$\pi_{c\mu}$ $vs$ $\pi^*$")
    plt.show()

    err_inf, err_s0 = TD_lambda(2, 0.75)


    a=0
