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

N = 500






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
        #P_pi[next_idx, i] = mu[act]
        #P_pi[i, i] = 1 - mu[act]

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
    try:
        if np.random.uniform() < mu[action]:
            next_state.remove(action)
    except Exception as e:
        print('b')
    next_state_idx = state_dict[tuple(next_state)]

    return next_state_idx, rewards[state_idx]

def get_step_size(visits, method, idx):
    if method == 0:
        return 1/visits[idx] if isinstance(idx,int) else 1/visits[idx[0], idx[1]]
    elif method == 1:
        return 0.01
    else:
        return 10/(100+visits[idx]) if isinstance(idx, int) else 10/(100+visits[idx[0], idx[1]])

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
    i = 0
    err = np.abs(V_hat - V)
    err_s0.append(err[0].copy())
    err_inf.append(max(err.copy()))
    while i < N-1:
        next_state, reward = simulation(state_idx, pi[state_idx])
        step_size = get_step_size(visits, method, state_idx)
        visits[state_idx] += 1
        if state_idx == 31:
            err = np.abs(V_hat - V)
            err_s0.append(err[0].copy())
            err_inf.append(max(err.copy()))
            state_idx = next_state
            i+=1
            continue
        dt = (reward + V_hat[next_state] - V_hat[state_idx])
        one_hot = np.zeros((32, 1))
        one_hot[state_idx] = 1
        e_t = lamb*e_tm1+one_hot
        V_hat += step_size*dt*e_t
        e_tm1 = e_t
        state_idx = next_state

    return err_inf, err_s0

def initialize_pi():
    pi = []
    for i in range(31):
        state = list(index_dict[i])
        pi += [np.random.choice(state)]
    pi += [0]

    return pi

def Q_learning(eps, method, V_star):

    pi = initialize_pi()
    err_inf = []
    err_s0 = []
    visits = np.ones((32, 5))
    Q_hat = np.zeros((32, 5))
    state_idx = 0
    i = 0
    while i < 8000*N-1:
        curr_state = list(index_dict[state_idx])
        if np.random.uniform() < eps and len(curr_state) > 1: # epsilon greedy
            act = np.random.choice(curr_state)
        else:
            act = pi[state_idx]

        next_state, reward = simulation(state_idx, act)
        step_size = get_step_size(visits, method, [state_idx, act])
        visits[state_idx, act] += 1

        if state_idx == 31:
            if i % 3 == 0:
                for s in range(31):
                    state_upd = index_dict[s]
                    pi[s] = state_upd[np.argmax(Q_hat[s, state_upd])]
            if i % 200 == 0:
                V = solve_belman(pi)
                err = np.abs(V - V_star)
                err_s0.append(np.abs(np.max(Q_hat[0, :])-V_star[0]))
                err_inf.append(max(err.copy()))
                state_idx = next_state
            i += 1
            continue
        Q_hat[state_idx, act] += step_size * (reward + Q_hat[next_state, pi[next_state]] - Q_hat[state_idx, act])

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

    plt.plot(range(len(V_s0)), -np.array(V_s0), color='lightskyblue')
    plt.ylabel('$Value$ $Function$')
    plt.xlabel('$iteration$')
    plt.title("$Initial$ $State$ $S_0$ $Value$ $Function$ $for$ $Policy$ $Iteration$")
    plt.show()

    # section e
    V_star = solve_belman(pi_star)
    pi_c_mu = c_mu_policy()
    V_c_mu = solve_belman(pi_c_mu)

    plt.plot(range(32), -np.array(V_c), color='lightskyblue')
    plt.plot(range(32), -np.array(V_star), color='pink')
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


    # section g
    err_inf_1, err_s0_1 = TD_0(0)
    err_inf_2, err_s0_2 = TD_0(1)
    err_inf_3, err_s0_3 = TD_0(2)

    plt.plot(range(len(err_inf_1)), err_inf_1, color='lightskyblue')
    plt.plot(range(len(err_inf_2)), err_inf_2, color='pink')
    plt.plot(range(len(err_inf_3)), err_inf_3, color='green')
    plt.legend([r'$\alpha=\frac{1}{no. visits (s)}$', r'$\alpha=0.01$', r'$\alpha=\frac{10}{100 + no. visits (s)}$'])
    plt.ylabel(r'$|V^{\pi_c}$ - $\^V_{TD}|_{\infty}$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Infinity Norm$ $|V^{\pi_c}$ - $\^V_{TD}|_{\infty}$')
    plt.show()

    plt.plot(range(len(err_s0_1)), err_s0_1, color='lightskyblue')
    plt.plot(range(len(err_s0_2)), err_s0_2, color='pink')
    plt.plot(range(len(err_s0_3)), err_s0_3, color='green')
    plt.legend([r'$\alpha=\frac{1}{no. visits (s)}$', r'$\alpha=0.01$', r'$\alpha=\frac{10}{100 + no. visits (s)}$'])
    plt.ylabel(r'$|V^{\pi_c}(s_0)$ - $\^V_{TD}(s_0)|$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Initial$ $State$ $s_0$ $|V^{\pi_c}(s_0)$ - $\^V_{TD}(s_0)|$')
    plt.show()

    # section h

    lambdas = [0.25, 0.5, 0.75]
    lis_V_inf = []
    lis_V_s0 = []
    for lamb in lambdas:
        V_inf = np.zeros((N, 1))
        V_s0 = np.zeros((N, 1))
        for i in range(20):
            err_inf, err_s0 = TD_lambda(0, lamb)
            V_inf += np.array(err_inf)/20
            V_s0 += np.array(err_s0) / 20
        lis_V_inf.append(V_inf)
        lis_V_s0.append(V_s0)

    plt.plot(range(len(lis_V_inf[0])), lis_V_inf[0], color='lightskyblue')
    plt.plot(range(len(lis_V_inf[1])), lis_V_inf[1], color='pink')
    plt.plot(range(len(lis_V_inf[2])), lis_V_inf[2], color='green')
    plt.legend([r'$\lambda=0.25$', r'$\lambda=0.5$', r'$\lambda=0.75$'])
    plt.ylabel(r'$|V^{\pi_c}$ - $\^V_{TD(\lambda)}|_{\infty}$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Infinity Norm$ $|V^{\pi_c}$ - $\^V_{TD(\lambda)}|_{\infty}$')
    plt.show()

    plt.plot(range(len(lis_V_s0[0])), lis_V_s0[0], color='lightskyblue')
    plt.plot(range(len(lis_V_s0[1])), lis_V_s0[1], color='pink')
    plt.plot(range(len(lis_V_s0[2])), lis_V_s0[2], color='green')
    plt.legend([r'$\lambda=0.25$', r'$\lambda=0.5$', r'$\lambda=0.75$'])
    plt.ylabel(r'$|V^{\pi_c}(s_0)$ - $\^V_{TD(\lambda)}(s_0)|$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Initial$ $State$ $s_0$ $|V^{\pi_c}(s_0)$ - $\^V_{TD(\lambda)}(s_0)|$')
    plt.show()

    # section i

    err_inf_1, err_s0_1 = Q_learning(0.1, 0, V_star)
    err_inf_2, err_s0_2 = Q_learning(0.1, 1, V_star)
    err_inf_3, err_s0_3 = Q_learning(0.1, 2, V_star)

    plt.plot(range(len(err_inf_1)), err_inf_1, color='lightskyblue')
    plt.plot(range(len(err_inf_2)), err_inf_2, color='pink')
    plt.plot(range(len(err_inf_3)), err_inf_3, color='green')
    plt.legend([r'$\alpha=\frac{1}{no. visits (s)}$', r'$\alpha=0.01$', r'$\alpha=\frac{10}{100 + no. visits (s)}$'])
    plt.ylabel(r'$|V^{\pi^*}$ - $\^V_{\pi_Q}|_{\infty}$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Infinity Norm$ $|V^{\pi^*}$ - $\^V_{\pi_Q}|_{\infty}$')
    plt.show()

    plt.plot(range(len(err_s0_1)), err_s0_1, color='lightskyblue')
    plt.plot(range(len(err_s0_2)), err_s0_2, color='pink')
    plt.plot(range(len(err_s0_3)), err_s0_3, color='green')
    plt.legend([r'$\alpha=\frac{1}{no. visits (s)}$', r'$\alpha=0.01$', r'$\alpha=\frac{10}{100 + no. visits (s)}$'])
    plt.ylabel(r'$|V^{\pi^*}(s_0)$ - $min_a{Q(s_0,a)}|$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Initial$ $State$ $s_0$ $|V^{\pi^*}(s_0)$ - $min_a{Q(s_0,a)}|$')
    plt.show()


    # section j

    err_inf_01, err_s0_01 = Q_learning(0.01, 1, V_star)

    plt.plot(range(len(err_inf_2)), err_inf_2, color='lightskyblue')
    plt.plot(range(len(err_inf_01)), err_inf_01, color='pink')
    plt.legend([r'$\epsilon=0.1$', r'$\epsilon=0.01$'])
    plt.ylabel(r'$|V^{\pi^*}$ - $\^V_{\pi_Q}|_{\infty}$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Infinity Norm$ $|V^{\pi^*}$ - $\^V_{\pi_Q}|_{\infty}$')
    plt.show()

    plt.plot(range(len(err_s0_2)), err_s0_2, color='lightskyblue')
    plt.plot(range(len(err_s0_01)), err_s0_01, color='pink')
    plt.legend([r'$\epsilon=0.1$', r'$\epsilon=0.01$'])
    plt.ylabel(r'$|V^{\pi^*}(s_0)$ - $min_a{Q(s_0,a)}|$')
    plt.xlabel('$iteration$')
    plt.title(r'$Error$ $of$ $Initial$ $State$ $s_0$ $|V^{\pi^*}(s_0)$ - $min_a{Q(s_0,a)}|$')
    plt.show()


