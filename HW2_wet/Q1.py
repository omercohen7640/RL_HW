import numpy as np
import matplotlib.pyplot as plt
N_s = 191 # include one terminal state when a player sticks
actions = ['hit','stick']
x_range = range(4,23)
y_range = range(2,12)
x_offset = 4
x_size = 19
y_size = 10
y_offset = 2
final_offset = 17
def calc_Py_g_hat():
    """calculate the probability dealer end in y given he started in y_hat"""
    M = 9 # bound on the steps
    vec = np.array([1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,4/13,1/13])
    A = np.zeros((20, 20))
    start_states = 10
    end_states = 6
    for i in range(15):
        for j in range(min(len(vec), 18 - i)):
            A[2 + i + j][i] = vec[j]
    final_probs = np.zeros((start_states, end_states))
    for y in range(start_states):
        prob = np.zeros((20,1))
        prob[y] = 1
        prob_sum = np.zeros((20,1))
        for m in range(M):
            prob = np.matmul(A,prob)
            prob_sum+=prob
        final_probs[y][:-1] = np.transpose(prob_sum[-(end_states-1):])
        final_probs[y][-1] = 1-np.sum(final_probs[y][:-1])
    return final_probs

def calc_prob_matrices():
    P_hit = np.zeros((19,19))
    vec = np.array([1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 4 / 13, 1 / 13])
    for i in range(19):
        for j in range(min(len(vec), 16 - i)):
            P_hit[2 + i + j][i] = vec[j]
        P_hit[-1][i] = 1- np.sum(P_hit[:-1,i])
    P_stick = np.zeros((19, 19))
    P_stick[-1,:] = 1
    return P_hit, P_stick

def get_reward_vectors():
    """calculate reward foreach (s,a)"""
    # calculate reward for action ="hit"
    P_hit, _ = calc_prob_matrices()
    P_final_g_y = calc_Py_g_hat()
    hit_reward = -P_hit[-1,:] #the probability to go bust times reward of -1
    hit_reward[-1] = 0 # terminal state no more reward
    hit_reward[-2] = 1 # reached 21 (natural)
    stick_reward = np.zeros((x_size,y_size))
    for x in range(x_size):
        for y in range(y_size):
            if (x+x_offset)<17:
                p_y_bust = P_final_g_y[y,-1]
                stick_reward[x,y] = p_y_bust - (1-p_y_bust)
            elif (x+x_offset) == 21:
                stick_reward[x, y] = 1
            elif (x+x_offset) == 22:
                stick_reward[x, y] = 0
            else:
                p_y_bust = P_final_g_y[y, -1]
                # reward = p(x=21) + p(y'>21) + p(x>y') - p(x<y')
                stick_reward[x,y] = p_y_bust + np.sum(P_final_g_y[y, :(x+x_offset-final_offset)]) -np.sum(P_final_g_y[y, (x+x_offset-final_offset+1):-1])
    """hit_reward_vector = np.zeros(N_s)
    for x in x_range:
        for y in y_range:
            if x <= 9:
                hit_reward_vector[x_size*(x-x_offset+y-y_offset)] = 0
            #elif:"""
    return hit_reward, stick_reward

def belman_equations(V_t,x,y):
    hit_reward, stick_reward = get_reward_vectors() # can be removed to the main
    P_hit, P_stick = calc_prob_matrices() # can be removed to the main

    expected_hit = hit_reward[x]
    for x_tp1 in range(x_size):
        expected_hit += P_hit[x_tp1,x]*V_t[x_tp1,y]

    expected_stick = stick_reward[x,y]
    return expected_hit, expected_stick




if __name__ == '__main__':
    """probs = calc_Py_g_hat()
    p_hit = calc_prob_matrices()
    V = np.zeros(N_s)
    r_a = get_reward_vectors
    P_a = calc_prob_matrices()
    while True: #TODO: set real condition
        values_of_actions = []
        for action in actions:
            values_of_actions.append(r_a + P_a@V)
        V_stack = np.stack((values_of_actions[0],values_of_actions[1]),axis=-1)
        V = np.max(V_stack,axis=-1)"""
    #calc_Py_g_hat()
    V_t = np.zeros((x_size, y_size))
    policy = np.zeros((x_size, y_size))
    V_tp1 = V_t
    error = 1
    eps = 1e-2
    for t in range(25):
        for x in range(x_size):
            for y in range(y_size):
                expected_hit, expected_stick = belman_equations(V_t, x, y)
                V_tp1[x, y] = max(expected_hit, expected_stick)
                if expected_hit>expected_stick:
                    policy[x, y] = 0
                else:
                    policy[x, y] = 1
        V_t = V_tp1

    x_axis = list(range(4, x_size + 4 - 1))
    y_axis = list(range(2, y_size + 2))
    x, y = np.meshgrid(x_axis, y_axis)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('$Player\'s$ $Sum$')
    ax.set_ylabel('$Dealer\'s$ $First$ $Card$')
    ax.set_zlabel('$Value$ $Function$')
    ax.set_xticks(list(range(4, 4 + x_size - 1, 2)))
    surf = ax.plot_surface(x, y, V_t[:-1, :].T, cmap="plasma")

    fig.colorbar(surf, shrink=0.5)
    ax.view_init(30, 120)
    plt.show()

    x_axis = list(range(4, x_size + 4 - 1))
    y_axis = list(range(2, y_size + 2))
    x, y = np.meshgrid(x_axis, y_axis)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('$Player\'s$ $Sum$')
    ax.set_ylabel('$Dealer\'s$ $First$ $Card$')
    ax.set_zlabel('$Value$ $Function$')
    ax.set_xticks(list(range(4, 4 + x_size - 1, 2)))
    surf = ax.plot_surface(x, y, V_t[:-1, :].T, cmap="plasma")

    fig.colorbar(surf, shrink=0.5)
    ax.view_init(90, 270)
    plt.show()

    min_value = []
    for j in range(y_size):
        for i in range(x_size):
            if policy[i, j] == 1:
                min_value.append(i + 4)
                break

    plt.plot(y_axis, min_value, 'o-', color='lightskyblue')
    plt.ylim((4, 21))
    plt.yticks(list(range(4, 4 + x_size - 1)))
    plt.xticks(list(range(2, 2 + y_size)))
    plt.fill_between(y_axis, min_value, color="pink", alpha=0.3)
    plt.fill_between(y_axis, min_value, 21, color='lightskyblue', alpha=0.3)
    plt.ylabel('$Player\'s$ $Sum$')
    plt.xlabel('$Dealer\'s$ $First$ $Card$')
    plt.title("$Hit/Stick$ $Policy$")
    plt.show()