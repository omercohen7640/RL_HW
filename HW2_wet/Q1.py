import numpy as np
N_s = 190
actions = ['hit','stick']

def calc_prob_matrix(action)

def get_reward_matrix():
    """calculate reward foreach (s,a)"""
if __name__ == '__main__':
    V = np.zeros(N_s)
    while True: #TODO: set real condition
        values_of_actions = []
        for action in actions:
            P_a = calc_prob_matrix(action)
            r_a= ?
            values_of_actions.append(r_a + P_a@V)
        V =