import numpy as np


def q2():
    d = {}

    def psy(N,x):
        if N == x or N == 1:
            return 1
        elif N > x:
            return 0
        else:
            rec1 = d[(N-1, x-1)] if (N-1, x-1) in d else psy(N-1, x-1)
            d[(N-1, x-1)] = rec1
            rec2 = d[(N ,x-1)] if (N, x-1) in d else psy(N, x-1)
            d[(N, x - 1)] = rec2
            return rec1 + rec2
    print(psy(12,800))


def q3():
    N = 5
    B = 0
    letter_dict = {0: 'b', 1: 'k', 2: 'o', 3:'-'}
    num_dict = {'b': 0, 'k': 1, 'o': 2, '-': 3}
    p = np.array([[0.1, 0.325, 0.25, 0.325], [0.4, 0, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4], [1, 0, 0, 0]])
    paths = [{} for i in range(N)]
    logp = np.log(p)
    N_letters = p.shape[0]
    mat = -np.inf*np.ones((N, N_letters-1))
    for i in range(N-1,-1,-1):
        if i == N-1:
            for k in range(N_letters-1):
                mat[i, k] = logp[:, N_letters-1][k]
                paths[i][k] = 3
        elif i == 0:
            max_1 = -np.inf
            next_letter = 0
            for k in range(N_letters-1):
                if logp[B,k] + mat[i+1,k] > max_1:
                    max_1 = logp[B,k] + mat[i+1,k]
                    next_letter = k
            mat[i, B] = max_1
            paths[i][0] = next_letter

        else:
            for j in range(N_letters-1):
                max_j = -np.inf
                next_letter = 0
                for k in range(N_letters-1):
                    if logp[j, k] + mat[i+1, k] > max_j:
                        max_j = logp[j, k] + mat[i+1, j]
                        next_letter = k
                mat[i, j] = max_j
                paths[i][j] = next_letter
    word = ['b']
    for i in range(N):
        word += letter_dict[paths[i][num_dict[word[-1]]]]
    str = ""
    print(str.join(word))

q3()

