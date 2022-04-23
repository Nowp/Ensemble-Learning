from math import comb
import numpy as np
import matplotlib.pyplot as plt


def precompute_s_k(n, k):
    n_strong = np.zeros(n-k+2, dtype=np.int32)
    for i, j in enumerate(range(n, k-1, -1)):
        n_strong[i+1] = comb(n, j) - n_strong[i]

    return np.flip(n_strong[1:])


def probability_a(majority, n, p_s, p_w):
    prob = 0
    nr_strong = precompute_s_k(n, majority)  # for efficiency only compute higher than majority
    for i, k in enumerate(range(majority, n+1)):
        s_occur = nr_strong[i]
        res = (comb(n, k)-s_occur) * np.power(p_w, k) * np.power(1 - p_w, n - k - 1) * (1 - p_s) + \
            s_occur * np.power(p_w, k-1) * p_s * np.power(1 - p_w, n-k)
        prob += res
    return 'final probability: {}'.format(prob)


def probability_b(w, n, p_s, p_w):
    total_votes = 10 + w
    majority = total_votes//2 + 1
    nr_strong = precompute_s_k(n, 0)
    prob_p1 = 0
    for k in range(majority, n+1):
        s_occur = nr_strong[k]
        res = (comb(n, k)-s_occur) * np.power(p_w, k) * np.power(1 - p_w, n - k - 1) * (1 - p_s)
        prob_p1 += res
    prob_p2 = 0
    for k in range(majority-w+1, n+1):
        s_occur = nr_strong[k]
        res = s_occur * np.power(p_w, k-1) * p_s * np.power(1 - p_w, n-k)
        prob_p2 += res
    return prob_p1 + prob_p2


def run_b():
    plt.figure(figsize=(8, 8))
    weights = np.arange(1, 13)
    probabilities = np.zeros(len(weights))
    for i, weight in enumerate(weights):
        probabilities[i] = probability_b(weight, 11, 0.8, 0.6)
    plt.plot(weights, probabilities, c='g')
    plt.xlabel('weight for strong classifier')
    plt.ylabel('probability of majority voting correctly')
    plt.title('Probabilities of majority voting correctly with different weights for strong classifier')
    plt.show()
    return 'best weight: {}'.format(weights[np.argmax(probabilities)])


def run_c():
    weights = np.ones(11)
    for i in range(11):
        if i == 0:
            weights[i] = np.log((1-0.2)/0.2)
        else:
            weights[i] = np.log((1-0.4)/0.4)
    normalize_weights = 1/weights[-1]
    weights *= normalize_weights
    return 'final weights: {}'.format(weights)


def run_d():
    plt.figure(figsize=(8, 8))
    error_rates = np.arange(0.1, 1, 0.1)
    weights = np.ones(len(error_rates))
    for i, err_rate in enumerate(error_rates):
        weights[i] = np.log((1 - err_rate) / err_rate)
    plt.plot(error_rates, weights, c='g')
    plt.xlabel('error rate')
    plt.ylabel('weight assigned by adaboost')
    plt.title('Weight assigned to base learner with differing error rates')
    plt.show()
    return 'View plot'


def main(mode):
    if mode == 'a':
        return probability_a(6, 11, 0.8, 0.6)

    elif mode == 'b':
        return run_b()

    elif mode == 'c':
        return run_c()

    elif mode == 'd':
        return run_d()
    return 'Choose one of the following: a,b,c,d'


exercise = 'd'
final_result = main(exercise)
print(f'final result after running exercise {exercise}: {final_result}')
