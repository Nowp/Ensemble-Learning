from math import comb  # only works for python 3.8 or newer
import numpy as np
import matplotlib.pyplot as plt


def probability(majority, n, p):
    prob = 0
    for k in range(majority, n+1):
        res = comb(n, k) * np.power(p, k) * np.power(1 - p, n - k)
        prob += res
    return prob


def prob_majority(jury_size, competence):
    majority = jury_size//2 + 1
    return probability(majority, jury_size, competence)


def main(mode):
    if mode == 'b':
        final_prob = prob_majority(19, 0.6)
        print(f'final probability = {final_prob}')

    elif mode == 'c':
        competences = [0.55, 0.6, 0.7, 0.8]
        jury_sizes = np.arange(5, 41, 2)
        plt.figure(figsize=(8, 8))
        for competence in competences:
            probabilities = np.zeros(len(jury_sizes))
            for i, jury in enumerate(jury_sizes):
                final_prob = prob_majority(jury, competence)
                probabilities[i] = final_prob
            plt.plot(jury_sizes, probabilities, label=f'p={competence}')
            plt.xlabel('jury_size')
            plt.ylabel('probability of majority voting correctly')
            plt.title('Probabilities of majority voting correctly with different competences and group sizes')
        plt.legend(loc=4)
        plt.show()

    elif mode == 'd':
        competence = 0.6
        jury_sizes = np.arange(19, 61, 2)
        probabilities = np.zeros(len(jury_sizes))
        plt.figure(figsize=(8, 8))
        for i, jury in enumerate(jury_sizes):
            probabilities[i] = prob_majority(jury, competence)
        plt.plot(jury_sizes, probabilities, 'g')
        plt.xlabel('jury_size')
        plt.ylabel('probability of majority voting correctly')
        plt.title(f'Probabilities of majority voting correctly with different group sizes for p = {competence}')
        plt.axhline(0.896, ls='--')
        closest_jury_size = np.interp(0.896, probabilities, jury_sizes)
        plt.axvline(closest_jury_size, ls='--')
        plt.figtext(0.491, 0.09, f'{closest_jury_size:.2f}', ha="center", va="center", fontsize=8,
                    bbox={'facecolor': 'b', 'alpha': 0.5})
        plt.show()
    else:
        print('invalid mode, only b, c, and d work')


main('b')
