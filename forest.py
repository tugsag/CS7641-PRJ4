import gym
import numpy as np
import pandas as pd
from hiive.mdptoolbox.example import forest
from hiive.mdptoolbox.mdp import PolicyIteration, ValueIteration, QLearning
import matplotlib.pyplot as plt
import time
import os

SEED = 903454028
np.random.seed(SEED)

def test_policy(P, R, policy, g_decay=0.99, epochs=1000):
    n_states = P.shape[-1]
    R_total = 0
    for s in range(n_states):
        R_s = 0
        for e in range(epochs):
            R_e = 0
            gamma = 1
            while True:
                a = policy[s]
                probs = P[a][s]
                cands = list(range(len(probs)))
                s_next = np.random.choice(cands, 1, p=probs)[0]
                r = R[s][a] * gamma
                R_e += r
                gamma *= g_decay
                if s_next == 0:
                    break
            R_s += R_e
        R_total += R_s
    return R_total/(n_states * epochs)

def value_iteration(P, R, id=None):
    # gammas = [0.99, 0.9, 0.8, 0.7, 0.5]
    epsilons = [.1, .01, .001, .00001]
    runtimes = []
    Rs = []
    iters = []
    tracker = ''
    policies = []
    for e in epsilons:
        vi = ValueIteration(P, R, 0.9, epsilon=e, max_iter=100000)
        vi.run()
        r = test_policy(P, R, vi.policy)
        Rs.append(r)
        policies.append(vi.policy)
        runtimes.append(vi.time)
        iters.append(vi.iter)
        tracker += 'epsilon={}: reward was {}, iters was {}, time was {}\n'.format(e, r, vi.iter, vi.time)

    # write
    with open('figures/VI_variables_forest_{}.txt'.format(id), 'w') as f:
        f.write(tracker)

    with open('figures/VI_policies_forest_{}.txt'.format(id), 'w') as f:
        for i, e in enumerate(epsilons):
            f.write('epsilon={}: policy={}\n'.format(e, policies[i]))

    # plot
    plt.plot(epsilons, Rs)
    plt.title('VI Avg Rewards')
    plt.xlabel('Epsilons')
    plt.ylabel('Avg rewards')
    plt.savefig('figures/VI_rewards_forest_{}.png'.format(id))
    plt.clf()

    plt.plot(epsilons, iters)
    plt.title('VI iterations')
    plt.xlabel('Epsilons')
    plt.ylabel('Iterations')
    plt.savefig('figures/VI_iters_forest_{}.png'.format(id))
    plt.clf()
    # -----------------------------------
    gammas = [0.99, 0.9, 0.8, 0.7, 0.5]
    # epsilons = [.1, .01, .001, .00001]
    runtimes = []
    Rs = []
    iters = []
    tracker = ''
    policies = []
    for g in gammas:
        vi = ValueIteration(P, R, g, epsilon=0.01, max_iter=100000)
        vi.run()
        r = test_policy(P, R, vi.policy)
        Rs.append(r)
        policies.append(vi.policy)
        runtimes.append(vi.time)
        iters.append(vi.iter)
        tracker += 'gamma={}: reward was {}, iters was {}, time was {}\n'.format(g, r, vi.iter, vi.time)

    # write
    with open('figures/VI_variables_forest_{}_g.txt'.format(id), 'w') as f:
        f.write(tracker)

    with open('figures/VI_policies_forest_{}_g.txt'.format(id), 'w') as f:
        for i, g in enumerate(gammas):
            f.write('gamma={}: policy={}\n'.format(g, policies[i]))

    # plot
    plt.plot(gammas, Rs)
    plt.title('VI Avg Rewards')
    plt.xlabel('Gammas')
    plt.ylabel('Avg rewards')
    plt.savefig('figures/VI_rewards_forest_{}_g.png'.format(id))
    plt.clf()

    plt.plot(gammas, iters)
    plt.title('VI iterations')
    plt.xlabel('Gammas')
    plt.ylabel('Iterations')
    plt.savefig('figures/VI_iters_forest_{}_g.png'.format(id))
    plt.clf()
    print('done')

def policy_iteration(P, R, id=None):
    gammas = [0.99, 0.9, 0.8, 0.7, 0.5]
    runtimes = []
    Rs = []
    iters = []
    tracker = ''
    policies = []
    for g in gammas:
        pi = PolicyIteration(P, R, g, max_iter=100000)
        pi.run()
        r = test_policy(P, R, pi.policy)
        Rs.append(r)
        policies.append(pi.policy)
        runtimes.append(pi.time)
        iters.append(pi.iter)
        tracker += 'gamma={}: reward was {}, iters was {}, time was {}\n'.format(g, r, pi.iter, pi.time)

    # write
    with open('figures/PI_variables_forest_{}.txt'.format(id), 'w') as f:
        f.write(tracker)

    with open('figures/PI_policies_forest_{}.txt'.format(id), 'w') as f:
        for i, g in enumerate(gammas):
            f.write('gamma={}: policy={}\n'.format(g, policies[i]))

    # plot
    plt.plot(gammas, Rs)
    plt.title('PI Avg Rewards')
    plt.xlabel('Gammas')
    plt.ylabel('Avg rewards')
    plt.savefig('figures/PI_rewards_forest_{}.png'.format(id))
    plt.clf()

    plt.plot(gammas, iters)
    plt.title('PI iterations')
    plt.xlabel('Gammas')
    plt.ylabel('Iterations')
    plt.savefig('figures/PI_iters_forest_{}.png'.format(id))
    plt.clf()
    print('done')


def Q_learner(P, R, id=None):
    alpha_mins = [0.0001, 0.01]
    epsilon_decays = [0.99, 0.999]
    gammas = [0.99, 0.7]
    tracker = ''
    Rs = []
    runtimes = []
    params = []
    policies = []
    for g in gammas:
        print(g)
        for ed in epsilon_decays:
            for am in alpha_mins:
                start = time.time()
                Q = QLearning(P, R, gamma=g, alpha_min=am, epsilon_decay=ed, n_iter=10000000)
                Q.run()
                end = time.time()
                runtimes.append(end - start)
                r = test_policy(P, R, Q.policy)
                Rs.append(r)
                policies.append(Q.policy)
                params.append('gamma={}, a_min={}, e_dec={}'.format(g, am, ed))
                tracker += 'gamma={}, alpha_min={}, eplison_dec={}: reward was {}, time was {}\n'.format(g, am, ed, r, end-start)

    # write
    with open('figures/Q_variables_forest_{}_mil.txt'.format(id), 'w') as f:
        f.write(tracker)

    with open('figures/Q_policies_forest_{}_mil.txt'.format(id), 'w') as f:
        for i, p in enumerate(params):
            f.write('{}: policy={}'.format(p, policies[i]))

    # plot
    plt.plot(params, Rs)
    plt.title('Q learning params avg reward')
    plt.ylabel('Avg rewards')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/Q_rewards_forest_{}_mil.png'.format(id))
    plt.clf()

    plt.plot(params, runtimes)
    plt.title('Q learning runtimes')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/Q_runtimes_forest_{}_mil.png'.format(id))
    plt.clf()
    print('done')


if __name__ == '__main__':
    if not os.path.isdir('figures/'):
        os.mkdir('figures')

    y = input('''Choose environment:
                    S=250, r1=10, r2=5: 1,
                    S=1000, r1=15, r2=5: 2: ''')

    x = input('''Choose algorithm:
                    VI: v,
                    PI: p,
                    Q: q: ''')

    if y == '1':
        P, R = forest(S=250, r1=10, r2=5)
        if x == 'v':
            value_iteration(P, R, id='250')
        elif x == 'p':
            policy_iteration(P, R, id='250')
        elif x == 'q':
            Q_learner(P, R, id='250')
    elif y == '2':
        P, R = forest(S=1000, r1=15, r2=5)
        if x == 'v':
            value_iteration(P, R, id='1000')
        elif x == 'p':
            policy_iteration(P, R, id='1000')
        elif x == 'q':
            Q_learner(P, R, id='1000')
