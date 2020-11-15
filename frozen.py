import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

SEED = 903454028
np.random.seed(SEED)

def run_VI(env, id=None):
    gammas = [0.99, 0.9, 0.8, 0.7, 0.5]
    R = []
    iters = []
    tracker = ''
    for g in gammas:
        P, iter, runtime = value_iteration(env, g)
        Rs, steps = test_policy(env, P)
        mean_R = np.mean(Rs)
        mean_steps = np.mean(steps)
        R.append(mean_R)
        iters.append(mean_steps)
        tracker += 'gamma={}: mean R was {}, mean_steps was {}\n'.format(g, mean_R, mean_steps)

    with open('figures/VI_variables_lake_{}.txt'.format(id), 'w') as f:
        f.write(tracker)

    # plot mean iters and Rs
    plt.plot(gammas, R, label='Mean rewards')
    plt.title('VI mean rewards')
    plt.xlabel('gammas')
    plt.ylabel('Mean rewards')
    plt.savefig('figures/VI_rewards_lake_{}.png'.format(id))
    plt.clf()

    plt.plot(gammas, iters)
    plt.title('VI mean iterations')
    plt.xlabel('gammas')
    plt.ylabel('Mean iterations')
    plt.savefig('figures/VI_iters_lake_{}.png'.format(id))
    plt.clf()
    print('Done')

def run_PI(env, id=None):
    gammas = [0.99, 0.9, 0.8, 0.7, 0.5]
    R = []
    iters = []
    tracker = ''
    for g in gammas:
        P, iter, runtime = policy_iteration(env, g)
        Rs, steps = test_policy(env, P)
        mean_R = np.mean(Rs)
        mean_steps = np.mean(steps)
        R.append(mean_R)
        iters.append(mean_steps)
        tracker += 'gamma={}: mean R was {}, mean_steps was {}\n'.format(g, mean_R, mean_steps)

    with open('figures/PI_variables_lake_{}.txt'.format(id), 'w') as f:
        f.write(tracker)

    # plot mean iters and Rs
    plt.plot(gammas, R, label='Mean rewards')
    plt.title('PI mean rewards')
    plt.xlabel('gammas')
    plt.ylabel('Mean rewards')
    plt.savefig('figures/PI_rewards_lake_{}.png'.format(id))
    plt.clf()

    plt.plot(gammas, iters)
    plt.title('PI mean iterations')
    plt.xlabel('gammas')
    plt.ylabel('Mean iterations')
    plt.savefig('figures/PI_iters_lake_{}.png'.format(id))
    plt.clf()
    print('Done')

def run_Q(env, id=None):
    lrs = [0.01, 0.1]
    min_epsilons = [0.1, 0.01]
    gammas = [0.99, 0.7]
    tracker = ''
    params = []
    R_over = []
    iters = []
    for lr in lrs:
        for me in min_epsilons:
            for g in gammas:
                Q, runtime, R = Q_learner(env, min_epsilon=me, lr=lr, gamma=g)
                Rs, steps = test_policy(env, np.argmax(Q, axis=1))
                mean_R = np.mean(Rs)
                mean_steps = np.mean(steps)
                params.append('lr={}, min_ep={}, gamma={}'.format(lr, me, g))
                R_over.append(mean_R)
                iters.append(mean_steps)
                tracker += 'lr={}, min_ep={}, gamma={}: mean R was {}, mean_steps was {}\n'.format(lr, me, g, mean_R, mean_steps)

    with open('figures/Q_variables_lake_{}.txt'.format(id), 'w') as f:
        f.write(tracker)

    # plot
    plt.plot(params, R_over)
    plt.title('Q learning mean rewards')
    plt.ylabel('Mean rewards')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/Q_rewards_lake_{}.png'.format(id))
    plt.clf()

    plt.plot(params, iters)
    plt.title('Q learning mean iterations')
    plt.ylabel('Mean iterations')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('figures/Q_iters_lake_{}.png'.format(id))
    plt.clf()
    print('Done')

def test_policy(env, P, epochs=1000, id=None):
    R_total = []
    steps_solved = []
    for i in range(epochs):
        s = env.reset()
        steps = 0
        epoch_r = 0
        while steps < 10000:
            a = int(P[s])
            s_next, R, done, _ = env.step(a)
            epoch_r += R
            s = s_next
            steps += 1
            if done:
                break
        R_total.append(epoch_r)
        steps_solved.append(steps)
    return R_total, steps_solved

def value_iteration(env, gamma=0.99, theta=0.001):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    V_old = V.copy()
    P = np.zeros(n_states)
    iters = 0
    delta = 1
    start = time.time()
    while delta > theta:
        iters += 1
        for s in range(n_states):
            q_best = np.NINF
            for a in range(n_actions):
                total_q = 0
                for prob, s_next, R, done in env.P[s][a]:
                    v = V_old[s_next]
                    if done:
                        q = R
                    else:
                        q = R + gamma*v
                    total_q += q*prob
                if total_q > q_best:
                    q_best = total_q
                    P[s] = a
                    V[s] = q_best
        delta = np.max(np.abs(V - V_old))
        V_old = V.copy()
    end = time.time()
    runtime = end - start
    print('VI done in {} sec and {} iterations'.format(runtime, iters))
    return P, iters, runtime

def policy_iteration(env, gamma=0.99, theta=0.001, max_steps=1000):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)
    P = np.random.randint(0, n_actions, size=n_states)
    iters = 0
    start = time.time()
    while True:
        iters += 1
        while True:
            delta = 0
            for s in range(n_states):
                v = V[s]
                a = P[s]
                q_total = 0
                for prob, s_next, R, done in env.P[s][a]:
                    if done:
                        q = R
                    else:
                        q = R + gamma*V[s_next]
                    q_total += q*prob
                V[s] = q_total
                delta = max(delta, np.abs(v - V[s]))
            if delta < theta:
                break
        policy_stable = True
        for s in range(n_states):
            old_a = P[s]
            best_q = np.NINF
            for a in range(n_actions):
                total_q = 0
                for prob, s_next, R, done in env.P[s][a]:
                    if done:
                        q = R
                    else:
                        q = R + gamma*V[s_next]
                    total_q += q*prob
                if total_q > best_q:
                    best_q = total_q
                    P[s] = a
            if old_a != P[s]:
                policy_stable = False
        if policy_stable:
            break
    end = time.time()
    runtime = end - start
    print('PI done in {} sec and {} iterations'.format(runtime, iters))
    return P, iters, runtime

def Q_learner(env, episodes=10000, min_epsilon=0.01, lr=0.1, gamma=0.99):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    e_decay = 1/episodes
    epsilon = 1
    max_epsilon = 1
    Q = np.zeros((n_states, n_actions))
    start = time.time()
    Rs = []
    for e in range(episodes):
        s = env.reset()
        R = 0
        t = 0
        while True:
            if np.random.uniform(0, 1) < epsilon:
                a = np.random.randint(n_actions)
            else:
                # a = np.argmax(Q[s, :])
                b = Q[s, :]
                a = np.random.choice(np.where(b==b.max())[0])

            s_next, r, done, _ = env.step(a)
            R += r
            # update Q
            if not done:
                Q[s, a] = Q[s, a] + lr*(r + gamma*np.max(Q[s_next, :]) - Q[s, a])
            else:
                Q[s, a] = Q[s, a] + lr*(r - Q[s, a])

            s = s_next
            t += 1
            if done:
                break
        # epsilon decay
        epsilon = max(max_epsilon - e_decay*e, min_epsilon)
        Rs.append(R)
    end = time.time()
    runtime = end - start
    print('Q-learner done in {} sec and avg R of {} achieved'.format(runtime, np.mean(Rs)))
    return Q, runtime, np.mean(Rs)

if __name__ == '__main__':
    if not os.path.isdir('figures/'):
        os.mkdir('figures')

    # random_map = generate_random_map(size=16)
    env = gym.make('FrozenLake-v0', map_name='8x8')

    x = input('''Choose algorithm:
                    VI: v,
                    PI: p,
                    Q: q: ''')

    y = input('''Choose lake size:
                    4x4: 4,
                    8x8: 8,
                    16x16: 16: ''')

    if y == '4':
        env = gym.make('FrozenLake-v0')
        if x == 'v':
            run_VI(env, id=y)
        elif x == 'p':
            run_PI(env, id=y)
        elif x == 'q':
            run_Q(env, id=y)
    elif y == '8':
        env = gym.make('FrozenLake-v0', map_name='8x8')
        if x == 'v':
            run_VI(env, id=y)
        elif x == 'p':
            run_PI(env, id=y)
        elif x == 'q':
            run_Q(env, id=y)
    elif y == '16':
        random_map = generate_random_map(size=16)
        env = gym.make('FrozenLake-v0', desc=random_map)
        if x == 'v':
            run_VI(env, id=y)
        elif x == 'p':
            run_PI(env, id=y)
        elif x == 'q':
            run_Q(env, id=y)
