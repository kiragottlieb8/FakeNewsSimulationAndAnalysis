import numpy as np
import networkx as nx
import random
import itertools
import pandas as pd


N = 100
timesteps = 50
num_trials = 20

# values for each of the selected parameters
network_types = ['ER', 'BA', 'WS']
noise_levels = [0.05, 0.1, 0.15]
thresholds = [0.5,0.6, 0.7, 0.8,0.9]
trust_ranges = [(0.2,0.4), (0.4, 0.6), (0.6, .8), (.8,1)]
truth_fractions = [0.05, 0.1, 0.2, .3,.4,.5]

# Network structure params
p = .1
m = 3
k = 6
beta = 0.05
radius = 0.2
x, y = 1, 4


reliability_low = 0.5
reliability_high = 1.0

#creates the different types of network structures
def create_network(network_type):
    if network_type == 'ER':
        G = nx.erdos_renyi_graph(N, p)
    elif network_type == 'BA':
        G = nx.barabasi_albert_graph(N, m)
    elif network_type == 'WS':
        G = nx.watts_strogatz_graph(N, k, beta)
    return G

#sets up matricies for edges and weights(trust+reliability)
def initialize_matrices(G, trust_min, trust_max):
    adj_matrix = nx.to_numpy_array(G)
    trust_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if adj_matrix[i, j] == 1:
                trust_matrix[i, j] = np.random.uniform(trust_min, trust_max)
    reliability = np.random.uniform(reliability_low, reliability_high, size=N)
    return adj_matrix, trust_matrix, reliability

#uses bayesian inference formula derived in section 2
def bayesian_update(agent_idx, beliefs, trust_matrix, reliability, interactions, noise_prob):
    if round(beliefs[agent_idx]) == 1:
        return 1
    numerator = beliefs[agent_idx]
    denominator = 1 - beliefs[agent_idx]
    for i, b in interactions.items():
        if b:
            numerator *= trust_matrix[agent_idx][i] * reliability[i]
            denominator *= 1 - trust_matrix[agent_idx][i] * reliability[i]
        else:
            numerator *= 1 - trust_matrix[agent_idx][i] * reliability[i]
            denominator *= trust_matrix[agent_idx][i] * reliability[i]
    denominator += numerator
    updated = numerator / denominator
    noise = (random.random() * 2 - 1) * noise_prob
    updated_w_stochastic = max(0, min(updated + noise, 1))
    return updated_w_stochastic

#runs a single step of the simulation
def simulation_step(beliefs, trust_matrix, reliability, threshold, x, y, seen, noise_prob):
    new_beliefs = beliefs.copy()
    for i in range(N):
        neighbors = [j for j in range(N) if trust_matrix[i, j] > 0]
        if len(neighbors)==0:
            continue
        num_interactions = random.randint(x, min(len(neighbors), y))
        chosen = random.sample(neighbors, num_interactions)
        for j in chosen:
            seen[i][j] = bool(beliefs[j] >= threshold)
        updated_belief = bayesian_update(i, beliefs, trust_matrix, reliability, seen[i], noise_prob)
        new_beliefs[i] = 1.0 if updated_belief >= threshold else updated_belief
    return new_beliefs

#runs an entire trial of the simulation
def run_single_trial(network_type, noise_prob, threshold, trust_min, trust_max, truth_fraction):
    G = create_network(network_type)
    adj_matrix, trust_matrix, reliability = initialize_matrices(G, trust_min, trust_max)
    seen = [dict() for _ in range(N)]
    beliefs = np.full(N, 0.1)
    truth_indices = np.random.choice(N, size=int(N * truth_fraction), replace=False)
    beliefs[truth_indices] = 1.0

    for t in range(timesteps):
        beliefs = simulation_step(beliefs, trust_matrix, reliability, threshold, x, y, seen, noise_prob)
    return beliefs



all_results = []

#runs all trials for all combinations of selected parameters and saves them to a csv file
param_combinations = list(itertools.product(network_types, noise_levels, thresholds, trust_ranges, truth_fractions))
for combo in param_combinations:
    network_type, noise_prob, threshold, trust_range, truth_fraction = combo
    trust_min, trust_max = trust_range
    final_fractions = []
    persistent_fractions = []

    for trial in range(num_trials):
        final_beliefs = run_single_trial(network_type, noise_prob, threshold, trust_min, trust_max, truth_fraction)
        final_fraction = (final_beliefs >= threshold).mean()
        persistent_fraction = (final_beliefs < threshold).mean()
        final_fractions.append(final_fraction)
        persistent_fractions.append(persistent_fraction)

    all_results.append({
        'network_type': network_type,
        'noise': noise_prob,
        'threshold': threshold,
        'trust_min': trust_min,
        'trust_max': trust_max,
        'truth_fraction': truth_fraction,
        'mean_persistent': np.mean(persistent_fractions),
        'std_persistent': np.std(persistent_fractions)
    })



df_results = pd.DataFrame(all_results)

df_results.to_csv("stats.csv")