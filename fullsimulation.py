import numpy as np
import networkx as nx
import random
import itertools
import pandas as pd
import collections
from tqdm import tqdm  

N = 100 #num agents
timesteps = 100 #simulation length/time
num_trials = 15 

#various variables tested to see effects on persistence
network_types = ['ER', 'BA', 'WS']
noise_levels = [0.05, 0.2]
thresholds = [0.5, 0.6, 0.7]
trust_ranges = [(0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, .9)]
truth_fractions = [0.1, 0.3, 0.5, 0.7]
#reliability was kept constant for my experiment, but generall you can give a range (low, high)
reliability_low = 1
reliability_high = 1

# network hyperparameters
p = 0.15 #for ER
m = 3 #for BA
k = 8 #for WS
beta = 0.1 #for WS

memorySize = 50 #remembers the last X number of interactions
alpha = 0.2 #arbitrary weight used in calculating which neighbors to visit (scales the importance of the # of nodes/degree a neighbor has)
bias_factor = 0.1 #represents confirmation bias (adds more or less weight to evidence in bayesian update) 

trust_delta_up = 0.02 #how much trust between 2 neighbors goes up after an interaction (if they agree)
trust_delta_down = 0.03 #how much trust between 2 neighbors goes up after an interaction (if they disagree)


NUM_NODES_TO_BOOST = 10  #this is for the number of nodes to apply the intervention to

x, y = 1, 4 # lower and upper bound of how many interactions an agent has at every timestep

#creates the different networks with networkx
def create_network(network_type):
    if network_type == 'ER':
        return nx.erdos_renyi_graph(N, p)
    elif network_type == 'BA':
        return nx.barabasi_albert_graph(N, m)
    else:
        return nx.watts_strogatz_graph(N, k, beta)

#clamps arrays to make sure beliefs and probabilites stay within the range of 0-1
def clamp_arr(a, lo=0.0, hi=1.0):
    return np.minimum(np.maximum(a, lo), hi)

"""
    chooses a given number of neighbors to interact with for a given agent (at a given timestep)
    takes into account both trust and degree of neighboring agents, giving those with higher trust
    and higher degree more likelihood to be interacted with(to model interactions in real life where 
    you usually talk to people you trust more or people who are popular) 
    Additionally, after interventions kick in, this gives a boost to the nodes affected by the intervention,
    making them more likely to be interacted with
"""
def sample_neighbors(i, neighbors, degree_norm, num_samples, trust_mat, alpha=0.2, visibility_boost=None ):
    
    weights = np.array([trust_mat[i][nb] for nb in neighbors])
    weights = np.clip(weights, 1e-12, None)
    weights = weights * (1.0 + alpha * np.array([degree_norm[nb] for nb in neighbors]))

    if visibility_boost is not None:
        boost_array = np.array([visibility_boost.get(nb, 1.0) for nb in neighbors])
        weights *= boost_array

    probs = weights / weights.sum()
    return list(np.random.choice(neighbors, size=num_samples, replace=False, p=probs))

"""
    creates an adjacency matrix to track the network
    creates a trust matrix to track trust between neighbors
    creates a reliability array (for the purposes of this experiment, they are all just 1 
    but this is left here to allow for customization) 
"""
def initialize_matrices(G, trust_min, trust_max):
    adj_matrix = nx.to_numpy_array(G)
    trust_matrix = np.zeros((N, N))
    for i, j in zip(*np.where(adj_matrix == 1)):
        trust_matrix[i, j] = np.random.uniform(trust_min, trust_max)
    reliability = np.random.uniform(reliability_low, reliability_high, size=N)
    return adj_matrix, trust_matrix, reliability

"""
    uses a bayesian inference formula based on the derivation from 
    the research paper (see research paper for more info)
    modified to be in log space to make numbers more precise and 
    to avoid truncation (and the limitations that come with it) 
"""
def bayesian_update(agent_idx, prior, seen_deque, trust_matrix, reliability, threshold, bias_factor, noise_prob):
    counts = {}
    for j, sig in seen_deque:
        if j not in counts:
            counts[j] = [0, 0]
        if sig:
            counts[j][0] += 1
        else:
            counts[j][1] += 1
    if len(counts) == 0:
        noise = (random.random()*2-1) * noise_prob
        return clamp_arr(prior + noise)
    
    log_PEH = log_PEnotH = 0.0
    expressed_sign = 1 if prior >= threshold else 0
    for j, (s_true, s_false) in counts.items():
        a = trust_matrix[agent_idx, j] * reliability[j]
        neighbor_sign = 1 if s_true >= s_false else 0
        a *= (1 + bias_factor) if neighbor_sign == expressed_sign else (1 - bias_factor)
        a = min(max(a, 1e-6), 1-1e-6)
        log_PEH += s_true*np.log(a) + s_false*np.log(1-a)
        log_PEnotH += s_true*np.log(1-a) + s_false*np.log(a)
    log_prior = np.log(max(min(prior,1-1e-12),1e-12))
    log_one_minus = np.log(max(min(1-prior,1-1e-12),1e-12))
    log_num = log_prior + log_PEH
    log_den = np.logaddexp(log_num, log_one_minus + log_PEnotH)
    updated = np.exp(log_num - log_den)
    noise = (random.random()*2-1)*noise_prob
    return clamp_arr(updated + noise)

"""
    runs a single trial of the experiment
    at time 50, it breaks into 3 separate simulations to model the baseline and 2 interventions
    separate lists and arrays are kept for each separate simulation 
    at each timestep, neighbors are randomly selected, then 
"""
def run_trial(network_type, noise_prob, threshold, trust_min, trust_max, truth_fraction):
    G = create_network(network_type)
    adj_matrix, trust_base, reliability = initialize_matrices(G, trust_min, trust_max)
    degrees = np.array([d for _, d in G.degree()])
    degree_norm = degrees / max(1, degrees.max())

    neighbors_list = [list(np.where(adj_matrix[i] != 0)[0]) for i in range(N)]
    seen_history_base = [collections.deque(maxlen=memorySize) for _ in range(N)]
    beliefs = np.full(N, 0.1)
    truth_indices = np.random.choice(N, size=max(1, int(N*truth_fraction)), replace=False)
    beliefs[truth_indices] = 1.0

    beliefs_degree = beliefs.copy()
    beliefs_betweenness = beliefs.copy()

    seen_degree = None
    seen_betweenness = None

    baseline_50 = None
    baseline_100 = None
    degree_100 = None
    betweenness_100 = None

    top_degree_indices = []
    top_betweenness_indices = []

    for t in range(timesteps):
        expressed = np.array([1 if b >= threshold else 0 for b in beliefs])
        new_beliefs = beliefs.copy() 
        for i in range(N):
            neighbors = neighbors_list[i]
            if len(neighbors) == 0: continue
            num_interactions = min(len(neighbors), random.randint(x, y))
 
            chosen = sample_neighbors(i, neighbors, degree_norm, num_interactions, trust_base, alpha)
            for j in chosen:
                seen_history_base[i].append((j, bool(expressed[j])))
            updated = bayesian_update(i, beliefs[i], seen_history_base[i], trust_base, reliability, threshold, bias_factor, noise_prob)
            new_beliefs[i] = max(updated, beliefs[i], threshold if beliefs[i] >= threshold else 0.0)
            for j in chosen:
                trust_base[i, j] = min(1.0, trust_base[i, j]+trust_delta_up) if bool(expressed[j]) == (new_beliefs[i] >= threshold) else max(0.0, trust_base[i, j]-trust_delta_down)

        beliefs = new_beliefs

        
        #right before the interventions begin, save the current data
        if t == 49:
            baseline_50 = np.sum(beliefs < threshold)

            # preparation for interventions by finding top nodes and boosting them
            top_degree_indices = [n for n,_ in sorted(G.degree, key=lambda x:x[1], reverse=True)[:NUM_NODES_TO_BOOST]]
            betw_centrality = nx.betweenness_centrality(G)
            top_betweenness_indices = sorted(betw_centrality, key=betw_centrality.get, reverse=True)[:NUM_NODES_TO_BOOST]
            for n in top_degree_indices:
                beliefs_degree[n] = 1.0
            for n in top_betweenness_indices:
                beliefs_betweenness[n] = 1.0
            degree_boost = {n: 2.0 for n in top_degree_indices}
            betweenness_boost = {n: 2.0 for n in top_betweenness_indices}



            #copying current values into the interventions before they split
            seen_degree = [collections.deque(sh, maxlen=memorySize) for sh in seen_history_base]
            seen_betweenness = [collections.deque(sh, maxlen=memorySize) for sh in seen_history_base]
            trust_degree = trust_base.copy()
            trust_betweenness = trust_base.copy()
            

        #intervention timesteps
        if t >= 50:
            
            for beliefs_array, boost_dict, history, trust_mat in [
                (beliefs_degree, degree_boost, seen_degree, trust_degree),
                (beliefs_betweenness, betweenness_boost, seen_betweenness, trust_betweenness)
            ]:
                expressed_set = np.array([1 if b >= threshold else 0 for b in beliefs_array])
                new_beliefs_array = beliefs_array.copy()
                for i in range(N):
                    neighbors = neighbors_list[i]
                    if len(neighbors) == 0: continue
                    num_interactions = random.randint(x, min(len(neighbors), y))
                    chosen = sample_neighbors(i, neighbors, degree_norm, num_interactions, trust_mat, alpha, boost_dict)
                    for j in chosen:
                        history[i].append((j, bool(expressed_set[j])))
                    updated = bayesian_update(i, beliefs_array[i], history[i], trust_mat, reliability, threshold, bias_factor, noise_prob)
                    new_beliefs_array[i] = max(updated, beliefs_array[i], threshold if beliefs_array[i] >= threshold else 0.0)
                    for j in chosen:
                        trust_mat[i, j] = min(1.0, trust_mat[i, j]+trust_delta_up) if bool(expressed_set[j]) == (new_beliefs_array[i] >= threshold) else max(0.0, trust_mat[i, j]-trust_delta_down)

                if beliefs_array is beliefs_degree:
                    beliefs_degree = new_beliefs_array
                else:
                    beliefs_betweenness = new_beliefs_array

           
        #saving final data values
        if t == timesteps - 1:
            baseline_100 = np.sum(beliefs < threshold)
            degree_100 = np.sum(beliefs_degree < threshold)
            betweenness_100 = np.sum(beliefs_betweenness < threshold)

    return baseline_50, baseline_100, degree_100, betweenness_100


all_results = []
#creating all combinations of parameters to test
param_combinations = list(itertools.product(network_types, noise_levels, thresholds, trust_ranges, truth_fractions))

#tqdm just shows a progress bar for the simulation. this makes it easier to run and test
#this runs the simulation 15 times for each combination and saves all of the data into a CSV
for combo in tqdm(param_combinations, desc="Parameter combinations"):
    network_type, noise_prob, threshold, trust_range, truth_fraction = combo
    trust_min, trust_max = trust_range

    baseline_50_list, baseline_100_list, degree_100_list, betweenness_100_list = [],[],[],[]

    for trial in tqdm(range(num_trials), desc="Trials", leave=False):
        b50, b100, d100, btw100 = run_trial(network_type, noise_prob, threshold, trust_min, trust_max, truth_fraction)
        baseline_50_list.append(b50)
        baseline_100_list.append(b100)
        degree_100_list.append(d100)
        betweenness_100_list.append(btw100)

    #see data dictionary to see explanations for each column
    all_results.append({
        'network_type': network_type,
        'noise': noise_prob,
        'threshold': threshold,
        'trust_min': trust_min,
        'trust_max': trust_max,
        'truth_fraction': truth_fraction,
        'baseline_50': int(np.mean(baseline_50_list)),
        'baseline_100': int(np.mean(baseline_100_list)),
        'degree_100': int(np.mean(degree_100_list)),
        'betweenness_100': int(np.mean(betweenness_100_list)),
        'baseline_50_std': int(np.std(baseline_50_list)),
        'baseline_100_std': int(np.std(baseline_100_list)),
        'degree_100_std': int(np.std(degree_100_list)),
        'betweenness_100_std': int(np.std(betweenness_100_list))
    })

df_results = pd.DataFrame(all_results)

#change the CSV name here to save to your preferred file name
df_results.to_csv("stats_interventions_centralityBugFixAll1.csv", index=False) 