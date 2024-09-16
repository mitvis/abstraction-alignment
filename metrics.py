# Abstraction alignment metrics

import numpy as np
from tqdm import tqdm
from scipy import stats
from itertools import combinations


def abstraction_match(fitted_abstractions, level):
    level_entropies = []
    next_level_entropies = []
    
    for fitted_abstraction in fitted_abstractions:
        level_nodes = fitted_abstraction.filter_nodes(lambda x: fitted_abstraction.depth(x) == level)
        level_values = [node.data for node in level_nodes]
        level_entropies.append(stats.entropy(level_values))
        
        next_level_nodes = fitted_abstraction.filter_nodes(lambda x: fitted_abstraction.depth(x) == level+1)
        next_level_values = [node.data for node in next_level_nodes]
        next_level_entropies.append(stats.entropy(next_level_values))
        
    mean_level_entropy = np.mean(level_entropies)
    mean_next_level_entropy = np.mean(next_level_entropies)
    entropy_reduced = (mean_next_level_entropy - mean_level_entropy) / mean_next_level_entropy
    return entropy_reduced


def joint_entropy(fitted_abstractions, threshold):
    joint_entropy = {}
    for fitted_abstraction in tqdm(fitted_abstractions):
        node_pairs = list(combinations([node.identifier for node in fitted_abstraction.all_nodes() if node.data >= threshold], 2))
        for pair in node_pairs:
            pair = sorted(pair)
            pair_name = f'{pair[0]},{pair[1]}'
            if pair_name not in joint_entropy:
                joint_entropy[pair_name] = 0
            node_a = fitted_abstraction.get_node(pair[0])
            node_b = fitted_abstraction.get_node(pair[1])
            value = stats.entropy([node_a.data, node_b.data])
            joint_entropy[pair_name] += value
    return joint_entropy

