# Abstraction alignment metrics

import numpy as np
from tqdm import tqdm
from scipy import stats
from itertools import combinations


def abstraction_match(fitted_abstractions, level, next_level=None):
    """Measures the abstraction match between level and next_level. Abstraction
    match measures the decreace in entropy between the nodes at level and the
    nodes at the next_level."""
    if next_level is None:
        next_level = level + 1
    level_entropies = []
    next_level_entropies = []
    
    for fitted_abstraction in fitted_abstractions:
        level_nodes = fitted_abstraction.filter_nodes(lambda x: fitted_abstraction.depth(x) == level)
        level_values = [node.data for node in level_nodes]
        level_entropies.append(stats.entropy(level_values))
        
        next_level_nodes = fitted_abstraction.filter_nodes(lambda x: fitted_abstraction.depth(x) == next_level)
        next_level_values = [node.data for node in next_level_nodes]
        next_level_entropies.append(stats.entropy(next_level_values))
        
    mean_level_entropy = np.mean(level_entropies)
    mean_next_level_entropy = np.mean(next_level_entropies)
    entropy_reduced = (mean_next_level_entropy - mean_level_entropy) / mean_next_level_entropy
    return entropy_reduced


def concept_coconfusion(fitted_abstractions, threshold, normalization=None):
    """Measures the concept coconfusion across a set of fitted abstractions.
    
    Args:
        fitted_abstractions: Either a list of DAGs containing nodes with values
            or a list of dictionaries mapping node ids to node values.
        threshold: Nodes with values lower than threshold will be igored.
        
    Returns:
        A dictionary mapping node pairs to their concept coconfusion.
    """
    coconfusion = {}
    for fitted_abstraction in tqdm(fitted_abstractions):
        
        if isinstance(fitted_abstraction, dict): # if fitted abstraction is are dictionary
            node_pairs = list(combinations([node_id for node_id, value in fitted_abstraction.items() if value >= threshold], 2))
            get_value = lambda node: fitted_abstraction[node]
        else: # if fitted abstraction is a tree
            node_pairs = list(combinations([node.identifier for node in fitted_abstraction.all_nodes() if node.data >= threshold], 2))
            get_value = lambda node: fitted_abstraction.get_node(node).data
            
        for pair in node_pairs:
            pair = sorted(pair)
            pair_name = f'{pair[0]},{pair[1]}'
            if pair_name not in coconfusion:
                coconfusion[pair_name] = 0
            value = stats.entropy([get_value(pair[0]), get_value(pair[1])])
            coconfusion[pair_name] += value  

    if normalization is None:
        normalization = stats.entropy([0.5,0.5]) * len(fitted_abstractions)
        
    for pair, value in coconfusion.items():
        coconfusion[pair] = value / normalization
    return coconfusion


def concept_coconfusion_inplace(outputs, abstraction_graph, propagate, threshold, value=None, normalization=None):
    """Measures the concept coconfusion by creating a set of fitted abstractions.
    
    Args:
        outputs: A list of output values corresponding to nodes in the abstraction graph.
        abstraction_graph: The abstraction graph to propagate through.
        propagate: Function that propagates outputs through the abstraction graph.
        threshold: Nodes with values lower than threshold will be igored.
        value: The value to add for every coconfusion. If None, then the entropy
            of the nodes' values will be added.
        normalization: The value to normalize the coconfusion by. If None, then
            it will be normalized by the maximum entropy.
        
    Returns:
        A dictionary mapping node pairs to their concept coconfusion.
    """
    coconfusion = {}
    for output in tqdm(outputs):
        fitted_abstraction = propagate(output, abstraction_graph)
        weighted_nodes = fitted_abstraction.filter_nodes(lambda n: n.data >= threshold)
        for pair in combinations(weighted_nodes, 2):
            pair = tuple(sorted(list(pair)))
            coconfusion.setdefault(pair, 0)
            if value is None:
                value = stats.entropy([fitted_abstraction.get_node(pair[0]).data, fitted_abstraction.get_node(pair[1]).data])
            coconfusion[pair] += value
            
    if normalization is None:
        normalization = stats.entropy([0.5,0.5]) * len(outputs)
        
    for pair, value in coconfusion.items():
        coconfusion[pair] = value / normalization
    return coconfusion
