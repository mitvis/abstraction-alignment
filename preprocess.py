"""
This script processes all datasets in the 'data/' directory.

For each folder in the 'data/' directory, it creates additional data files
needed to run the abstraction alignment interface.
"""


import json
import os
from itertools import combinations
from multiprocessing import Pool, cpu_count
import time

import argparse
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm


class Node:
    """
    A class to represent a node in a tree.

    Attributes:
    id (int): The ID of the node.
    parent (Node): The parent of the node.
    value (int): The value of the node.
    children (list): The children of the node.

    Methods:
    add_child(child): Add a child to the node.
    """
    def __init__(self, id, parent=None, value=0):
        """
        Initialize a Node object.

        Parameters:
        id (int): The ID of the node.
        parent (Node): The parent of the node. Defaults to None.
        value (int): The value of the node. Defaults to 0.
        """
        self.id = id
        self.parent = parent
        self.value = value
        self.children = []

    def add_child(self, child):
        """
        Add a child to the node.

        Parameters:
        child (Node): The child to add.

        Returns:
        None
        """
        self.children.append(child)

    def __repr__(self):
        """
        Return a string representation of the node.

        Returns:
        str: A string representation of the node.
        """
        return f'Node {self.id} (Parent: {self.parent}, Value: {self.value}, Num Children: {len(self.children)})'


class Tree:
    """
    A class to represent a tree of nodes.

    Attributes:
    nodes (dict): A dictionary mapping node IDs to Node objects.
    root_id (int): The ID of the root node.

    Methods:
    reset(): Reset the value of each node in the tree to 0.
    propagate(scores): Assign scores to the leaf nodes and propagate them up the tree.
    """
    def __init__(self, hierarchy):
        """
        Initialize a Tree object.

        Parameters:
        hierarchy (list): A list of dictionaries, each representing a node with 'id' and 'parent' keys.
        """
        self.nodes = {}
        self.root_id = None

        # Initialize very node in the tree
        for node in hierarchy:
            if 'parent' not in node:
                node['parent'] = None
            if node['parent'] is None:
                self.root_id = node['id']
            self.nodes[node['id']] = Node(node['id'], node['parent'], 0)

        # Add the children to each node
        for node_id, node in self.nodes.items():
            if node.parent is not None:
                self.nodes[node.parent].add_child(node)

    def reset(self):
        """
        Reset the value of each node in the tree to 0.

        Returns:
        None
        """
        for node in self.nodes.values():
            node.value = 0

    def propagate(self, scores):
        """
        Assign scores to the leaf nodes and propagate them up the tree.

        Parameters:
        scores (dict): A dictionary mapping node IDs to scores.

        Returns:
        None
        """
        self.reset()

        # Assign scores to the leaf nodes
        for node_id, score in scores.items():
            node_id = int(node_id)
            self.nodes[node_id].value = score

            # Propagate the scores up the tree
            parent_id = self.nodes[node_id].parent
            while parent_id is not None:
                self.nodes[parent_id].value += score
                parent_id = self.nodes[parent_id].parent


        
def compute_joint_entropy(hierarchy, scores, threshold):
    joint_entropy = {}
    tree = Tree(hierarchy)
    for i, instance_scores in enumerate(scores):
        start_time = time.time()
        tree.propagate(instance_scores)
        for node_a, node_b in combinations(tree.nodes, 2):
            node_a = int(node_a)
            node_b = int(node_b)
            if tree.nodes[node_a].value <= threshold or tree.nodes[node_b].value <= threshold:
                continue
            key = f'{node_a},{node_b}'
            joint_entropy.setdefault(key, 0)
            entropy_value = entropy([tree.nodes[node_a].value, tree.nodes[node_b].value])
            joint_entropy[key] += round(entropy_value, 5)
        
        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        print(f'..processed {i} instances in {elapsed_time} seconds. Estimated time remaining: {elapsed_time * (len(scores) - i)} seconds.')
    return joint_entropy


def compute_joint_occurrence(hierarchy, scores, threshold):
    joint_occurrence = {}
    tree = Tree(hierarchy)
    for i, instance_scores in enumerate(scores):
        start_time = time.time()
        tree.propagate(instance_scores)
        for node_a, node_b in combinations(tree.nodes, 2):
            node_a = int(node_a)
            node_b = int(node_b)
            if tree.nodes[node_a].value <= threshold or tree.nodes[node_b].value <= threshold:
                continue
            key = f'{node_a},{node_b}'
            joint_occurrence.setdefault(key, 0)
            joint_occurrence[key] += 1
        
        end_time = time.time()
        elapsed_time = int(end_time - start_time)
        print(f'..processed {i} instances in {elapsed_time} seconds. Estimated time remaining: {elapsed_time * (len(scores) - i)} seconds.')
    return joint_occurrence


def parallel_compute(function, hierarchy, scores, threshold, num_cores=None):
    result = {}
    if num_cores is None:
        num_cores = cpu_count()
    print(f"Computing {function.__name__} on {num_cores} cores doing {int(len(scores)/num_cores)} instances each...")
    
    args = [(hierarchy, chunk_scores, threshold) for chunk_scores in np.array_split(scores, num_cores)]
    with Pool(num_cores) as pool:
        results = pool.starmap(function, args)
    
    for partial_joint_entropy in results:
        for key, value in partial_joint_entropy.items():
            result.setdefault(key, 0)
            result[key] += value

    return result

def load_json(filename):
    """
    Load a JSON file.

    Parameters:
    filename (str): The path to the JSON file.

    Returns:
    dict: The data from the JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def write_json(filename, data):
    """
    Write data to a JSON file.

    Parameters:
    filename (str): The path to the JSON file.
    data (json serializable data): The data to write to the file.

    Returns:
    None
    """
    with open(filename, 'w') as f:
        return json.dump(data, f, indent=4)

def process_dataset(dataset_directory, threshold, num_cores):
    """
    Process a dataset by computing the joint entropy of its hierarchy of nodes.

    Parameters:
    dataset_directory (str): The directory containing the dataset.

    Returns:
    None
    """
    hierarchy = load_json(os.path.join(dataset_directory, 'hierarchy.json'))
    scores = load_json(os.path.join(dataset_directory, 'scores.json'))
    labels = load_json(os.path.join(dataset_directory, 'labels.json'))

    # Format labels like scores to use for label <--> hierarchy alignment
    name_to_id = {node['name']: node['id'] for node in hierarchy}
    label_scores = [{name_to_id[node_name]: 1 for node_name in instance_labels} for instance_labels in labels]

    joint_entropy_path = os.path.join(dataset_directory, 'joint_entropy.json')
    if not os.path.isfile(joint_entropy_path):
        joint_entropy = parallel_compute(compute_joint_entropy, hierarchy, scores, threshold, num_cores=num_cores)
        write_json(joint_entropy_path, joint_entropy)
    
    joint_occurrence_path = os.path.join(dataset_directory, 'joint_occurrence.json')
    if not os.path.isfile(joint_occurrence_path):
        joint_occurrence = parallel_compute(compute_joint_occurrence, hierarchy, scores, threshold, num_cores=num_cores)
        write_json(joint_occurrence_path, joint_occurrence)

    joint_entropy_labels_path = os.path.join(dataset_directory, 'joint_entropy_labels.json')
    if not os.path.isfile(joint_entropy_labels_path):
        joint_entropy_labels = parallel_compute(compute_joint_entropy, hierarchy, label_scores, 0, num_cores=num_cores)
        write_json(joint_entropy_labels_path, joint_entropy_labels)

    joint_occurrence_labels_path = os.path.join(dataset_directory, 'joint_occurrence_labels.json')
    if not os.path.isfile(joint_occurrence_labels_path):
        joint_occurrence_labels = parallel_compute(compute_joint_occurrence, hierarchy, label_scores, 0, num_cores=num_cores)
        write_json(joint_occurrence_labels_path, joint_occurrence_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a dataset by computing the joint entropy of its hierarchy of nodes.')
    parser.add_argument('--data_directory', type=str, default='data/', help='The directory containing the data.')
    parser.add_argument('--num_cores', type=int, default=None, help='The number of cores to use for parallel computation.')
    parser.add_argument('--threshold', type=float, default=None, help='The threshold to consider probabilities at.')
    args = parser.parse_args()

    for folder in os.listdir(args.data_directory):
        dataset_directory = os.path.join(args.data_directory, folder)
        if not os.path.isdir(dataset_directory):
            continue

        if folder not in ['cifar', 'mimic']:
            continue
        
        if args.threshold is None:
            if folder == 'cifar': args.threshold = 0.01
            if folder == 'mimic': args.threshold = 0.5

        print(f'Processing {folder}...')
        process_dataset(dataset_directory, args.threshold, args.num_cores)
        print(f'{folder} complete.')