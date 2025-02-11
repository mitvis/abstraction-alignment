# Get MIMIC Medical Dataset data for the Abstraction Alignment interface.

import os
import csv
import json
import argparse
from tqdm import tqdm
from treelib import Tree
from itertools import combinations

from abstraction_graph_mimic import make_abstraction_graph, serialize_abstraction_graph, propagate


def main(icd9_file, data_file):
    """Create data files for the medical dataset analysis case study."""
    # Load the MIMIC-III data
    notes = {}
    labels = {}
    with open(data_file, 'r') as f:
        notes_reader = csv.reader(f)
        for i, note in enumerate(notes_reader):
            if i == 0: continue # skip header
            hadm_id = note[1]
            notes[hadm_id] = note[2]
            labels[hadm_id] = note[3].split(';')
    print(f'{len(notes)} data instances')
    
    with open(icd9_file, 'r') as f:
        codes = [line.strip().split('\t') for line in f.readlines()]
        code_map = {code: code_name for code, code_name in codes}
    
    # Create the ICD-9 abstraction graph
    abstraction_graph = make_abstraction_graph(icd9_file)
    print('Created MIMIC abstraction_graph of ICD-9 codes')
    print(f'Abstraction_graph depth: {abstraction_graph.depth()}; Num nodes: {abstraction_graph.size()}; Num leaves: {len(abstraction_graph.leaves())}')
    
    # Prune the abstraction_graph to remove nodes that aren't seen in the data
    relevant_codes = []
    for label in labels.values():
        relevant_codes.extend(label)
    relevant_codes = set(relevant_codes)

    relevant_nodes = set([])
    for code in relevant_codes:
        node = abstraction_graph.get_node(code)
        if node is None: continue
        relevant_nodes.add(code)
        for ancestor in abstraction_graph.rsearch(code):
            relevant_nodes.add(ancestor)
    print(f"{len(relevant_nodes)} relevant nodes.")

    pruned_abstraction_graph = Tree()
    for level in tqdm(range(abstraction_graph.depth() + 1)):
        level_nodes = abstraction_graph.filter_nodes(lambda n: abstraction_graph.depth(n) == level)
        for node in level_nodes:
            if node.identifier in relevant_nodes:
                parent = abstraction_graph.parent(node.identifier)
                if parent is not None:
                    parent = parent.identifier
                pruned_abstraction_graph.create_node(
                    tag=node.tag, 
                    identifier=node.identifier, 
                    parent=parent,
                    data=None
                )
    print(f'Pruned abstraction graph depth: {pruned_abstraction_graph.depth()}; Num nodes: {pruned_abstraction_graph.size()}; Num leaves: {len(pruned_abstraction_graph.leaves())}')
    
    # Write out the abstraction graph files.
    output_dir = 'interface/data/mimic/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Write out the icd9 abstraction graph.
    serialized_abstraction_graph = serialize_abstraction_graph(pruned_abstraction_graph)
    with open(os.path.join(output_dir, 'abstraction_graph.json'), 'w') as f:
        json.dump(serialized_abstraction_graph, f, indent=4)
    name_to_id = {node['name']: node['id'] for node in serialized_abstraction_graph}
        
    # Write out the labels for each instance.
    hadm_ids = list(labels.keys())
    labels_list = []
    for hadm_id in hadm_ids:
        instance_labels = []
        for label in labels[hadm_id]:
            full_label = f'{label}'
            if label in code_map:
                full_label += f': {code_map[label]}'
            instance_labels.append(full_label)
        labels_list.append(instance_labels)
    # labels_list = [[f'{label}: {code_map[label]}' for label in labels[hadm_id]] for hadm_id in hadm_ids]
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(labels_list, f, indent=4)
    print(f"Dumped {len(labels_list)} labels.")
    
    # Write out the concept coconfusion
    node_pairs = {}
    for hadm_id, label in tqdm(labels.items()):
        fitted_abstractions = propagate(label, pruned_abstraction_graph)
        weighted_nodes = fitted_abstractions.filter_nodes(lambda n: n.data > 0)
        for pair in combinations(weighted_nodes, 2):
            pair_ids = [name_to_id[f'{p.identifier}: {p.tag}'] for p in pair]
            pair_ids = tuple(sorted(list(pair_ids)))
            node_pairs.setdefault(pair_ids, 0)
            node_pairs[pair_ids] += 1
    normalized_node_pairs = {f'{pair[0]},{pair[1]}': value/len(labels) for pair, value in node_pairs.items()}
    with open(os.path.join(output_dir, f"coconfusion.json"), 'w') as f:
        json.dump(normalized_node_pairs, f, indent=4) 
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory path for MIMIC-III dataset')
    args = parser.parse_args()
    args.data_dir = os.path.normpath(args.data_dir) # e.g., '/nobackup/users/aboggust/data/mimic/mimicdata/'
    icd9_file = os.path.join(args.data_dir, 'ICD9_descriptions')
    test_data_file = os.path.join(args.data_dir, 'mimic3', 'test_full.csv')
    main(icd9_file, test_data_file)
    