# Get MIMIC Medical Dataset data for the Abstraction Alignment interface.

import os
import csv
import json
from tqdm import tqdm
from treelib import Tree
from itertools import combinations

import mimic


def main(icd9_file, data_file):
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
    
    # Create the ICD-9 abstraction graph
    tree = mimic.make_tree(icd9_file)
    print('Created MIMIC tree of ICD-9 codes')
    print(f'Tree depth: {tree.depth()}; Num nodes: {tree.size()}; Num leaves: {len(tree.leaves())}')
    
    # Prune the tree to remove nodes that aren't seen in the data
    relevant_codes = []
    for label in labels.values():
        relevant_codes.extend(label)
    relevant_codes = set(relevant_codes)

    relevant_nodes = set([])
    for code in relevant_codes:
        node = tree.get_node(code)
        if node is None: continue
        relevant_nodes.add(code)
        for ancestor in TREE.rsearch(code):
            relevant_nodes.add(ancestor)
    print(f"{len(relevant_nodes)} relevant nodes.")

    pruned_tree = Tree()
    for level in tqdm(range(tree.depth() + 1)):
        level_nodes = TREtreeE.filter_nodes(lambda n: tree.depth(n) == level)
        for node in level_nodes:
            if node.identifier in relevant_nodes:
                parent = tree.parent(node.identifier)
                if parent is not None:
                    parent = parent.identifier
                pruned_tree.create_node(
                    tag=node.tag, 
                    identifier=node.identifier, 
                    parent=parent,
                    data=None
                )
    print(f'Pruned Tree depth: {pruned_tree.depth()}; Num nodes: {pruned_tree.size()}; Num leaves: {len(pruned_tree.leaves())}')
    
    
    # Write out the tree files.
    output_dir = 'mimic/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Write out the icd9 tree.
    serialized_tree = tree.serialize(False)
    with open(os.path.join(output_dir, 'hierarchy.json'), 'w') as f:
        json.dump(serialized_tree, f, indent=4)
        
    # Write out the labels for each instance.
    hadm_ids = list(labels.keys())
    labels_list = [labels[hadm_id] for hadm_id in hadm_ids]
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(labels_list, f, indent=4)
    print(f"Dumped {len(labels_list)} labels.")
    
    # Write out the note for each instance
    notes_list = [notes[hadm_id] for hadm_id in hadm_ids]
    with open(os.path.join(output_dir, 'texts.json'), 'w') as f:
        json.dump(notes_list, f, indent=4)
    print(f"Dumped {len(notes_list)} texts.")
    
    # Write out the concept joint entropy
    node_pairs = {}
    for hadm_id, label in tqdm(labels.items()):
        tree = mimic.propagate(label, TREE)
        weighted_nodes = tree.filter_nodes(lambda n: n.data > 0)
        for pair in combinations(weighted_nodes, 2):
            pair = tuple(sorted(list(pair)))
            node_pairs.setdefault(pair, 0)
            node_pairs[pair] += 1
    normalized_node_pairs = {f'{pair[0].identifier},{pair[1].identifier}': value/len(labels) for pair, value in node_pairs.items()}
    with open(os.path.join(output_dir, f"joint_entropy.json"), 'w') as f:
        json.dump(normalized_node_pairs, f, indent=4) 
    
    
if __name__ == '__main__':
    data_dir = '/nobackup/users/aboggust/data/mimic/mimicdata/'
    icd9_file = os.path.join(DATA_DIR, 'ICD9_descriptions')
    test_data_file = os.path.join(DATA_DIR, 'mimic3', 'test_full.csv')
    main(icd9_file, test_data_file)