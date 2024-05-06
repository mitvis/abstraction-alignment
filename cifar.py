"""Create hierarchy for CIFAR-100 dataset."""

import numpy as np
from treelib import Tree

from cifar_metadata import CLASS_LABELS, SUPERCLASS_LABELS, CLASS_TO_SUPERCLASS


def make_tree():
    """Create a Tree containing the CIFAR100 hierarchy."""
    tree = Tree()
        
    # Add root
    tree.create_node(tag='root', identifier='root', parent=None, data=None)
    
    # Add superclass nodes
    for superclass in SUPERCLASS_LABELS:
        tree.create_node(tag=superclass, identifier=superclass, parent='root', data=None)
            
    # Add class nodes
    for class_label in CLASS_LABELS:
        tree.create_node(tag=class_label, 
                         identifier=class_label, 
                         parent=CLASS_TO_SUPERCLASS[class_label], 
                         data=None)
        
    return tree

def propagate(outputs, tree):
    """Propagate model outputs through the tree."""
    # Assign values to the leaves of the tree
    for i, value in enumerate(outputs):
        name = CLASS_LABELS[i]
        node = tree.get_node(name)
        node.data = value
        
    # Propagate values up the tree
    level = tree.depth() - 1 # leaf level = depth
    while level >= 0:
        nodes = tree.filter_nodes(lambda x: tree.depth(x) == level)
        for node in nodes:
            reachable_leaves = tree.leaves(node.identifier)
            node.data = np.sum([leaf.data for leaf in reachable_leaves])
        level -= 1
    
    return tree

def show(tree):
    string = tree.show(stdout=False)
    for node_id, node in tree.nodes.items():
        node_value = node.data
        if node_value is not None and node_value >= 0.01:
            node_value = f'{node_value:.2f}'
        string = string.replace(f'{node_id}\n', f'{node_id} ({node_value})\n')
    return string
