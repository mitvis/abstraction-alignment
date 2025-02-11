"""Create an abstraction graph for the CIFAR-100 dataset."""

import re
import numpy as np
from treelib import Tree

from util.cifar.cifar_metadata import CLASS_LABELS, SUPERCLASS_LABELS, CLASS_TO_SUPERCLASS


def make_abstraction_graph():
    """Create a Tree containing the CIFAR100 abstraction graph."""
    abstraction_graph = Tree()
        
    # Add root
    abstraction_graph.create_node(tag='root', identifier='root', parent=None, data=None)
    
    # Add superclass nodes
    for superclass in SUPERCLASS_LABELS:
        abstraction_graph.create_node(tag=superclass, identifier=superclass, parent='root', data=None)
            
    # Add class nodes
    for class_label in CLASS_LABELS:
        abstraction_graph.create_node(tag=class_label, 
                         identifier=class_label, 
                         parent=CLASS_TO_SUPERCLASS[class_label], 
                         data=None)
        
    return abstraction_graph

def propagate(outputs, abstraction_graph):
    """Propagate model outputs through the abstraction_graph."""
    # Assign values to the leaves of the abstraction_graph
    for i, value in enumerate(outputs):
        name = CLASS_LABELS[i]
        node = abstraction_graph.get_node(name)
        node.data = value
        
    # Propagate values up the abstraction_graph
    level = abstraction_graph.depth() - 1 # leaf level = depth
    while level >= 0:
        nodes = abstraction_graph.filter_nodes(lambda x: abstraction_graph.depth(x) == level)
        for node in nodes:
            reachable_leaves = abstraction_graph.leaves(node.identifier)
            node.data = np.sum([leaf.data for leaf in reachable_leaves])
        level -= 1
    
    return abstraction_graph

def show_abstraction_graph(abstraction_graph, hide_zeros=True):
    string = abstraction_graph.show(stdout=False)
    for node_id, node in abstraction_graph.nodes.items():
        node_value = node.data
        if node_value is not None:
            node_value = round(node_value, 2)
            if node_value == 0 and hide_zeros:
                string = re.sub(rf"\n.*{node_id}\n", '\n', string)
            else:
                node_value = f'{node_value:.2f}'
                string = string.replace(f'{node_id}\n', f'{node_id} ({node_value})\n')
    return string

def serialize_abstraction_graph(abstraction_graph):
    output = []
    node_ids = {node.identifier: i for i, node in enumerate(abstraction_graph.all_nodes())}
    for i, node in enumerate(abstraction_graph.all_nodes()):
        json_object = {'id': node_ids[node.identifier], 'name': node.identifier}
        if abstraction_graph.parent(node.identifier) is not None:
            json_object['parent'] = node_ids[abstraction_graph.parent(node.identifier).identifier]
        output.append(json_object)
    return output

