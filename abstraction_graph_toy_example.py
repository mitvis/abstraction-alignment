"""Create an abstraction graph for the toy example."""

import numpy as np
from treelib import Tree

def make_abstraction_graph(misaligned=False):
    abstraction_graph = Tree()

    abstraction_graph.create_node(tag='root', identifier='root', parent=None, data=None)

    # Add superclass nodes
    abstraction_graph.create_node(tag='parent0', identifier='parent0', parent='root', data=None)
    abstraction_graph.create_node(tag='parent1', identifier='parent1', parent='root', data=None)

    # Add class nodes
    abstraction_graph.create_node(tag='child0', 
                     identifier='child0', 
                     parent='parent0', 
                     data=None)
    if misaligned:
        abstraction_graph.create_node(tag='child1', 
                         identifier='child1', 
                         parent='parent0', 
                         data=None)
    else: 
        abstraction_graph.create_node(tag='child1', 
                         identifier='child1', 
                         parent='parent1', 
                         data=None)
    abstraction_graph.create_node(tag='child2', 
                     identifier='child2', 
                     parent='parent1', 
                     data=None)
    return abstraction_graph

def show_abstraction_graph(abstraction_graph, hide_zeros=False):
    string = abstraction_graph.show(stdout=False)
    for node_id, node in abstraction_graph.nodes.items():
        node_value = node.data
        if node_value is not None:
            node_value = round(node_value, 2)
            if node_value == 0 and hide_zeros:
                node_value = ''
            else:
                node_value = f'{node_value:.2f}'
        string = string.replace(f'{node_id}\n', f'{node_id} ({node_value})\n')
    return string

def propagate(outputs, abstraction_graph):
    """Propagate model outputs through the abstraction_graph."""
    # Assign values to the leaves of the abstraction_graph
    for i, value in enumerate(outputs):
        name = f'child{i}'
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