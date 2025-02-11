# Graph class for representing human abstraction graphs.

import numpy as np
from queue import Queue


class Graph():
    """A directed acyclic graph class."""
    
    def __init__(self, nodes, root_id):
        """
        Initialize a graph with its nodes, including a root.
        
        Args:
            nodes (dict): Dictionary mapping node IDs to Node objects
            root_id: Identifier for the root node, must exist in nodes
        """
        self.root_id = root_id
        self.nodes = {root_id: nodes.pop(root_id), **nodes}
        
    def __str__(self):
        """String representations showing the root and number of nodes."""
        return f"Graph: root={self.root_id} num_nodes={len(self.nodes)}"
        
    def __len__(self):
        """Return the number of nodes in the graph."""
        return len(self.nodes)
    
    def __getitem__(self, node_name):
        """Given an node ID, returns the requested Node object."""
        return self.nodes[node_name]
    
    def compute_reachable_leaves(self):
        """Computes the leaves reachable from each node in the graph."""
        leaves = [node for node in self.nodes.values() if len(node.children) == 0]
        for leaf in leaves:
            leaf.reachable_leaves = set([leaf])
            
        self.nodes[self.root_id].set_reachable_leaves()
        
    def compute_height(self):
        """Computes the height of all nodes from the root."""
        self.nodes[self.root_id].set_height()
        
    def compute_depth(self):
        """Computes the depths of all nodes starting from the leaves."""
        leaves = [node for node in self.nodes.values() if len(node.children) == 0]
        for leaf in leaves:
            leaf.set_depth()
        
    def finalize(self):
        """Complete graph initialization."""
        self.compute_reachable_leaves()
        self.compute_height()
        self.compute_depth()
        for node in self.nodes.values():
            assert node.reachable_leaves is not None
            assert node.height is not None
            assert node.depth is not None
            
    def serialize(self, include_values=False):
        """Converts the graph into a JSON-serializable format. If 
        includes_values, the node's values will be added."""
        output = []
        node_ids = {node.name: i for i, node in enumerate(self.nodes.values())}
        for i, (node_name, node) in enumerate(self.nodes.items()):
            json_object = {'id': node_ids[node.name], 'name': node.name}
            if include_values:
                json_object['values'] = node.values
            json_object['parents'] = [node_ids[parent.name] for parent in node.parents]
            output.append(json_object)
        return output
            
    def check_tree(self):
        """Checks if the graph is a tree."""
        for node_name, node in self.nodes.items():
            assert node.is_connected() # check all are connected
            if node_name != self.root_id:
                assert len(node.parents) > 0 # check only the root doesn't have a parent

        reachable_node_names = set()
        queue = Queue()
        queue.put(self.nodes[self.root_id])
        while not queue.empty():
            node = queue.get()
            reachable_node_names.add(node.name)
            for child in node.children:
                queue.put(child)
        assert len(reachable_node_names) == len(self.nodes) # check all nodes are reachable from root
        assert np.all([name in reachable_node_names for name in self.nodes])
        print(f"Graph is a tree.")
        
        
class Node():
    """Nodes in a DAG structure."""
    
    def __init__(self, name, value=None, parents=[], children=[]):
        """Initialize nodes with a name, value, parent, and children."""
        self.name = name
        self.value = value
        self.parents = set(parents)
        self.children = set(children)
        self.reachable_leaves = None
        self.height = None
        self.depth = None
        
    def connect_child(self, child):
        """Add child to node."""
        if self not in child.parents:
            child.parents.add(self)
        self.children.add(child)
        
    def __str__(self):
        """String representation of the node's name, value, parent, and children."""
        return f"{self.name} values={self.values} parents={[parent.name for parent in self.parents]} num children={len(self.children)} depth={self.depth} height={self.height}"
    
    def __repr__(self):
        """Returnts the node's name."""
        return self.name
    
    def set_reachable_leaves(self):
        """Sets and returns the set of leaf nodes reachable from the node."""
        if len(self.children) == 0:
            self.reachable_leaves = set([self])

        if self.reachable_leaves is None:
            child_reachable_leaves = [child.set_reachable_leaves() for child in self.children]
            self.reachable_leaves = set().union(*child_reachable_leaves)
            
        return self.reachable_leaves
        
    def set_depth(self):
        """Sets and returns the node's depth."""
        if len(self.parents) == 0:
            self.depth = 0
        if self.depth is None:
            parent_depths = [parent.set_depth() for parent in self.parents]
            self.depth = max(parent_depths) + 1
        return self.depth
    
    def set_height(self):
        """Sets and returns the node's height."""
        if len(self.children) == 0:
            self.height = 0
        if self.height is None:
            child_heights = [child.set_height() for child in self.children]
            self.height = max(child_heights) + 1
        return self.height
        
    def is_connected(self):
        """Returns if the node is connected to other nodes."""
        return len(self.children) > 0 or len(self.parents) > 0