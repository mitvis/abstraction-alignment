# Utility functions for dealing with trees

from collections import defaultdict
import numpy as np
import torch 
from node import Node
from queue import Queue
import json

class Tree():
    
    def __init__(self, nodes, root_id):
        self.root_id = root_id
        self.nodes = {root_id: nodes.pop(root_id), **nodes}
        
    def __str__(self):
        return f"Tree: root={self.root_id} num_nodes={len(self.nodes)}"
        
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, node_name):
        return self.nodes[node_name]
    
    def compute_reachable_leaves(self):
        leaves = [node for node in self.nodes.values() if len(node.children) == 0]
        for leaf in leaves:
            leaf.reachable_leaves = set([leaf])
            
        self.nodes[self.root_id].set_reachable_leaves()
        
    def compute_height(self):
        self.nodes[self.root_id].set_height()
        
    def compute_depth(self):
        leaves = [node for node in self.nodes.values() if len(node.children) == 0]
        for leaf in leaves:
            leaf.set_depth()
        
    def finalize(self):
        self.compute_reachable_leaves()
        self.compute_height()
        self.compute_depth()
        for node in self.nodes.values():
            assert node.reachable_leaves is not None
            assert node.height is not None
            assert node.depth is not None
            
    def serialize(self, include_values=False):
        output = []
        node_ids = {node.name: i for i, node in enumerate(self.nodes.values())}
        for i, (node_name, node) in enumerate(self.nodes.items()):
            json_object = {'id': node_ids[node.name], 'name': node.name}
            if include_values:
                json_object['value'] = node.value
            if node.parent is not None:
                json_object['parent'] = node_ids[node.parent.name]
            output.append(json_object)
        return output
    
    def specify_tree(self, model_input, model, device, labels):
        """Specifies the confidence tree for a specific input.
        
        Args:
            model_input (Tensor): the model_input used to set the confidence.
            model (Pytorch Model): the model that takes in the model_input.
            device (Pytoch Device): the device the model is on.
            labels (list): a list of class names corresponding the model's 
                outputs.
            allow_multiparents (bool, default=False): if True, confidence of a
                non-leaf node is the sum of the confidence of its reachable
                leaves. If False, multiple parents are not allowed, so 
                confidence can only travel one path from a leaf to any other 
                node.
                
        Returns: nothing. Updates the nodes in the tree with their confidence
            values.
        """
        model.eval()
        model_inputs = model_input.unsqueeze(0)
        with torch.no_grad():
            model_inputs = model_inputs.to(device)

        # Compute model predictions
        output = model(model_inputs)
        confidences = torch.nn.functional.softmax(output, dim=1).squeeze(0).detach().cpu().numpy()

        # Propogate confidences up the tree
        self.set_and_propogate_confidences(confidences, labels)
        
    def set_and_propogate_confidences(self, confidences, labels):
        """Sets the leaf confidences and propogates them through the tree."""
        self.clear_confidence()
        for i, confidence in enumerate(confidences):
            node_name = labels[i]
            node = self.nodes[node_name]
            node.value = confidence
        self.propogate_confidence()
        
    def clear_confidence(self):
        """Clears the confidence for every node in the tree."""
        for node in self.nodes.values():
            node.value = None

    def propogate_confidence(self, confidence_threshold=1e-4):
        """Propograte confidence values from leaf to root in a tree. Confidence
        at the leaf node is the model's predicted confidence. Confidence at an 
        internal node is the sum of the confidence from leaf nodes the internal
        node can reach.

        Args:
            allow_multiparents (bool, default=False): if True, confidence of a
                non-leaf node is the sum of the confidence of its reachable
                leaves. If False, multiple parents are not allowed, so 
                confidence can only travel one path from a leaf to any other 
                node. At each confident node, it allows confidence to travel to 
                the first parent only.
            confidence_threshold (float, default=1e-4): A node is considered
                confident if its confidence is above the confidence_threshold.
                Only matters when allow_multiparents=False and confident nodes
                can only have one parent.
                
        Returns: nothing. Nodes in the tree are modified with updated confidence
            values.
        """
        for node in self.nodes.values():
            value = np.sum([reachable_leaf.value for reachable_leaf in node.reachable_leaves])
            node.value = value

    def to_json(self, filename=None):
        result = []
        name_to_id = {node.name: i for i, node in enumerate(self.nodes.values())}
        for i, (code, node) in enumerate(self.nodes.items()):
            node_result = {}
            node_result['name'] = node.name
            node_result['id'] = name_to_id[node.name]
            if node.parent is not None:
                node_result['parent'] = node.parent.name
            result.append(node_result)
        if filename is not None: 
            with open(filename, 'w') as f:
                json.dump(result, f, indent=4)
        return result
            
            
def check_tree(tree):
    """Checks to ensure the tree is an accurate tree."""
    for node_name, node in tree.nodes.items():
        assert node.is_connected() # check all are connected
        if node_name != tree.root_id:
            assert node.parent is not None # check only the root doesn't have a parent

    reachable_node_names = set()
    queue = Queue()
    queue.put(tree[tree.root_id])
    while not queue.empty():
        node = queue.get()
        reachable_node_names.add(node.name)
        for child in node.children:
            queue.put(child)
    assert len(reachable_node_names) == len(tree.nodes) # check all nodes are reachable from root
    assert np.all([name in reachable_node_names for name in tree.nodes])
    print(f"Tree passes tree checks.")

