"""Create hierarchy for MIMIC-III dataset."""

import queue
import numpy as np
from treelib import Tree


def parse_code(code):
    """Parses an ICD code into its parts."""
    prefix = ''
    start = code
    start_prefix = code
    start_suffix = ''
    end = code
    end_prefix = code
    end_suffix = ''
    
    if '-' in code:
        start, end = code.split('-')
        start_prefix, end_prefix = code.split('-')
    
    if code[0] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        prefix = code[0]
        start = start[1:]
        start_prefix = start_prefix[1:]
        end = end[1:]
        end_prefix = end_prefix[1:]
        
    if '.' in start:
        start_prefix, start_suffix = start.split('.')
    if '.' in end:
        end_prefix, end_suffix = end.split('.')
        
    return prefix, (start, start_prefix, start_suffix), (end, end_prefix, end_suffix)


def make_tree(icd9_file):
    tree = Tree()
    
    # Get the ICD-9 codes from the MIMIC-III dataset
    with open(icd9_file, 'r') as f:
        lines = f.readlines()
        codes = [line.strip().split('\t') for line in lines]
        code_map = {code: code_name for code, code_name in codes}
    
    # Map the codes to their parents
    child_to_parent = {code: None for code, _ in codes}
    for code, code_name in codes:
        if code == '@': continue
        
        prefix, start, end = parse_code(code)
        
        # If code is xx.xx check for direct parent
        if start == end and start[-1] is not '':
            for i in range(1, len(start[1])+1):
                parent_code = prefix + start[1] + '.' + start[-1][:-i]
                if parent_code.endswith('.'):
                    parent_code = parent_code[:-1]
                if parent_code in child_to_parent:
                    child_to_parent[code] = parent_code
                    break
                    
        # If code is xx or xx-yy check for smallest range it fits in           
        if child_to_parent[code] is None:
            smallest_range = None
            parent_name = None
            for parent_code, parent_node in code_map.items():
                parent_prefix, parent_start, parent_end = parse_code(parent_code)
                if parent_code == '@': continue # assign root parent later
                if code == parent_code: continue # can not be a parent of yourself
                if prefix != parent_prefix: continue # not in the same prefix family
                if parent_start == parent_end: continue # not a range node
                if len(parent_start[1]) != len(start[1]): continue # not the same length = not in the same code family
                if float(start[0]) >= float(parent_start[0]) and float(end[0]) <= float(parent_end[0]):
                    parent_range = float(parent_end[0]) - float(parent_start[0])
                    if smallest_range is None or parent_range < smallest_range:
                        smallest_range = parent_range
                        parent_name = parent_code
            if parent_name is not None:
                child_to_parent[code] = parent_name
            else:
                child_to_parent[code] = '@'
    
    # Create the tree
    parent_to_children = {}
    for child, parent in child_to_parent.items():
        if parent is not None:
            parent_to_children.setdefault(parent, []).append(child)
        
    q = queue.Queue()
    q.put('@')
    i = 0
    while not q.empty():
        code = q.get()
        tree.create_node(
            tag=code_map[code], 
            identifier=code, 
            parent=child_to_parent[code],
            data=None
        )
        if code in parent_to_children:
            for child in sorted(parent_to_children[code]):
                q.put(child)
    return tree

def show(tree, hide_zeros=True):
    string = tree.show(stdout=False, key=lambda n: n.identifier)
    for node_id, node in tree.nodes.items():
        node_value = node.data
        if node_value is not None:
            node_value = round(node_value, 2)
            if node_value == 0 and hide_zeros:
                node_value = ''
            else:
                node_value = f'{node_value:.2f}'
        string = string.replace(f'{node.tag}\n', f'{node_id}: {node.tag} ({node_value})\n')
    return string


def propagate(labels, tree):
    # Set all nodes to 0
    for node in tree.all_nodes():
        node.data = 0
    
    # Overwrite the label nodes as 1
    for label in labels:
        if tree.contains(label): # some labels are not in ICD hierarchy file
            node = tree.get_node(label)
            node.data = 1
        
    # Propagate values up the tree
    level = tree.depth()
    while level >= 0:
        nodes = tree.filter_nodes(lambda n: tree.depth(n) == level)
        for node in nodes:
            if not node.is_leaf() and node.data == 0:
                node.data = np.sum([child.data for child in tree.children(node.identifier)])
        level -= 1

    return tree

def serialize_tree(tree):
    output = []
    node_ids = {node.identifier: i for i, node in enumerate(tree.all_nodes())}
    for i, node in enumerate(tree.all_nodes()):
        json_object = {'id': node_ids[node.identifier], 'name': node.identifier}
        if tree.parent(node.identifier) is not None:
            json_object['parent'] = node_ids[tree.parent(node.identifier).identifier]
        output.append(json_object)
    return output
