# Get Language Model Specificity data for the Abstraction Alignment interface.

import json
import os
import random
import pickle
import numpy as np
from tqdm import tqdm
from queue import Queue
from collections import Counter
from itertools import combinations
from scipy import stats

import metrics
from nltk.corpus import wordnet as wn
from graph import Graph, Node


def load_task_data(task_id, data_dir):
    """Load the data instances for a task_id. There can be duplicates, but we
    handle them the same way the S-TEST repo does."""
    data = {}
    with open(os.path.join(data_dir, f'{task_id}.jsonl'), 'r') as f:
        for line in f:
            datum = json.loads(line)
            data[datum['sub_label']] = datum
    return data


def get_synset_name(synset):
    """Parse string name from WordNet synset."""
    return synset.name().split('.')[0]


def get_synsets_relatives(synset, traversal_fn_name, root=None, include_self=True):
    """Get all synsets connected to synset via travesal_fn_name."""
    traversal_fn = getattr(synset, traversal_fn_name)
    words = set([])
    if include_self:
        words.add(get_synset_name(synset))
    if (root is not None and synset == root) or len(traversal_fn()) == 0:
        return words
    for word in traversal_fn():
        next_words = get_synsets_relatives(word, traversal_fn_name, root)
        words.update(next_words)
    return words


def get_task_synsets(task, synset_labels):
    """Get every synset connected to a label in the task data."""
    task_synsets = set([])
    for synset in synset_labels:
        if synset in task_synsets: 
            continue
        children = get_synsets_relatives(synset, task['down_fn'], root=task['root'], include_self=False)
        parents = get_synsets_relatives(synset, task['up_fn'], root=task['root'], include_self=False)
        task_synsets.add(get_synset_name(synset))
        task_synsets.update(children)
        task_synsets.update(parents)
    return task_synsets


def create_wordnet_dag(task, synsets):
    """Create a DAG representing the task's synsets and thier relationships."""
    task_synsets = get_task_synsets(task, synsets)
    print(f"{len(task_synsets)} concepts synsets related to {len(set(synsets))} synsets from the {task['name']} prediction task.")
    print(f'Example concepts: {random.sample(task_synsets, 5)}')
    
    root = task['root']
    nodes = {}
    queue = Queue()
    queue.put((root, None)) # Queue contains the synset and its parent DAG node.
    while not queue.empty():
        synset, parent_node = queue.get()
        synset_name = get_synset_name(synset)
        
        # Create a node for the synset if it does not already exit.
        if synset_name not in nodes:
            synset_node = Node(synset_name)
        else: 
            synset_node = nodes[synset_name]
            
        # Connect the node to its parent and update the graph.
        if parent_node is not None: # Only the root node has parent = None.
            parent_node.connect_child(synset_node)
        nodes[synset_name] = synset_node
        
        # Continue the traversal down the abstraction graph.
        traversal_fn = getattr(synset, task['down_fn'])
        for next_synset in traversal_fn():
            if get_synset_name(next_synset) not in task_synsets:
                # print(get_synset_name(next_synset))
                continue # Skip relatives that are not related to the task.
            queue.put((next_synset, synset_node))
    wordnet_dag = Graph(nodes, get_synset_name(root))
    wordnet_dag.finalize()
    return wordnet_dag


def convert_dag_to_tree(dag, task, node_name_to_parent_name):
    """Converts a Graph into a Tree."""
    # Create tree with blank versions of all nodes
    tree_nodes = {}
    for graph_node_name, graph_node in dag.nodes.items():
        tree_nodes[graph_node_name] = Node(graph_node_name, value=0, parents=[])
    print(f'Added {len(tree_nodes)}/{len(dag.nodes)} node to the tree.')

    # Connect the tree nodes by choosing the parent with the lowest depth
    for graph_node_name, graph_node in dag.nodes.items():
        node = tree_nodes[graph_node_name]
        assert len(node.parents) == 0 # Confirm the tree node has not already been assigned a parent.
        if len(graph_node.parents) == 0:
            continue # Skip the root node.
        if graph_node.name in node_name_to_parent_name:
            parent_node_name = node_name_to_parent_name[graph_node.name]
        elif len(graph_node.parents) == 1:
            parent_node_name = next(iter(graph_node.parents)).name
        else:
            raise ValueError(f"Node {graph_node.name} has multiple parents {graph_node.parents} and is not in node_name_to_parent_name.")
        parent_node = tree_nodes[parent_node_name]
        parent_node.connect_child(node) 

    # Remove nodes from the tree that are no longer connected (if needed)
    connected_nodes = {}
    for node_name, node in tree_nodes.items():
        if node.is_connected:
            connected_nodes[node_name] = node
    print(f'Removed {len(tree_nodes) - len(connected_nodes)} unconnected nodes from the tree.')
    tree_nodes = tree_nodes

    tree = Graph(tree_nodes, get_synset_name(task['root']))
    tree.finalize()
    return tree


def main(case_study_dir, data_dir, results_dir, model, task):
    """Create data files for the language model case study."""
    print(f"EXTRACTING DATA FOR {model} ON THE {task['name']} TASK.")
    
    # Load the S-TEST data for the task.
    data = load_task_data(task['id'], data_dir)
    print(f"LOADED DATA")
    print(f"{len(data)} instances for {task['name']} prediciton task.")
    print(f"Example data:")
    print(data[list(data.keys())[0]])
    
    # Load the model results.
    with open(os.path.join(results_dir, model, task['id'], 'result.pkl'), 'rb') as f:
        results = pickle.load(f)['list_of_results']
    print(f"LOADED MODEL OUTPUTS")
    print(f"{len(results)} predictions for {model} on {task['name']} prediciton task.")
    print(f"Predictions for results[0] sum to {np.sum([np.exp(w['log_prob']) for w in results[0]['masked_topk']['topk']])}")
    print(f"Computed probabilities for {len(results[0]['masked_topk']['topk'])} words.")
    
    # Load the labels and map them to their corresponding wordnet synset.
    all_labels = [data[result['sample']['sub_label']]['obj_label'] for result in results]
    with open(os.path.join(case_study_dir, f"{task['id']}_synsets.json"), 'r') as f:
        label_synsets = json.load(f)
        label_to_synset = {label: wn.synset(synset) for label, synset in label_synsets if synset is not None}
    idx_to_keep = [i for i in range(len(all_labels)) if all_labels[i] in label_to_synset]
    print(f'Removed {len(results) - len(idx_to_keep)} instances with labels {[label for i, label in enumerate(all_labels) if i not in idx_to_keep]}')
    labels = np.array(all_labels)[idx_to_keep]
    synsets = [label_to_synset[label] for label in labels]
    print(f'Resulting dataset has {len(labels)}/{len(all_labels)} labels mapping to {len(synsets)} total/{len(set(synsets))} unique synsets')
    print(f'First 5 labels: {labels[:5]}')
    print(f'First 5 synsets: {synsets[:5]}')
    
    # Create a DAG of the WordNet concepts related to the task.
    wordnet_dag = create_wordnet_dag(task, synsets)
    print(f"CREATED WORDNET DAG with root node '{wordnet_dag.root_id}' and {len(wordnet_dag.nodes)} synset concepts.")
    
    # Convert the DAG to a tree for visualization interface.
    name_to_parent_filename = os.path.join(case_study_dir, f"{task['id']}_concept_to_parent.json")
    with open(name_to_parent_filename, 'r') as f:
        node_name_to_parent_name = json.load(f)
    print(f"Loaded node to parent mapping for {len(node_name_to_parent_name)} nodes.")
    wordnet_tree = convert_dag_to_tree(wordnet_dag, task, node_name_to_parent_name)
    wordnet_tree.check_tree()
    print(f"CREATED WORDNET TREE for {task['name']} with {len(wordnet_tree.nodes)} nodes.")
    
    # Write out the tree files.
    output_dir = 'interface/data/llm'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Write out the wordnet tree.
    serialized_wordnet_tree = wordnet_tree.serialize(False)
    with open(os.path.join(output_dir, 'abstraction_graph.json'), 'w') as f:
        json.dump(serialized_wordnet_tree, f, indent=4)
        
    # Write out the labels for each instance.
    synset_labels = [[get_synset_name(synset)] for synset in synsets]
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(synset_labels, f, indent=4)
    print(f"Dumped {len(synset_labels)} labels.")
    
    # Write out the model's outputs for each instance.
    name_to_id = {node['name']: node['id'] for node in serialized_wordnet_tree}
    model_outputs_per_instance = []
    for i in tqdm(idx_to_keep):
        result = results[i]
        outputs = {}
        for token in result['masked_topk']['topk']:
            token_prob = np.exp(token['log_prob'])
            token_word = token['token_word_form']
            if token_word in name_to_id:
                outputs[name_to_id[token_word]] = token_prob
        model_outputs_per_instance.append(outputs)
    print(f'Got outputs for {len(model_outputs_per_instance)} results.')
    print(f'Number of scores per result: {Counter([len(outputs) for outputs in model_outputs_per_instance])}')
    with open(os.path.join(output_dir, 'scores.json'), 'w') as f:
        json.dump(model_outputs_per_instance, f, indent=4)
        
    # Write out the sentence inputs for each instance.
    texts = []
    for i in tqdm(idx_to_keep):
        result = results[i]
        instance = data[result['sample']['sub_label']]
        sentences = result['sample']['masked_sentences']
        assert len(sentences) == 1
        sentence = sentences[0]
        texts.append(sentence)
    print(f"Got texts for {len(texts)} instances.")
    with open(os.path.join(output_dir, 'texts.json'), 'w') as f:
        json.dump(texts, f, indent=4)
            
    # Compute the concept coconfusion of node concepts across all instances.
    node_pairs = list(combinations([node['name'] for node in serialized_wordnet_tree], 2))
    coconfusion = metrics.concept_coconfusion(model_outputs_per_instance, task['threshold'])
    print(f'{len(coconfusion)}/{len(node_pairs)} node pairs with confusion.')               
    with open(os.path.join(output_dir, f"coconfusion.json"), 'w') as f:
        json.dump(coconfusion, f, indent=4)   


if __name__ == '__main__':
    case_study_dir = 'util/llm/'
    data_dir = os.path.join(case_study_dir, 'S-TEST', 'data/S-TEST/')
    results_dir = os.path.join(case_study_dir, 'S-TEST', 'output/results/')
    model = 'bert_base'
    task = {
        'name': 'occupation', 
        'id': 'P106', 
        'up_fn': 'hypernyms', 
        'down_fn': 'hyponyms', 
        'root': wn.synset('person.n.01'), 
        'threshold': 0.01,
    }
    main(case_study_dir, data_dir, results_dir, model, task)