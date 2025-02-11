# Get CIFAR Image Classification data for the Abstraction Alignment interface.

import os
import json
import torch
import argparse
import torchvision
import numpy as np
from tqdm import tqdm
from itertools import combinations

import metrics
from abstraction_graph_cifar import make_abstraction_graph, serialize_abstraction_graph, propagate
from util.cifar.cifar_util import load_model, cifar_test_transform, load_dataset
from util.cifar.cifar_metadata import CLASS_LABELS


def main(model_path, data_dir):
    """Create data files for the CIFAR-100 classification case study."""
    print('CREATING CIFAR-100 DATA FOR ABSTRACTION ALIGNMENT.')
    
    # Load a trained CIFAR-100 model
    checkpoint = os.path.join(model_path, 'checkpoints', 'checkpoint.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = model_path.split('/')[-1].split('_')[0]
    model = load_model(architecture)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model = model.eval()
    
    # Load the CIFAR data
    _, dataloader = load_dataset(
        data_dir, 
        data_augmentation=True,
        batch_size=128
    )
    dataset = dataloader.dataset
    
    # Compute the CIFAR model outputs
    outputs = []
    labels = []
    for i, (images, label_batch) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images = images.to(device)
        output = model(images)
        output = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        outputs.append(output)
        labels.extend(label_batch.numpy())
    outputs = np.vstack(outputs)
    
    # Make the CIFAR abstraction graph
    abstraction_graph = make_abstraction_graph()
    print(f'Root Node: {abstraction_graph.root}. Number of nodes in abstraction_graph: {len(abstraction_graph)}')
    
    # Write out the data files
    output_dir = 'interface/data/cifar/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Write out the CIFAR abstraction_graph
    serialized_abstraction_graph = serialize_abstraction_graph(abstraction_graph)
    with open(os.path.join(output_dir, 'abstraction_graph.json'), 'w') as f:
        json.dump(serialized_abstraction_graph, f, indent=4)
        
    # Write out the CIFAR labels
    label_names = [[CLASS_LABELS[label]] for label in labels]
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(label_names, f, indent=4)
    print(f"Dumped {len(label_names)} labels.")
    
    # Write out the CIFAR scores
    name_to_id = {node['name']: node['id'] for node in serialized_abstraction_graph}
    scores = []
    for output in outputs:
        instance_scores = {}
        for i, score in enumerate(output):
            label_name = CLASS_LABELS[i]
            node_id = name_to_id[label_name]
            instance_scores[node_id] = float(score)
        scores.append(instance_scores)
    with open(os.path.join(output_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)
        
    # Write out the CIFAR concept coconfusion
    fitted_abstractions = []
    for i in tqdm(range(len(labels))):
        fitted_abstraction = propagate(outputs[i], make_abstraction_graph())
        fitted_abstractions.append(fitted_abstraction)
    coconfusion = metrics.concept_coconfusion(fitted_abstractions, 0.00001)
    coconfusion_ids = {}
    for pair, value in coconfusion.items():
        node_a, node_b = pair.split(',')
        node_a_id = name_to_id[node_a]
        node_b_id = name_to_id[node_b]
        coconfusion_ids[f'{node_a_id},{node_b_id}'] = value
    print(f'{len(coconfusion_ids)} node pairs with confusion.')               
    with open(os.path.join(output_dir, f"coconfusion.json"), 'w') as f:
        json.dump(coconfusion_ids, f, indent=4)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create CIFAR-100 data files for Abstraction Alignment.')
    parser.add_argument('--model_path', type=str, default='util/cifar/resnet20',
                        help='Path to the model directory containing checkpoints/checkpoint.pt')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory path for CIFAR-100 dataset') # e.g., '/nobackup/users/aboggust/data/cifar'
    
    # Parse arguments
    args = parser.parse_args()
    args.model_path = os.path.normpath(args.model_path)
    args.data_dir = os.path.normpath(args.data_dir)
    
    # Call main with parsed arguments
    main(args.model_path, args.data_dir)
