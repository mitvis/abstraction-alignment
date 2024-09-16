# Get CIFAR Image Classification data for the Abstraction Alignment interface.

import os
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from itertools import combinations

import metrics
from cifar import make_tree, serialize_tree, propagate
from cifar_util import load_model, cifar_test_transform
from cifar_metadata import CLASS_LABELS


def main(model_path, data_dir):
    """Create and save the CIFAR-100 data files."""
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
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=cifar_test_transform(),
        download=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=128, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
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
    
    # Make the CIFAR tree
    tree = make_tree()
    print(f'Root Node: {tree.root}. Number of nodes in tree: {len(tree)}')
    
    # Write out the data files
    output_dir = 'cifar/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Write out the CIFAR hierarchy
    serialized_tree = serialize_tree(tree)
    with open(os.path.join(output_dir, 'hierarchy.json'), 'w') as f:
        json.dump(serialized_tree, f, indent=4)
        
    # Write out the CIFAR labels
    label_names = [[CLASS_LABELS[label]] for label in labels]
    with open(os.path.join(output_dir, 'labels.json'), 'w') as f:
        json.dump(label_names, f, indent=4)
    print(f"Dumped {len(label_names)} labels.")
    
    # Write out the CIFAR scores
    name_to_id = {node['name']: node['id'] for node in serialized_tree}
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
        
    # Write out the CIFAR concept joint entropy
    fitted_abstractions = []
    for i in tqdm(range(len(labels))):
        tree = propagate(outputs[i], make_tree())
        fitted_abstractions.append(tree)
    joint_entropy = metrics.joint_entropy(fitted_abstractions, 0.00001)
    print(f'{len(joint_entropy)} node pairs with confusion.')               
    with open(os.path.join(output_dir, f"joint_entropy.json"), 'w') as f:
        json.dump(joint_entropy, f, indent=4)  


if __name__ == '__main__':
    model_path = 'cifar/resnet20'
    data_dir = '/nobackup/users/aboggust/data/cifar'
    main(model_path, data_dir)