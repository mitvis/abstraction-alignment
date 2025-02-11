# Abstraction Alignment: Comparing Model-Learned and Human-Encoded Conceptual Relationships

This repo contains code to recreate the experiments in Abstraction Alignment: Comparing Model-Learned and Human-Encoded Conceptual Relationships.
To explore the case studies in an live interactive interface instead, check out the [abstraction alignment interface](http://128.52.138.87:8000/).

To explore abstraction alignment, check out the example notebooks. 
In [`cifar_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/cifar_abstraction_alignment.ipynb) we use abstraction alignment to interpret image classification models, in [`language_model_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/language_model_abstraction_alignment.ipynb) we benchmark langugage model specificity, and in [`mimic_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/mimic_abstraction_alignment.ipynb) we analyze medical dataset encodings.

## Repository Structure

```
# Data storage
./cifar/ # Stores the CIFAR model and output data files
./wordnet/ # Stores the S-TEST data and the output data files
./mimic/ # Stores the MIMIC-III data and the output data files

# Code to extract interface data files
./extract_cifar_data.py
./extract_mimic_data.py
./extract_wordnet_data.py

# Notebooks to explore abstraction alignment
./cifar_abstraction_alignment.ipynb
./language_model_abstraction_alignment.ipynb
./mimic_abstraction_alignment.ipynb

# Supporting data files to compute abstraction alignment
metrics.py # abstraction alignment metrics
...
```

## Set Up

### Download the case study data
#### CIFAR-100
The CIFAR-100 data will autmatically download during analysis in the notebooks.

#### WordNet and S-TEST
1. Follow instructions to download the [S-TEST dataset](https://github.com/jeffhj/S-TEST). Put it in a folder called `S-TEST` in the `abstraction-alignment` repo.
2. Run `python S-TEST/scripts/run_experiments.py` to compute the model's output on the data.
3. Update the paths in `./extract_wordnet_data.py` to match your file structure.

#### MIMIC-III
1. Request access to MIMIC-III via [PhysioNet](https://physionet.org/content/mimiciii/1.4/).
2. Download the MIMIC-III dataset and update the paths in `./extract_mimic_data.py` to point to it.

## Usage

### Exploring Abstraction Alignment
To explore abstraction alignment, check out the example notebooks. 
In [`cifar_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/cifar_abstraction_alignment.ipynb) we use abstraction alignment to interpret image classification models, in [`language_model_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/language_model_abstraction_alignment.ipynb) we benchmark langugage model specificity, and in [`mimic_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/mimic_abstraction_alignment.ipynb) we analyze medical dataset encodings.

### Running the Abstraction Alignment Interface Locally
The code in `extract_{cifar/wordnet/mimic}_data.py` creates the data files needed to run the abstraction alignment interface. Run these if you'd like to run the abstraction alignment interface locally or reference them for the data file set up to run the interface with your own data.
