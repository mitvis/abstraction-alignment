# Abstraction Alignment: Comparing Model-Learned and Human-Encoded Conceptual Relationships

This repo contains code to recreate the experiments in Abstraction Alignment: Comparing Model-Learned and Human-Encoded Conceptual Relationships.
To explore the case studies in an live interactive interface instead, check out the abstraction alignment interface.

Each of the experiments is contained within its own notebook:
* [`cifar_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/cifar_abstraction_alignment.ipynb) --- section 4.1. Interpreting model behavior with abstraction alignment
* [`language_model_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/language_model_abstraction_alignment.ipynb) --- section 4.2. Benchmarking language models’ abstraction alignment
* [`mimic_abstraction_alignment.ipynb`](https://github.com/mitvis/abstraction-alignment/blob/main/mimic_abstraction_alignment.ipynb) --- section 4.3. Analyzing datasets using abstraction alignment


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
```

## Set Up

### Download the case study data
#### CIFAR-100
The CIFAR-100 data will autmatically download during analysis in the notebooks.

#### WordNet and S-TEST

#### MIMIC-III


## Usage

### Exploring Abstraction Alignment

### Running the Abstraction Alignment Interface Locally
The code in `extract_{cifar/wordnet/mimic}_data.py` creates the data files needed to run the abstraction alignment interface. Run these if you'd like to run the abstraction alignment interface locally or reference them for the data file set up to run the interface with your own data.