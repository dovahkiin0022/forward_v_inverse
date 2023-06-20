# Comparing Forward and Inverse Design Paradigms: A Case Study on Refractory High-Entropy Alloys

by Arindam Debnath, Lavanya Raman, Wenjie Li, Adam M. Krajewski, Marcia Ahn, Shuang Lin, Shunli Shang, Allison M. Beese, Zi-Kui Liu, and Wesley F. Reinhart


This repository is the official implementation of the paper 'Comparing Forward and Inverse Design Paradigms: A Case Study on Refractory High-Entropy Alloys', which has been submitted for publication in Journal of Materials Research

In this paper, we have considered some case studies for comparing forward vs inverse design methods using the design of novel High Entropy Alloys (HEAs) as the material of interest.

# Getting the code

You can download a copy of all the files in this repository by cloning the git repository:

```
git clone https://github.com/dovahkiin0022/forward_v_inverse.git
```

A copy of the repository is also archived at [![DOI](https://zenodo.org/badge/561584458.svg)](https://zenodo.org/badge/latestdoi/561584458)



# Installation

Run the following command to install the required dependencies specified in the file `environment.yml`
```
conda env create -f environment.yml
```

# Implementation

The repository contains the following directories:

```
.
├── credentials.json
├── dataset
│   ├── PureElements.csv
│   ├── synthetic_dataset.csv
│   ├── ultimate tensile strength.csv
├── environment.yml
├── figures
├── misc
├── modules
│   ├── functions.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── mongodb_rom.py
│   ├── pytorch_models.py
│   └── trained_models.py
├── notebooks
│   ├── forward_single.ipynb
│   ├── inverse_multi.ipynb
│   ├── inverse_single.ipynb
│   ├── moo_multi.ipynb
│   ├── moo_single.ipynb
│   ├── train_cGAN_multi.ipynb
│   ├── train_cGAN_single.ipynb
│   ├── train_uts_model.ipynb
│   └── visualization.ipynb
├── README.md
├── results.json
└── saved_models

```

## Credentials

The credential file contains the username and password necessary for accessing the ULTERA MongoDB database. Currently, the database is restricted to limited users, so the credential file contains a dummy username and password. In case you are interested in using the database, you can reach out to [ak@psu.edu](mailto:ak@psu.edu)

## Dataset 

The `dataset` folder contains the following files - 

* PureElements.csv - A csv file containing the properties of the periodic table elements
* synthetic_dataset.csv - The synthetic dataset that has been used to training the conditional GAN and for high-throughput search.
* ultimate_tensile_strength.csv - The ultimate tensile strength dataset used for training the surrogate model

## Figures

All the figures depicting the results from the various tests performed are contained in this folder.

## Miscellaneous 

The information necessary for reproducing the results in the paper have been stored in the `misc` folder as pickle or json files.

## Modules

The `modules` folder contains several .py files with necessary functions and codes for the project.

* function.py - Contains some general functions (like converting string compositions to Pymatgen Composition objects)
* metrics.py - Contains functions for metrics for evaluating performances.
* mongodb_rom.py - Contains functions for calculating the value of properties from linear combination of elemental DFT properties from the ULTERA MongoDB database.
* model_select.py - Contains code to evaluate several out-of-the-box `scikit-learn` regression models for the regression task using Root Mean Squared Error and Pearson's Correlation Coefficient.
* pytorch_models.py - Contains the pytorch model for the conditional GAN
* trained_models.py - Contains a function to obtain the UTS at 1200 &deg;C from the trained surrogate model

## Notebooks

The `notebook` folder contains the Jupyter notebooks used for running the codes for the different tests. 

## Results

The results from the different Jupyter notebooks are stored as json objects in the `results` folder.

## Saved models

The single-task models are stored as `pytorch` objects in the `saved_models` folder.
