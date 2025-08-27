# MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning

This contains the code for the submission MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning to the New Perspectives in Advancing Graph Machine Learning Workshop at NeurIPS 2025. The code was developed and tested using Python 3.12.3. The package requirements are listed in the requirements.txt file.

## Instructions

In each notebook, simply run all cells. Note that model training may take some time (up to 30 min on the hardware reported in the paper).

### Generating Data

To generate the datasets used in the paper, run the `generate_data.ipynb` notebook in the `data/` folder.

### Training Models

Model training and related plots are included in the notebooks `train_gnn_bellman_ford.ipynb` and `train_gnn_parallel.ipynb`.

### Circuit Discovery

Circuit discovery experiments and related plots are included in the notebooks `circuits_bellman_ford.ipynb` and `circuits_parallel.ipynb`.