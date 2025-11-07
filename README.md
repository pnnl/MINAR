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

---
### DISCLAIMER
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
