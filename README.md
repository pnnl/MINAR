# MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning

This contains the code for the submission MINAR: Mechanistic Interpretability for Neural Algorithmic Reasoning to ICML 2026. The code was developed and tested using Python 3.12.3. The package requirements are listed in `environment.yml`.

## Instructions

### Bellman-Ford

Data generation can be found in the folder `bellman_ford_experiments/data`.

Bellman-Ford experiments are divided into subfolders. Each subfolder contains 
1. A training notebook
2. A circuit analysis notebook, and
3. A plotting notebook.
In each notebook, simply run all cells. Note that model training may take some time (up to 30 min on the hardware reported in the paper).

### SALSA-CLRS
The SALSA-CLRS experiments are contained in the folder `new_experiments`.
1. Data for BFS, DFS, Dijkstra's algorithm, and Prim's MST are from SALSA-CLRS (https://arxiv.org/abs/2309.12253)
2. A data generation script for Bellman-Ford, Articulation Points, and Bridges is provided in `generate_new_salsa_clrs_data.py`.
3. Clean/corrupted data generation is in `generate_corrupted_data.py`.
4. The model training script used is `train_salsa_clrs_distributed_l1_schedule.py`. (NOTE: The script assumes access to 7 GPUs to perform multi-GPU parallel training.)
5. Plotting of model progress is in `salsa_clrs_plots.ipynb`
6. Circuit analysis is performed in `salsa-clrs_circuits.ipynb`. Since identifying large circuits can be time-consuming, an alternative script is given in `compute_salsa-clrs_circuits.py` which can be run in the background.

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