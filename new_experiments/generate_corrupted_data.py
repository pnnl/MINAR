import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')
from salsaclrs import SALSACLRSDataset
from salsaclrs.data import SALSA_CLRS_DATASETS

import os
import torch

algorithms = ['bfs', 'dfs', 'dijkstra', 'mst_prim', 'bellman_ford', 'articulation_points', 'bridges']
local_dir = './data/'
clean_data_dir = './data_clean/'
corr_data_dir = './data_corrupted/'

for algorithm in algorithms:
    print(f'Generating corrupted data for {algorithm}...')
    os.makedirs(f'{clean_data_dir}/{algorithm}', exist_ok=True)
    os.makedirs(f'{corr_data_dir}/{algorithm}', exist_ok=True)
    test_data = SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="test",
                                algorithm=algorithm, num_samples=1000, graph_generator="er",
                                graph_generator_kwargs=SALSA_CLRS_DATASETS["test"]["er_16"],
                                nickname="er_16")
    for i, data in enumerate(test_data):
        torch.save(data, f'{clean_data_dir}/{algorithm}/data_{i}.pt')
        for key in data.inputs:
            corrupted_input = torch.zeros_like(data[key])
            data[key] = corrupted_input
        if hasattr(data, 'weights') :
            corrupted_weights = torch.zeros_like(data.weights)
            data.weights = corrupted_weights
        torch.save(data, f'{corr_data_dir}/{algorithm}/data_{i}.pt')