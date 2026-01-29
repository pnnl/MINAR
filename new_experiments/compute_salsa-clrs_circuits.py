import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')

import os
import argparse

import torch
from model.MinAggGNN import MinAggGNN
from model.GINE import GINE
from model.RecGINE import RecGINE

from SALSACLRSComputationGraph import SALSACLRSComputationGraph
from SALSACLRSComputationGraph import SALSACLRSCircuit
import networkx as nx

from loguru import logger
from baselines.core.models.encoder import Encoder
from baselines.core.models.decoder import Decoder, grab_outputs, output_mask
from baselines.core.models.processor import Processor
from baselines.core.loss import CLRSLoss
from salsaclrs import specs
from baselines.core.metrics import calc_metrics
from salsaclrs.data import SALSACLRSDataLoader
from torch_geometric.loader import DataLoader

args = argparse.ArgumentParser()
args.add_argument('--lr', type=float, default=0.001, help='Learning rate')
args.add_argument('--eta', type=float, default=0.001, help='L1 regularization coefficient')
args.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
args.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
args.add_argument('--K', type=int, default=100, help='Number of paths to add the circuit')
args.add_argument('--score', type=str, default='weight', help='Scoring method to use')
args = args.parse_args()

device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
lr = args.lr
eta = args.eta
weight_decay = args.weight_decay

from EncodeProcessDecode import EncodeProcessDecode
algorithms = ['bfs', 'dfs', 'dijkstra', 'mst_prim', 'bellman_ford', 'articulation_points', 'bridges']
output_types = {
    'bfs' : 'pointer',
    'dfs' : 'pointer',
    'dijkstra' : 'pointer',
    'mst_prim' : 'pointer',
    'bellman_ford' : 'pointer',
    'articulation_points' : 'mask',
    'bridges' : 'pointer',
}
logger.disable('baselines.core.models.encoder')
logger.disable('baselines.core.models.decoder')

hidden_dim = 128
encoders = torch.nn.ModuleDict({
    task : Encoder(specs=specs.SPECS[task]) for task in algorithms
})

decoders = torch.nn.ModuleDict({
    task : Decoder(specs=specs.SPECS[task], 
                   hidden_dim = hidden_dim * 2,
                   no_hint=False) for task in algorithms
})

for encoder in encoders.values():
    encoder.to(device)
for decoder in decoders.values():
    decoder.to(device)
processor = GINE(3*128, 128, 2, 128, edge_dim=1, aggr='max')
processor.to(device)
model = EncodeProcessDecode(encoders, decoders, processor, device=device)

model_checkpoint = f'distributed_GINE_l1_schedule_lr={lr}_eta={eta}_weight_decay={weight_decay}_batch_size=32_seed=0/model_best.pt'
model_state = torch.load(f'checkpoints/{model_checkpoint}', map_location=device)
model.load_state_dict(model_state)
model.eval()
model.to(device)

G = SALSACLRSComputationGraph(model, special_modules=['convs.0.lin', 'convs.1.lin'])
G.add_module('convs.0.lin', processor.convs[0].lin,
            module_inputs='edge_attr',
            module_outputs=0,
            layer=0)
G.add_module('convs.1.lin', processor.convs[1].lin,
            module_inputs='edge_attr',
            module_outputs=2,
            layer=0)
G.correct_layers()

G_save_string = f'G_scores_lr={lr}_eta={eta}_weight_decay={weight_decay}.pt'
G_scores = torch.load(f'scored_computation_graphs/{G_save_string}', weights_only=False, map_location=device)
for edge, data in G_scores.items():
    G.add_edge(*edge, **data)

circuit_path = f'salsa_clrs_circuits_lr={lr}_eta={eta}_weight_decay={weight_decay}_K={args.K}_score={args.score}.pt'
circuits = {}
if os.path.exists(f'circuits/{circuit_path}'):
    circuits = torch.load(f'circuits/{circuit_path}', weights_only=False, map_location=device)

K = args.K
score_method = args.score

if score_method == 'weight':
    weight_circuit = SALSACLRSCircuit(model, G, K, key='weight')

for algorithm in algorithms:
    if (algorithm, K, score_method) in circuits:
        print(f'Circuit for {algorithm} with K={K} and score method {score_method} already exists. Skipping...')
    elif score_method == 'weight':
        circuits[(algorithm, K, score_method)] = weight_circuit
        print(f'{algorithm} circuit with K={K} and score method {score_method}: {circuits[(algorithm, K, score_method)].number_of_edges()} edges')
    else:
        circuits[(algorithm, K, score_method)] = SALSACLRSCircuit(model, G, K, key=f'{score_method}_{algorithm}')
        print(f'{algorithm} circuit with K={K} and score method {score_method}: {circuits[(algorithm, K, score_method)].number_of_edges()} edges')

# Remove model and G references to avoid pickling the model
for circuit in circuits.values():
    circuit.G = None
    circuit.model = None
    circuit.EncodeProcessDecode = None
torch.save(circuits, f'circuits/{circuit_path}')