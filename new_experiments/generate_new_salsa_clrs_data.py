import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')
from salsaclrs import SALSACLRSDataset
from salsaclrs.data import SALSA_CLRS_DATASETS

algorithms = ['bellman_ford', 'articulation_points', 'bridges']
local_dir = './data/'
cores = -1 # Don't use multiprocessing - this avoids issues with random seeding across processes
seed = 42

for algorithm in algorithms:
    for k in SALSA_CLRS_DATASETS["test"]:
        if k[-2:] == '16' or k[-2:] == '80' or k[-3:] == '160':
            SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="test",
                             algorithm=algorithm, num_samples=1000, graph_generator=k.split("_")[0],
                             graph_generator_kwargs=SALSA_CLRS_DATASETS["test"][k],
                             nickname=k, max_cores=cores,
                             **{'seed': seed})
    print(f"Generated {algorithm} test set")
    
    SALSACLRSDataset(ignore_all_hints=True, root=local_dir, split="val", algorithm=algorithm,
                     num_samples=1000, graph_generator="er",
                     graph_generator_kwargs=SALSA_CLRS_DATASETS["val"], max_cores=cores,
                     **{'seed': seed})
    print(f"Generated {algorithm} val set")
    SALSACLRSDataset(ignore_all_hints=False, root=local_dir, split="train", algorithm=algorithm,
                     num_samples=10000, graph_generator="er",
                     graph_generator_kwargs=SALSA_CLRS_DATASETS["train"], max_cores=cores,
                     **{'seed': seed})
    print(f"Generated {algorithm} train set")