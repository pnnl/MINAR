import os
import sys
sys.path.append('../')
sys.path.append('./SALSA-CLRS/')

import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from loguru import logger

from model.MinAggGNN import MinAggGNN
from model.GINE import GINE
from model.RecGINE import RecGINE
from EncodeProcessDecode import EncodeProcessDecode

from baselines.core.models.encoder import Encoder
from baselines.core.models.decoder import Decoder
from baselines.core.loss import CLRSLoss
from baselines.core.metrics import calc_metrics

from salsaclrs import specs, load_dataset
from salsaclrs.data import SALSACLRSDataLoader

# Tasks and outputs (constants can remain at module level)
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

def _to_pylist(x):
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    try:
        return [float(x)]
    except Exception:
        return []

def setup(rank, world_size):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')

    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    try:
        dist.destroy_process_group()
    except Exception:
        pass

def resume_training(checkpoint_path, local_device, model, optimizer, scheduler,
                    algorithms, logger, current_task_losses, current_val_accs, current_test_accs,
                    current_best_avg_val, current_best_epoch):
    """
    Attempt to resume training from checkpoint files in checkpoint_path.
    Returns: start_epoch, task_losses, val_accs, test_accs, best_avg_val, best_epoch
    """
    start_epoch = 0
    task_losses = current_task_losses
    val_accs = current_val_accs
    test_accs = current_test_accs
    best_avg_val = current_best_avg_val
    best_epoch = current_best_epoch
    l1_list = []

    final_ckpt = os.path.join(checkpoint_path, 'model_final.pt')
    in_prog_ckpt = os.path.join(checkpoint_path, 'model_in_progress.pt')

    if os.path.exists(in_prog_ckpt):
        try:
            # load model state (support either raw state_dict or a wrapped training_state dict)
            state = torch.load(in_prog_ckpt, map_location=local_device)
            try:
                if isinstance(state, dict) and 'model_state' in state:
                    model.load_state_dict(state['model_state'])
                else:
                    model.load_state_dict(state)
                logger.info(f'Loaded model parameters from {in_prog_ckpt} (no model_final found).')
            except Exception as e:
                logger.warning(f'Could not load model parameters from {in_prog_ckpt}: {e}')

            # attempt to also restore optimizer and scheduler state (if saved)
            try:
                training_state_path = os.path.join(checkpoint_path, 'training_state_in_progress.pt')
                training_state = None
                if os.path.exists(training_state_path):
                    training_state = torch.load(training_state_path, map_location=local_device)
                    if isinstance(training_state, dict):
                        # restore optimizer state (move tensors to current device)
                        if 'optimizer_state' in training_state and optimizer is not None:
                            try:
                                optimizer.load_state_dict(training_state['optimizer_state'])
                                for st in optimizer.state.values():
                                    for k, v in list(st.items()):
                                        if isinstance(v, torch.Tensor):
                                            st[k] = v.to(local_device)
                                logger.info(f'Loaded optimizer state from {training_state_path}')
                            except Exception as e:
                                logger.warning(f'Could not load optimizer state: {e}')

                        # restore scheduler state
                        if 'scheduler_state' in training_state and scheduler is not None:
                            try:
                                scheduler.load_state_dict(training_state['scheduler_state'])
                                logger.info(f'Loaded scheduler state from {training_state_path}')
                            except Exception as e:
                                logger.warning(f'Could not load scheduler state from {training_state_path}: {e}')

                        # restore bookkeeping (best metrics / epoch) if present
                        best_avg_val = training_state.get('best_avg_val', best_avg_val)
                        best_epoch = training_state.get('best_epoch', best_epoch)
                    else:
                        logger.warning(f'Unexpected training_state format in {training_state_path}')
            except Exception as e:
                logger.warning(f'Failed to restore optimizer/scheduler from training state: {e}')

            # attempt to also restore training progress (task_losses, val_accs, test_accs)
            tl_path = os.path.join(checkpoint_path, 'task_losses.pt')
            va_path = os.path.join(checkpoint_path, 'val_accs.pt')
            ta_path = os.path.join(checkpoint_path, 'test_accs.pt')
            loaded_task_losses = {}
            loaded_val_accs = {}
            loaded_test_accs = {}
            if os.path.exists(tl_path) and os.path.exists(va_path) and os.path.exists(ta_path):
                try:
                    loaded_task_losses = torch.load(tl_path)
                    print(f'Loaded task_losses from {tl_path} with keys: {list(loaded_task_losses.keys())}')
                    loaded_val_accs = torch.load(va_path)
                    print(f'Loaded val_accs from {va_path} with keys: {list(loaded_val_accs.keys())}')
                    loaded_test_accs = torch.load(ta_path)
                    print(f'Loaded test_accs from {ta_path} with keys: {list(loaded_test_accs.keys())}')
                except Exception as e:
                    logger.warning(f'Could not load task_losses/val_accs/test_accs: {e}')

            # normalize into dicts of python lists (one list per task)
            if loaded_task_losses:
                task_losses = {task: _to_pylist(loaded_task_losses.get(task, [])) for task in algorithms}
            else:
                task_losses = {task: _to_pylist(current_task_losses.get(task, [])) for task in algorithms}
            if loaded_val_accs:
                val_accs = {task: _to_pylist(loaded_val_accs.get(task, [])) for task in algorithms}
            else:
                val_accs = {task: _to_pylist(current_val_accs.get(task, [])) for task in algorithms}
            if loaded_test_accs:
                test_accs = {task: _to_pylist(loaded_test_accs.get(task, [])) for task in algorithms}
            else:
                test_accs = {task: _to_pylist(current_test_accs.get(task, [])) for task in algorithms}

            # If the loaded lists are shorter than the current ones, extend them with zeros to match lengths
            for task in algorithms:
                desired_len_tl = len(current_task_losses.get(task, []))
                if len(task_losses.get(task, [])) < desired_len_tl:
                    task_losses[task].extend([0.0] * (desired_len_tl - len(task_losses[task])))

                desired_len_va = len(current_val_accs.get(task, []))
                if len(val_accs.get(task, [])) < desired_len_va:
                    val_accs[task].extend([0.0] * (desired_len_va - len(val_accs[task])))

                desired_len_ta = len(current_test_accs.get(task, []))
                if len(test_accs.get(task, [])) < desired_len_ta:
                    test_accs[task].extend([0.0] * (desired_len_ta - len(test_accs[task])))

            # compute minimum available length across all tasks/records
            lengths = [len(v) for v in list(task_losses.values()) + list(val_accs.values()) + list(test_accs.values())]
            min_len = min(lengths) if lengths else 0
            print(f'Checkpoint loaded with progress for {min_len} epochs. Checking for all-zero columns to determine resume point...')

            # find first index where task_losses[:,idx] or val_accs[:,idx] or test_accs[:,idx] are all zeros
            start_epoch = 0
            for i in range(min_len):
                all_task_zero = any(float(task_losses[t][i]) == 0.0 for t in algorithms)
                all_val_zero = any(float(val_accs[t][i]) == 0.0 for t in algorithms)
                all_test_zero = any(float(test_accs[t][i]) == 0.0 for t in algorithms)
                if all_task_zero or all_val_zero or all_test_zero:
                    start_epoch = i
                    break
            else:
                # no all-zero column found -> resume after the last available epoch
                start_epoch = min_len

            # load stored l1 norms list if present
            l1_path = os.path.join(checkpoint_path, 'l1_norms.pt')
            try:
                if os.path.exists(l1_path):
                    l1_list = torch.load(l1_path)
                    # ensure python list of floats
                    l1_list = [float(x) for x in list(l1_list)]
                else:
                    l1_list = []
            except Exception as e:
                logger.warning(f'Could not load l1_norms from {l1_path}: {e}')
                l1_list = []

            # safety: ensure l1_list length matches available epochs (truncate if needed)
            if len(l1_list) > start_epoch:
                l1_list = list(l1_list)[:start_epoch]

            logger.info(f'Loaded training progress from checkpoint. Resuming from epoch index {start_epoch}.')
        except Exception as e:
            logger.warning(f'Could not load checkpoint {in_prog_ckpt}: {e}')
            start_epoch = 0
            l1_list = []
    else:
        logger.warning('No in-progress checkpoint found (or model_final exists). Starting from epoch 0.')
        start_epoch = 0

    return start_epoch, task_losses, val_accs, test_accs, best_avg_val, best_epoch

def checkpoint_and_test(epoch, avg_val_scalar, checkpoint_path, model_ddp, optimizer, scheduler,
                                local_device, task_losses, val_accs, test_accs,
                                algorithms, logger, best_avg_val, best_epoch, total_losses):
    """
    Save checkpoints, update best model, and run per-epoch test (intended to be called only on rank 0).
    Returns updated (best_avg_val, best_epoch). Mutates task_losses/val_accs/test_accs in-place.
    """
    try:
        print(f'Epoch {epoch+1}/{len(total_losses)} - Train Loss: {total_losses[epoch]:.4f} - Val Accs: ' +
                ', '.join(f'{task}: {val_accs[task][epoch]:.4f}' for task in algorithms))
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(task_losses, os.path.join(checkpoint_path, 'task_losses.pt'))
        torch.save(val_accs, os.path.join(checkpoint_path, 'val_accs.pt'))
        torch.save(test_accs, os.path.join(checkpoint_path, 'test_accs.pt'))
        # always save in-progress model
        torch.save(model_ddp.module.state_dict(), os.path.join(checkpoint_path, f'model_in_progress.pt'))
        torch.save(model_ddp.module.state_dict(), os.path.join(checkpoint_path, f'model_in_progress_{epoch}.pt'))

        # also save optimizer and scheduler state so training can be fully resumed
        try:
            training_state = {
                'epoch': epoch,
                'model_state': model_ddp.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_avg_val': best_avg_val,
                'best_epoch': best_epoch,
            }
            # compute L1 norm (sum of absolute values) of the model parameters and save it
            try:
                l1_norm = float(sum(p.abs().sum().item() for p in model_ddp.module.parameters()))
            except Exception:
                l1_norm = 0.0
                for v in model_ddp.module.state_dict().values():
                    try:
                        l1_norm += float(v.abs().sum().item())
                    except Exception:
                        pass
            training_state['l1_norm'] = l1_norm
            torch.save(training_state, os.path.join(checkpoint_path, 'training_state_in_progress.pt'))

            # also keep a per-epoch list for easy plotting/inspection
            l1_path = os.path.join(checkpoint_path, 'l1_norms.pt')
            try:
                l1_list = torch.load(l1_path) if os.path.exists(l1_path) else []
            except Exception:
                l1_list = []
            l1_list = list(l1_list)
            l1_list.append(l1_norm)
            torch.save(l1_list, l1_path)
        except Exception as e:
            logger.warning(f'Could not save training state (optimizer/scheduler): {e}')

        # update best and save best model if improved
        if avg_val_scalar > best_avg_val:
            best_avg_val = avg_val_scalar
            best_epoch = epoch
            torch.save(model_ddp.module.state_dict(), os.path.join(checkpoint_path, 'model_best.pt'))
            print(f'New best model at epoch {epoch+1} with avg_val={best_avg_val:.4f} -> saved model_best.pt')

    except Exception as e:
        logger.warning(f'Rank0 checkpoint/test helper failed: {e}')
    return best_avg_val, best_epoch

# The spawn target must be a top-level function
def main(rank, world_size, args, checkpoint_path):
    print(f'Rank {rank} training {algorithms[rank]}')
    setup(rank, world_size)
    try:
        # Device setup
        local_device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        # reproducibility
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        # Load only this process's task datasets (rank selects the task)
        local_dir = args.root
        N_each = args.num_val
        local_task = algorithms[rank]
        epochs = args.epochs

        # train/val/test loaders for the local task
        print(f'Rank {rank} loading train data for task {local_task}...')
        train_data = load_dataset(local_task, 'train', local_dir)
        local_train_loader = SALSACLRSDataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=1)

        print(f'Rank {rank} loading val data for task {local_task}...')
        val_data = load_dataset(local_task, 'val', local_dir)
        local_val_loader = SALSACLRSDataLoader(val_data, batch_size=N_each, num_workers=1)

        print(f'Rank {rank} loading test data for task {local_task}...')
        test_data = load_dataset(local_task, 'test', local_dir)
        local_test_loader = SALSACLRSDataLoader(test_data['er_80'], batch_size=N_each, num_workers=1)

        logger.disable('baselines.core.models.encoder')
        logger.disable('baselines.core.models.decoder')

        # Model components
        hidden_dim = args.hidden_dim
        encoders = torch.nn.ModuleDict({
            task : Encoder(specs=specs.SPECS[task]) for task in algorithms
        })
        decoders = torch.nn.ModuleDict({
            task : Decoder(specs=specs.SPECS[task], hidden_dim = hidden_dim * 2, no_hint=False)
            for task in algorithms
        })
        for encoder in encoders.values():
            encoder.to(local_device)
        for decoder in decoders.values():
            decoder.to(local_device)
            
        if args.model == 'GINE':
            processor = GINE(3 * hidden_dim, hidden_dim, 2, hidden_dim, edge_dim=1, aggr='max')
        elif args.model == 'RecGINE':
            processor = RecGINE(3 * hidden_dim, hidden_dim, 2, hidden_dim, edge_dim=1, aggr='max')
        else:
            raise NotImplementedError
        processor.to(local_device)
        
        model = EncodeProcessDecode(encoders, decoders, processor, device=local_device)

        criteria = torch.nn.ModuleDict({
            task : CLRSLoss(specs.SPECS[task], 'l2') for task in algorithms
        })

        lambda_hint = args.lambda_hint
        lambda_hidden = args.lambda_hidden
        eta = args.eta
        weight_decay = args.weight_decay
        epochs = args.epochs

        # Wrap model for DDP
        model.to(local_device)
        model_ddp = DDP(model, device_ids=[rank], find_unused_parameters=True)
        # optimizer = torch.optim.Adam(model_ddp.parameters(), lr=args.lr, weight_decay=eta)
        optimizer = torch.optim.AdamW(model_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr / 100)

        # Storage for aggregated results (shared via all_gather)
        total_losses = [0 for _ in range(epochs)]
        task_losses = {task : [0 for _ in range(epochs)] for task in algorithms}
        val_accs = {task : [0 for _ in range(epochs)] for task in algorithms}
        test_accs = {task : [0 for _ in range(epochs)] for task in algorithms}
        best_avg_val = float('-inf')
        best_epoch = -1

        # Call resume helper to possibly restore model/optimizer/scheduler and progress
        start_epoch, task_losses, val_accs, test_accs, best_avg_val, best_epoch = resume_training(
            checkpoint_path, local_device, model, optimizer, scheduler,
            algorithms, logger, task_losses, val_accs, test_accs, best_avg_val, best_epoch
        )

        def train_one_epoch(loader):
            # with torch.autograd.detect_anomaly():
                model_ddp.train()
                running_out_sum = torch.tensor(0., device=local_device)
                num_batches = 0
                last_task_batch_loss = 0.0
                with torch.no_grad():
                    cos_schedule = torch.cos(torch.tensor(epoch / epochs * torch.pi))
                    l1_scaler = eta * (1 - cos_schedule)
                    for param_group in optimizer.param_groups:
                        param_group['weight_decay'] = weight_decay * cos_schedule
                for batch in loader:
                    num_batches += 1
                    batch.to(local_device)
                    batch.task = local_task
                    if hasattr(batch, 'weights'):
                        batch.edge_attr = batch.weights.unsqueeze(1)
                    else:
                        batch.edge_attr = torch.zeros((batch.num_edges, 1), device=local_device)

                    optimizer.zero_grad()
                    out, hints, hidden = model_ddp(batch)
                    out_loss, hint_loss, hidden_loss = criteria[local_task](batch, out, hints, hidden)

                    total_batch_loss = out_loss.sum() + lambda_hint * hint_loss.sum() + lambda_hidden * hidden_loss.sum()
                    l1_norm = sum(torch.abs(p).sum() for p in model_ddp.module.processor.parameters())
                    total_batch_loss = total_batch_loss + l1_scaler * l1_norm

                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_ddp.parameters(), max_norm=10.0)
                    optimizer.step()

                    running_out_sum += out_loss.sum().detach()
                    try:
                        last_task_batch_loss = out_loss.item()
                    except:
                        last_task_batch_loss = out_loss.sum().item()

                if num_batches == 0:
                    avg_out = torch.tensor(0., device=local_device)
                else:
                    avg_out = running_out_sum / float(num_batches)
                return avg_out, last_task_batch_loss

        def evaluate(loader):
            model_ddp.eval()
            with torch.no_grad():
                accuracies = []
                for batch in loader:
                    batch.to(local_device)
                    batch.task = local_task
                    if hasattr(batch, 'weights'):
                        batch.edge_attr = batch.weights.unsqueeze(1).to(local_device)
                    else:
                        batch.edge_attr = torch.zeros((batch.num_edges, 1), device=local_device)
                    out, hints, hidden = model_ddp(batch)
                    metrics = calc_metrics(batch.outputs[0], out, batch, output_types[local_task])
                    accuracies.append(metrics['node_accuracy'].to(local_device))
                if len(accuracies) == 0:
                    return torch.tensor(0., device=local_device)
                return torch.stack(accuracies).mean()

        # Training loop
        for epoch in range(start_epoch, epochs):
            local_avg_out, local_task_last = train_one_epoch(local_train_loader)

            # gather training losses from all ranks
            gather_list = [torch.zeros_like(local_avg_out) for _ in range(world_size)]
            dist.all_gather(gather_list, local_avg_out)
            gathered_train = torch.stack(gather_list)
            total_losses[epoch] = gathered_train.sum().item()
            for i, t in enumerate(algorithms):
                task_losses[t][epoch] = gathered_train[i].item()

            # validation
            local_val = evaluate(local_val_loader)
            gather_val = [torch.zeros_like(local_val) for _ in range(world_size)]
            dist.all_gather(gather_val, local_val)
            gathered_val = torch.stack(gather_val)
            for i, t in enumerate(algorithms):
                val_accs[t][epoch] = gathered_val[i].item()

            # scheduler step on average val (run on all ranks for consistency)
            avg_val = gathered_val.mean()
            # scheduler.step(avg_val.item() if isinstance(avg_val, torch.Tensor) else float(avg_val))
            scheduler.step()

            # determine scalar avg_val for comparisons
            avg_val_scalar = avg_val.item() if isinstance(avg_val, torch.Tensor) else float(avg_val)

            local_test = evaluate(local_test_loader)
            gather_test = [torch.zeros_like(local_test) for _ in range(world_size)]
            dist.all_gather(gather_test, local_test)
            gathered_test = torch.stack(gather_test)
            for i, t in enumerate(algorithms):
                test_accs[t][epoch] = gathered_test[i].item()

            # Replace the original in-main helper with a simple call to the top-level helper:
            if rank == 0:
                best_avg_val, best_epoch = checkpoint_and_test(
                    epoch, avg_val_scalar, checkpoint_path, model_ddp, optimizer, scheduler,
                    local_device, task_losses, val_accs, test_accs,
                    algorithms, logger, best_avg_val, best_epoch, total_losses
                )

        # After training, rank 0 saves final model
        if rank == 0:
            final_ckpt = os.path.join(checkpoint_path, 'model_in_progress.pt')
            if os.path.exists(final_ckpt):
                model_ddp.module.load_state_dict(torch.load(final_ckpt, map_location='cpu'))
                torch.save(model_ddp.module.state_dict(), os.path.join(checkpoint_path, 'model_final.pt'))
            else:
                torch.save(model_ddp.module.state_dict(), os.path.join(checkpoint_path, 'model_final.pt'))

    finally:
        cleanup()

# wrapper to pin each spawned process to a single CPU core and limit intra-op threads
def _entry(rank, world_size, args, checkpoint_path, cores):
    core_id = cores[rank % len(cores)]
    try:
        # Linux: bind this process to a single CPU core
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {core_id})
    except Exception:
        pass

    # limit threading libraries to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # call the original main
    main(rank, world_size, args, checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for GNNs on SALSA-CLRS")
    parser.add_argument("--lr", type=float, required=True, help="Initial learning rate.")
    parser.add_argument("--eta", type=float, required=True, help="L1 Regularization parameter.")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay parameter.")
    parser.add_argument("--model", type=str, default="GINE", help="Model name to use.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="Hidden dimension size.")
    parser.add_argument("--lambda_hint", default=1.0, type=float, help="Hint weight.")
    parser.add_argument("--lambda_hidden", default=0.1, type=float, help="Hidden embedding weight.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size.")
    parser.add_argument("--num_val", default=1000, type=int, help="Number of test and validation samples.")
    parser.add_argument("--epochs", default=100, type=int, help="Epochs to train.")
    parser.add_argument("--devices", nargs='+', type=int, default=None,
                        help="List of CUDA device indices to use (e.g. --devices 0 1 2).")
    parser.add_argument("--seed", default=42, type=int, help="Manual Seed.")
    parser.add_argument("--root", default='./data/', type=str, help="Root directory to load data.")
    args = parser.parse_args()

    # Ensure the number of CUDA devices matches the number of tasks/algorithms
    if not torch.cuda.is_available():
        raise AssertionError(f"CUDA is not available but {len(algorithms)} CUDA devices are required for distributed training.")
    num_cuda = torch.cuda.device_count()
    assert num_cuda == len(algorithms), (
        f"Expected exactly {len(algorithms)} CUDA devices (one per algorithm), but found {num_cuda}. "
        "Adjust CUDA_VISIBLE_DEVICES or your hardware configuration."
    )

    config_str = f'distributed_{args.model}_l1_schedule_lr={args.lr}_eta={args.eta}_weight_decay={args.weight_decay}_batch_size={args.batch_size}_seed={args.seed}'
    checkpoint_path = f'./checkpoints/{config_str}/'
    print(f'Saving to {checkpoint_path}')
    world_size = len(algorithms)

    # choose CPU cores to use (one core per algorithm)
    cpu_count = os.cpu_count() or 1
    if cpu_count < world_size:
        print(f'Warning: only {cpu_count} CPU cores available but {world_size} processes requested. '
              'Cores will be reused.')
    cores = list(range(cpu_count))

    mp.set_start_method('spawn', force=True)
    mp.spawn(_entry, args=(world_size, args, checkpoint_path, cores), nprocs=world_size, join=True)
