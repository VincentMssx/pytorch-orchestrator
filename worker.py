import argparse
import datetime
import glob
import os
import re
import time
import traceback
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# Import the model definition
from models.SimpleModel import SimpleModel

# --- Constants ---
DEFAULT_MASTER_PORT = '29500'
DEFAULT_BACKEND = 'gloo'
CHECKPOINT_FILENAME_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")


# --- Checkpoint Handling ---
def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Finds the latest checkpoint file in the given directory based on the step number
    encoded in the filename (e.g., 'checkpoint_step_100.pt').

    Args:
        checkpoint_dir: The directory containing checkpoint files.

    Returns:
        The full path to the latest checkpoint file, or None if no valid
        checkpoints are found or the directory doesn't exist.
    """
    latest_checkpoint_path = None
    latest_step = -1

    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint directory '{checkpoint_dir}' not found or is not a directory.")
        return None

    try:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    except OSError as e:
        print(f"Error accessing checkpoint directory '{checkpoint_dir}': {e}")
        return None

    if not checkpoint_files:
        print(f"No files matching 'checkpoint_step_*.pt' found in '{checkpoint_dir}'.")
        return None

    print(f"Found {len(checkpoint_files)} potential checkpoint files. Parsing...")
    for ckpt_file_path in checkpoint_files:
        filename = os.path.basename(ckpt_file_path)
        match = CHECKPOINT_FILENAME_PATTERN.search(filename)
        if match:
            try:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint_path = ckpt_file_path
            except ValueError:
                print(f"Warning - could not parse step number as integer from '{filename}'. Skipping.")
                continue

    if latest_checkpoint_path:
        print(f"Identified latest checkpoint '{latest_checkpoint_path}' at step {latest_step}.")
    else:
        print(f"Found potential checkpoint files, but couldn't identify the latest valid one based on step number pattern.")

    return latest_checkpoint_path


# --- Distributed Setup ---
def setup(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = DEFAULT_BACKEND):
    """
    Initializes the distributed process group.
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    print(f"Rank {rank}: Initializing process group with master {master_addr}:{master_port} (backend: {backend})...")
    init_timeout = datetime.timedelta(seconds=60)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=init_timeout)
    print(f"Rank {rank}: Process group initialized.")


def cleanup():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print(f"Cleaned up process group.")
    else:
        print("Cleanup: Process group was not initialized or already destroyed.")


# --- Gradient Averaging ---
def average_gradients(model: nn.Module, world_size: int):
    """
    Averages gradients across all workers manually using all_reduce.
    """
    if world_size <= 1:
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


# --- Checkpointing ---
def save_checkpoint(state: Dict[str, Any], filename: str):
    """
    Saves checkpoint state dictionary to a file.
    This function should typically only be called by rank 0.
    """
    rank = dist.get_rank()
    print(f"Rank {rank}: Attempting to save checkpoint to {filename}")

    checkpoint_dir = os.path.dirname(filename)
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Rank {rank}: Created checkpoint directory {checkpoint_dir}")
        except OSError as e:
            print(f"Rank {rank}: ERROR - Could not create checkpoint directory: {e}")
            return

    try:
        torch.save(state, filename)
        print(f"Rank {rank}: Checkpoint saved successfully to {filename}")
    except Exception as e:
        print(f"Rank {rank}: ERROR - Could not save checkpoint: {e}")


# --- Training Logic ---
def run_training(rank: int, world_size: int, args: argparse.Namespace, resume_from_checkpoint: Optional[str]):
    """
    Runs the main distributed training loop for a worker.
    """
    print(f"Rank {rank}: Starting training... Config: epochs={args.epochs}, steps_per_epoch={args.steps}, chkpt_interval={args.checkpoint_interval}")

    dist.barrier()
    print(f"Rank {rank}: Barrier passed, setting up model and data.")

    # --- Model, Optimizer, Loss ---
    input_size = 10
    output_size = 1
    model = SimpleModel(input_size=input_size, output_size=output_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # --- Checkpoint Loading ---
    start_epoch = 0
    next_global_step = 0

    if resume_from_checkpoint and os.path.isfile(resume_from_checkpoint):
        print(f"Rank {rank}: Attempting to load checkpoint '{resume_from_checkpoint}'")
        try:
            checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            completed_global_step = checkpoint['step']
            next_global_step = completed_global_step + 1
            print(f"Rank {rank}: Loaded checkpoint. Resuming from Epoch {start_epoch + 1}, starting at Global Step {next_global_step + 1}")
        except Exception as e:
            print(f"Rank {rank}: WARNING - Error loading checkpoint: {e}. Starting from scratch.")
    else:
        if resume_from_checkpoint:
            print(f"Rank {rank}: Checkpoint file not found at '{resume_from_checkpoint}'. Starting from scratch.")
        else:
            print(f"Rank {rank}: No checkpoint provided. Starting from scratch.")

    dist.barrier()

    # --- Training Loop ---
    print(f"Rank {rank}: Starting training loop from Epoch {start_epoch + 1}, Global Step {next_global_step + 1}")
    current_global_step = next_global_step

    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\n--- Starting Epoch {epoch+1}/{args.epochs} ---")
        dist.barrier()

        epoch_start_time = time.time()
        epoch_loss = 0.0
        steps_run_in_epoch = 0

        epoch_start_step_idx = 0
        if current_global_step > 0 and (current_global_step // args.steps) == epoch:
            epoch_start_step_idx = current_global_step % args.steps

        if rank == 0 and epoch_start_step_idx > 0:
            print(f"Rank 0: Resuming Epoch {epoch+1} from step index {epoch_start_step_idx}")

        for step_idx in range(epoch_start_step_idx, args.steps):
            inputs = torch.randn(args.batch_size, input_size)
            targets = torch.randn(args.batch_size, output_size)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            average_gradients(model, world_size)
            optimizer.step()

            epoch_loss += loss.item()
            steps_run_in_epoch += 1

            if (current_global_step + 1) % args.checkpoint_interval == 0:
                dist.barrier()
                if rank == 0:
                    checkpoint_filename = os.path.join(args.checkpoint_dir, f"checkpoint_step_{current_global_step + 1}.pt")
                    state = {
                        'epoch': epoch,
                        'step': current_global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }
                    save_checkpoint(state, checkpoint_filename)
                dist.barrier()

            if rank == 0 and (step_idx + 1) % 10 == 0:
                print(f"Rank {rank} | Epoch {epoch+1} | Step {step_idx+1}/{args.steps} | Global Step: {current_global_step + 1} | Loss: {loss.item():.4f}")

            current_global_step += 1

        dist.barrier()
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / steps_run_in_epoch if steps_run_in_epoch > 0 else 0
        if rank == 0:
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Steps Run: {steps_run_in_epoch}, Time: {epoch_time:.2f}s")

    print(f"Rank {rank}: Training finished successfully after {current_global_step} global steps.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fault-Tolerant PyTorch Distributed Worker (CPU)')
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Shared directory for saving/loading checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Save checkpoint every N global steps')
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND, help='Distributed backend')

    args = parser.parse_args()

    # --- Get distributed config from environment variables ---
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ.get('MASTER_PORT', DEFAULT_MASTER_PORT)
    except KeyError as e:
        print(f"ERROR: Missing required environment variable: {e}")
        exit(1)

    print(f"Starting worker Rank: {rank}, World Size: {world_size}, Master: {master_addr}:{master_port}")

    # --- Ensure Checkpoint Directory Exists ---
    try:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    except OSError as e:
        print(f"Rank {rank}: Could not create checkpoint directory {args.checkpoint_dir}: {e}")

    # --- Find latest checkpoint ---
    # This is done by all workers, but only rank 0 will save checkpoints.
    # All workers need to agree on whether to resume or not.
    latest_checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)

    # --- Run Training ---
    try:
        setup(rank, world_size, master_addr, master_port, args.backend)
        run_training(rank, world_size, args, latest_checkpoint_path)
    except Exception as e:
        print(f"Rank {rank}: ERROR during training - {e}")
        traceback.print_exc()
    finally:
        print(f"Rank {rank}: Reached final barrier before cleanup.")
        dist.barrier()
        cleanup()
        print(f"Rank {rank}: Worker script execution finished.")
