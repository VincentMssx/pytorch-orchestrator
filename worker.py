import argparse
import datetime
import os
import time
import traceback
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

# Import the model definition
from models.SimpleModel import SimpleModel

# --- Constants ---
DEFAULT_MASTER_ADDR = 'localhost'
DEFAULT_MASTER_PORT = '29500'
DEFAULT_BACKEND = 'gloo'


# --- Distributed Setup ---
def setup(rank: int, world_size: int, master_addr: str, master_port: str, backend: str = DEFAULT_BACKEND):
    """
    Initializes the distributed process group.

    Args:
        rank: Unique identifier of the current process.
        world_size: Total number of processes participating.
        master_addr: IP address of the master node (rank 0).
        master_port: Port number on the master node for coordination.
        backend: The distributed backend to use (e.g., 'gloo', 'nccl').
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

    Args:
        model: The model whose gradients need averaging.
        world_size: The total number of workers.
    """
    if world_size <= 1: # No need to average if only one worker
        return

    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


# --- Checkpointing ---
def save_checkpoint(state: Dict[str, Any], filename: str = "checkpoint.pt"):
    """
    Saves checkpoint state dictionary to a file.
    This function should typically only be called by rank 0.

    Args:
        state: A dictionary containing the state to save (model, optimizer, epoch, step, etc.).
        filename: The path to save the checkpoint file.
    """
    rank = dist.get_rank()
    print(f"Rank {rank}: Attempting to save checkpoint to {filename}")

    # Ensure the directory exists
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
def run_training(rank: int, world_size: int, args: argparse.Namespace):
    """
    Runs the main distributed training loop for a worker.

    Args:
        rank: The rank of this worker process.
        world_size: The total number of workers.
        args: Parsed command-line arguments containing hyperparameters and settings.
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

    if args.resume_from_checkpoint and os.path.isfile(args.resume_from_checkpoint):
        print(f"Rank {rank}: Attempting to load checkpoint '{args.resume_from_checkpoint}'")
        try:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            completed_global_step = checkpoint['step']
            next_global_step = completed_global_step + 1

            print(f"Rank {rank}: Loaded checkpoint. Resuming from Epoch {start_epoch + 1}, starting at Global Step {next_global_step + 1}")

        except Exception as e:
            print(f"Rank {rank}: WARNING - Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            next_global_step = 0
    else:
        if args.resume_from_checkpoint: # Path given but file not found
            print(f"Rank {rank}: Checkpoint file not found at '{args.resume_from_checkpoint}'. Starting from scratch.")
        else: # No checkpoint path provided
            print(f"Rank {rank}: No checkpoint provided. Starting from scratch.")

    # Barrier: Ensure all workers load/fail consistently before starting training loop
    dist.barrier()

    # --- Training Loop ---
    print(f"Rank {rank}: Starting training loop from Epoch {start_epoch + 1}, Global Step {next_global_step + 1}")
    current_global_step = next_global_step

    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\n--- Starting Epoch {epoch+1}/{args.epochs} ---")
        dist.barrier() # Sync all processes before starting steps in an epoch

        epoch_start_time = time.time()
        epoch_loss = 0.0
        steps_run_in_epoch = 0

        # Determine the starting step index for this epoch based on global step
        epoch_start_step_idx = 0
        if current_global_step > 0 and (current_global_step // args.steps) == epoch:
             epoch_start_step_idx = current_global_step % args.steps

        if rank == 0 and epoch_start_step_idx > 0:
            print(f"Rank 0: Resuming Epoch {epoch+1} from step index {epoch_start_step_idx}")

        # Loop through steps for the current epoch
        for step_idx in range(epoch_start_step_idx, args.steps):
            # Generate synthetic data for this step
            inputs = torch.randn(args.batch_size, input_size)
            targets = torch.randn(args.batch_size, output_size)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Average gradients across all workers
            average_gradients(model, world_size)

            # Optimizer step
            optimizer.step()

            epoch_loss += loss.item()
            steps_run_in_epoch += 1

            # --- Checkpoint Saving Logic ---
            if (current_global_step + 1) % args.checkpoint_interval == 0:
                # Ensure all workers reach this point before rank 0 saves
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
                # Ensure rank 0 finishes saving before others proceed
                dist.barrier()

            # Log progress periodically (only from rank 0)
            if rank == 0 and (step_idx + 1) % 10 == 0: # Log based on step within epoch
                print(f"Rank {rank} | Epoch {epoch+1} | Step {step_idx+1}/{args.steps} | Global Step: {current_global_step + 1} | Loss: {loss.item():.4f}")

            current_global_step += 1

        # --- End of Epoch ---
        dist.barrier() # Sync after each epoch completes
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / steps_run_in_epoch if steps_run_in_epoch > 0 else 0
        if rank == 0:
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Steps Run: {steps_run_in_epoch}, Time: {epoch_time:.2f}s")

    print(f"Rank {rank}: Training finished successfully after {current_global_step} global steps.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fault-Tolerant PyTorch Distributed Worker (CPU)')
    # Distributed arguments
    parser.add_argument('--rank', type=int, required=True, help='Rank of the process')
    parser.add_argument('--world_size', type=int, required=True, help='Number of processes participating')
    parser.add_argument('--master_addr', type=str, default=DEFAULT_MASTER_ADDR, help='Master node address')
    parser.add_argument('--master_port', type=str, default=DEFAULT_MASTER_PORT, help='Master node port')
    parser.add_argument('--backend', type=str, default=DEFAULT_BACKEND, help='Distributed backend')
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Save checkpoint every N global steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume from')

    args = parser.parse_args()

    worker_rank = args.rank
    print(f"Starting worker Rank: {worker_rank}, World Size: {args.world_size}, Port: {args.master_port}")

    # Basic validation
    if worker_rank >= args.world_size or worker_rank < 0:
         print(f"ERROR: Invalid rank {worker_rank} for world size {args.world_size}")
         exit(1)

    try:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    except OSError as e:
        print(f"Rank {worker_rank}: Could not create checkpoint directory {args.checkpoint_dir}: {e}")

    # Initialize distributed environment
    setup(worker_rank, args.world_size, args.master_addr, args.master_port, args.backend)

    # Run the main training loop within a try/finally block for cleanup
    try:
        run_training(worker_rank, args.world_size, args)
    except Exception as e:
        print(f"Rank {worker_rank}: ERROR during training - {e}")
        traceback.print_exc()
    finally:
        print(f"Rank {worker_rank}: Reached final barrier before cleanup.")
        dist.barrier()
        cleanup()
        print(f"Rank {worker_rank}: Worker script execution finished.")