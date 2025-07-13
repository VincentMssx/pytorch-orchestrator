import argparse
import glob
import os
import re
import traceback
from models.SimpleModel import SimpleModel
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

CHECKPOINT_FILENAME_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")

# --- CHANGE 1: Let Rank 0 find the checkpoint and tell everyone else ---
# This avoids a "thundering herd" where every process scans the filesystem.
def find_and_share_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    rank = dist.get_rank()
    latest_checkpoint_path = None

    if rank == 0:
        # Only rank 0 scans the directory
        if not os.path.isdir(checkpoint_dir):
            print(f"Checkpoint directory '{checkpoint_dir}' not found.")
        else:
            try:
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
                latest_step = -1
                for ckpt_file_path in checkpoint_files:
                    match = CHECKPOINT_FILENAME_PATTERN.search(os.path.basename(ckpt_file_path))
                    if match:
                        step = int(match.group(1))
                        if step > latest_step:
                            latest_step = step
                            latest_checkpoint_path = ckpt_file_path
                if latest_checkpoint_path:
                    print(f"Rank 0: Identified latest checkpoint '{latest_checkpoint_path}'.")
            except OSError as e:
                print(f"Error accessing checkpoint directory '{checkpoint_dir}': {e}")

    # Broadcast the path from rank 0 to all other processes.
    # We use a list to hold the string path.
    path_list = [latest_checkpoint_path]
    dist.broadcast_object_list(path_list, src=0)
    latest_checkpoint_path = path_list[0]

    if rank != 0 and latest_checkpoint_path:
        print(f"Rank {rank}: Received checkpoint path '{latest_checkpoint_path}' from Rank 0.")

    return latest_checkpoint_path


def save_checkpoint(state: Dict[str, Any], filename: str):
    rank = dist.get_rank()
    # No changes needed here, this is already correct (only rank 0 saves).
    print(f"Rank {rank}: Saving checkpoint to {filename}")
    try:
        torch.save(state, filename)
        print(f"Rank {rank}: Checkpoint saved successfully.")
    except Exception as e:
        print(f"Rank {rank}: ERROR - Could not save checkpoint: {e}")


def run_training(args: argparse.Namespace, resume_from_checkpoint: Optional[str]):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size}: Starting training session...")
    dist.barrier()

    # --- CHANGE 2: Device placement (important for GPU, good practice for CPU) ---
    # In a real scenario with GPUs, you'd use rank to select the device.
    # For CPU, this doesn't change much but is the correct pattern.
    device = torch.device("cpu")
    print(f"Rank {rank}: Using device: {device}")

    input_size = 10
    output_size = 1
    # DDP wraps the model and handles data parallelism
    model = DDP(SimpleModel(input_size, output_size).to(device))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    next_global_step = 0

    if resume_from_checkpoint:
        # All ranks load the checkpoint. DDP ensures model states are synced.
        print(f"Rank {rank}: Loading checkpoint '{resume_from_checkpoint}'")
        # Load onto the correct device
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        next_global_step = checkpoint['step'] + 1
        print(f"Rank {rank}: Resuming from Epoch {start_epoch + 1}, Global Step {next_global_step}")

    dist.barrier()
    current_global_step = next_global_step

    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\n--- Starting Epoch {epoch+1}/{args.epochs} ---")
        dist.barrier()

        for step_idx in range(args.steps):
            # Create data on the correct device
            inputs = torch.randn(args.batch_size, input_size).to(device)
            targets = torch.randn(args.batch_size, output_size).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if (current_global_step + 1) % args.checkpoint_interval == 0:
                dist.barrier()
                # Only rank 0 is responsible for saving the checkpoint
                if rank == 0:
                    chkpt_file = os.path.join(args.checkpoint_dir, f"checkpoint_step_{current_global_step + 1}.pt")
                    state = {
                        'epoch': epoch,
                        'step': current_global_step,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    save_checkpoint(state, chkpt_file)
                # All ranks wait until the checkpoint is saved before continuing.
                dist.barrier()

            if rank == 0 and (step_idx + 1) % 10 == 0:
                print(f"Rank {rank} | Epoch {epoch+1} | Step {step_idx+1}/{args.steps} | Loss: {loss.item():.4f}")

            current_global_step += 1

    print(f"Rank {rank}: Training finished after {current_global_step} global steps.")


def main():
    parser = argparse.ArgumentParser(description='Elastic PyTorch Distributed Worker using torchrun')
    # ... your arguments are fine ...
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--backend', type=str, default='gloo') # Gloo is good for CPU
    args = parser.parse_args()

    # --- CHANGE 3: The most important change! ---
    # `torchrun` will automatically set environment variables like RANK, WORLD_SIZE,
    # MASTER_ADDR, and MASTER_PORT. `init_process_group` will read them automatically.
    # Your script no longer needs to know what they are in advance.
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()

    try:
        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        dist.barrier() # Ensure directory is made before workers proceed

        # Use the new broadcast function
        latest_checkpoint_path = find_and_share_latest_checkpoint(args.checkpoint_dir)

        run_training(args, latest_checkpoint_path)

    except Exception as e:
        print(f"Rank {rank}: ERROR during training - {e}")
        traceback.print_exc()
    finally:
        # Clean up the process group
        dist.destroy_process_group()
        print(f"Rank {rank}: Worker script finished.")


if __name__ == "__main__":
    main()