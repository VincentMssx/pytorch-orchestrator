import argparse
import glob
import os
import re
import traceback
from typing import Dict, Any, Optional

from models.model import SimpleModel

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


CHECKPOINT_FILENAME_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")

# --- Checkpoint Discovery and Sharing (No changes needed, this part is excellent) ---
def find_and_share_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    rank = dist.get_rank()
    latest_checkpoint_path = None

    if rank == 0:
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

    path_list = [latest_checkpoint_path]
    dist.broadcast_object_list(path_list, src=0)
    latest_checkpoint_path = path_list[0]

    if rank != 0 and latest_checkpoint_path:
        print(f"Rank {rank}: Received checkpoint path '{latest_checkpoint_path}' from Rank 0.")

    return latest_checkpoint_path

# --- Checkpoint Saving (with Atomic Write) ---
def save_checkpoint(state: Dict[str, Any], filename: str):
    rank = dist.get_rank()
    print(f"Rank {rank}: Saving checkpoint to {filename}")
    try:
        temp_filename = filename + ".tmp"
        torch.save(state, temp_filename)
        os.rename(temp_filename, filename) # Atomic operation
        print(f"Rank {rank}: Checkpoint saved successfully.")
    except Exception as e:
        print(f"Rank {rank}: ERROR - Could not save checkpoint: {e}")
        # Clean up temp file on failure
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- The Main Training Function ---
def run_training(args: argparse.Namespace, resume_from_checkpoint: Optional[str]):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size}: Starting training session...")
    dist.barrier()

    device = torch.device("cpu") # For GPU: device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Using device: {device}")

    input_size = 10
    output_size = 1
    model = DDP(SimpleModel(input_size, output_size).to(device))
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    torch.manual_seed(42)
    train_dataset = TensorDataset(torch.randn(args.dataset_size, input_size), torch.randn(args.dataset_size, output_size))
    sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)

    start_global_step = 0
    if resume_from_checkpoint:
        print(f"Rank {rank}: Loading checkpoint '{resume_from_checkpoint}'")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Resume from the step *after* the last saved step.
        start_global_step = checkpoint['global_step'] + 1
        print(f"Rank {rank}: Resuming from Global Step {start_global_step}")

    dist.barrier()
    
    # --- MODIFICATION 2: Use a single loop that tracks global steps ---
    # This avoids re-doing work within an epoch upon restart.
    current_global_step = start_global_step
    
    # Run until the target number of global steps is reached
    while current_global_step < args.total_steps:
        # The sampler needs the epoch to be set for proper shuffling.
        # We can use the global step to derive a logical epoch for this purpose.
        epoch = current_global_step // len(dataloader)
        sampler.set_epoch(epoch)
        
        if rank == 0 and current_global_step % len(dataloader) == 0:
            print(f"\n--- Starting Logical Epoch {epoch+1} ---")
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Check if we have already reached the total number of steps
            if current_global_step >= args.total_steps:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Checkpoint logic based on global steps
            if (current_global_step + 1) % args.checkpoint_interval == 0:
                dist.barrier()
                if rank == 0:
                    chkpt_file = os.path.join(args.checkpoint_dir, f"checkpoint_step_{current_global_step + 1}.pt")
                    state = {
                        'global_step': current_global_step, # Save the completed step
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    save_checkpoint(state, chkpt_file)
                dist.barrier() # Wait for save to complete

            if rank == 0 and (current_global_step + 1) % 10 == 0:
                print(f"Rank {rank} | Global Step {current_global_step+1}/{args.total_steps} | Loss: {loss.item():.4f}")

            current_global_step += 1
        
        # This inner loop will break if we've hit the total_steps
        if current_global_step >= args.total_steps:
            break

    print(f"Rank {rank}: Training finished after {current_global_step} global steps.")

def main():
    parser = argparse.ArgumentParser(description='Elastic PyTorch Distributed Worker using torchrun')
    # Use total_steps instead of epochs/steps for clearer control
    parser.add_argument('--total_steps', type=int, default=1000, help='Total training steps to run.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--backend', type=str, default='gloo', help='Distributed backend (gloo for CPU, nccl for GPU)')
    args = parser.parse_args()

    # `torchrun` sets environment variables, `init_process_group` reads them.
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()

    try:
        # Rank 0 creates the directory, others wait.
        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        dist.barrier()

        latest_checkpoint_path = find_and_share_latest_checkpoint(args.checkpoint_dir)
        run_training(args, latest_checkpoint_path)

    except Exception as e:
        print(f"Rank {rank}: ERROR during training - {e}")
        traceback.print_exc()
    finally:
        dist.destroy_process_group()
        print(f"Rank {rank}: Worker script finished.")


if __name__ == "__main__":
    main()