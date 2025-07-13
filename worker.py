import argparse
import glob
import os
import re
import traceback
from typing import Optional

from models.SimpleModel import SimpleModel

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

CHECKPOINT_FILENAME_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")

class Trainer:
    """
    Encapsulates the entire distributed training process, including state management,
    checkpointing, and the training loop itself.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = -1
        self.world_size = -1
        self.device = None
        self.model = None
        self.optimizer = None
        self.dataloader = None
        self.sampler = None
        self.start_global_step = 0

    def _setup_distributed(self):
        """Initializes the distributed process group."""
        dist.init_process_group(backend=self.args.backend)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        print(f"Rank {self.rank}/{self.world_size}: Distributed process group initialized.")

    def _setup_components(self):
        """Sets up the model, optimizer, and data loaders."""
        self.device = torch.device("cpu")  # For GPU: torch.device(f"cuda:{self.rank}")
        print(f"Rank {self.rank}: Using device: {self.device}")

        # Model and optimizer
        input_size = 10
        output_size = 1
        model = SimpleModel(input_size, output_size).to(self.device)
        self.model = DDP(model, device_ids=None if self.device.type == 'cpu' else [self.rank])
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        # Dataset and DataLoader
        torch.manual_seed(42)
        train_dataset = TensorDataset(torch.randn(self.args.dataset_size, input_size), torch.randn(self.args.dataset_size, output_size))
        self.sampler = DistributedSampler(train_dataset)
        self.dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=self.sampler)

    def _find_and_share_latest_checkpoint(self) -> Optional[str]:
        """
        Identifies the latest checkpoint on rank 0 and broadcasts the path.
        """
        latest_checkpoint_path = None
        if self.rank == 0:
            if os.path.isdir(self.args.checkpoint_dir):
                try:
                    checkpoint_files = glob.glob(os.path.join(self.args.checkpoint_dir, "checkpoint_step_*.pt"))
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
                    print(f"Error accessing checkpoint directory '{self.args.checkpoint_dir}': {e}")

        path_list = [latest_checkpoint_path]
        dist.broadcast_object_list(path_list, src=0)
        latest_checkpoint_path = path_list[0]

        if self.rank != 0 and latest_checkpoint_path:
            print(f"Rank {self.rank}: Received checkpoint path '{latest_checkpoint_path}' from Rank 0.")

        return latest_checkpoint_path

    def _load_checkpoint(self, checkpoint_path: str):
        """Loads training state from a checkpoint file."""
        print(f"Rank {self.rank}: Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_global_step = checkpoint['global_step'] + 1
        print(f"Rank {self.rank}: Resuming from Global Step {self.start_global_step}")

    def _save_checkpoint(self, current_global_step: int):
        """Saves the current training state to a checkpoint file."""
        if self.rank == 0:
            chkpt_file = os.path.join(self.args.checkpoint_dir, f"checkpoint_step_{current_global_step + 1}.pt")
            print(f"Rank 0: Saving checkpoint to {chkpt_file}")
            state = {
                'global_step': current_global_step,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            try:
                temp_filename = chkpt_file + ".tmp"
                torch.save(state, temp_filename)
                os.rename(temp_filename, chkpt_file)  # Atomic write
                print(f"Rank 0: Checkpoint saved successfully.")
            except Exception as e:
                print(f"Rank 0: ERROR - Could not save checkpoint: {e}")
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

    def train(self):
        """The main training loop."""
        criterion = nn.MSELoss()
        current_global_step = self.start_global_step

        while current_global_step < self.args.total_steps:
            epoch = current_global_step // len(self.dataloader)
            self.sampler.set_epoch(epoch)

            if self.rank == 0 and current_global_step % len(self.dataloader) == 0:
                print(f"\n--- Starting Logical Epoch {epoch+1} ---")

            for inputs, targets in self.dataloader:
                if current_global_step >= self.args.total_steps:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if (current_global_step + 1) % self.args.checkpoint_interval == 0:
                    dist.barrier() # Ensure all ranks are at the same step
                    self._save_checkpoint(current_global_step)
                    dist.barrier() # Wait for rank 0 to finish saving

                if self.rank == 0 and (current_global_step + 1) % 10 == 0:
                    print(f"Rank {self.rank} | Global Step {current_global_step+1}/{self.args.total_steps} | Loss: {loss.item():.4f}")

                current_global_step += 1

            if current_global_step >= self.args.total_steps:
                break

        print(f"Rank {self.rank}: Training finished after {current_global_step} global steps.")

    def run(self):
        """Main execution method to run the entire training pipeline."""
        try:
            self._setup_distributed()
            self._setup_components()

            if self.rank == 0:
                os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            dist.barrier()

            latest_checkpoint_path = self._find_and_share_latest_checkpoint()
            if latest_checkpoint_path:
                self._load_checkpoint(latest_checkpoint_path)

            dist.barrier()
            self.train()

        except Exception as e:
            print(f"Rank {self.rank}: ERROR during training - {e}")
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleans up the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {self.rank}: Worker script finished.")

def main():
    """
    Parses command-line arguments and launches the training process.
    """
    parser = argparse.ArgumentParser(description='Elastic PyTorch Distributed Worker using torchrun')
    parser.add_argument('--total_steps', type=int, default=1000, help='Total training steps to run.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--backend', type=str, default='gloo', help='Distributed backend (gloo for CPU, nccl for GPU)')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()

if __name__ == "__main__":
    main()