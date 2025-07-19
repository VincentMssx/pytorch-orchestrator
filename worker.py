import argparse
import glob
import os
import re
import time
import traceback
from typing import Optional

from models.SimpleModel import SimpleModel

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

CHECKPOINT_FILENAME_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")

class Trainer:
    """
    Encapsulates a single run of the distributed training process.
    Designed to be restarted by an external agent like torchrun.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = -1
        self.local_rank = -1
        self.world_size = -1
        self.device = None
        self.model = None
        self.optimizer = None
        self.dataloader = None
        self.sampler = None
        self.start_global_step = 0
        self._last_logical_epoch_printed = -1

    def _setup_distributed(self):
        """Initializes the distributed process group and sets device."""
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend=self.args.backend)
        print(f"Rank {self.rank}/{self.world_size} (local_rank: {self.local_rank}): Distributed process group initialized.")
        
        # Determine device based on backend
        if self.args.backend == 'nccl':
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

    def _setup_components(self):
        """Sets up the model, optimizer, and data loaders."""
        if self.rank == 0:
            print(f"Rank {self.rank}: Setting up components on device: {self.device}")

        input_size = 1000
        output_size = 1
        model = SimpleModel(input_size, output_size).to(self.device)
        
        # DDP wrapper
        device_ids = [self.local_rank] if self.args.backend == 'nccl' else None
        self.model = DDP(model, device_ids=device_ids)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        torch.manual_seed(42) # Ensure all ranks start with the same data
        train_dataset = TensorDataset(torch.randn(self.args.dataset_size, input_size), torch.randn(self.args.dataset_size, output_size))
        self.sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        self.dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, sampler=self.sampler)

    def _find_and_share_latest_checkpoint(self) -> Optional[str]:
        # This function is well-written and correct for this purpose.
        latest_checkpoint_path = None
        if self.rank == 0:
            os.makedirs(self.args.checkpoint_dir, exist_ok=True)
            if os.path.isdir(self.args.checkpoint_dir):
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
        
        # Broadcast the path from rank 0 to all other ranks
        path_list = [latest_checkpoint_path]
        dist.broadcast_object_list(path_list, src=0)
        latest_checkpoint_path = path_list[0]

        if self.rank != 0 and latest_checkpoint_path:
            print(f"Rank {self.rank}: Received checkpoint path '{latest_checkpoint_path}' from Rank 0.")
        return latest_checkpoint_path

    def _load_checkpoint(self, checkpoint_path: str):
        # This function is also correct.
        print(f"Rank {self.rank}: Loading checkpoint '{checkpoint_path}'")
        map_location = self.device
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_global_step = checkpoint['global_step'] + 1
        print(f"Rank {self.rank}: Resuming from Global Step {self.start_global_step}")

    def _save_checkpoint(self, current_global_step: int):
        # This function is also correct, using atomic writes is great.
        if self.rank == 0:
            chkpt_file = os.path.join(self.args.checkpoint_dir, f"checkpoint_step_{current_global_step + 1}.pt")
            print(f"Rank 0: Saving checkpoint to {chkpt_file}")
            state = {
                'global_step': current_global_step,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            temp_filename = chkpt_file + ".tmp"
            torch.save(state, temp_filename)
            os.rename(temp_filename, chkpt_file)
            print(f"Rank 0: Checkpoint saved successfully.")

    def train(self):
        """The main training loop."""
        criterion = nn.MSELoss()
        current_global_step = self.start_global_step

        # If we loaded a checkpoint, we might be done already.
        if current_global_step >= self.args.total_steps:
            print(f"Rank {self.rank}: Training already completed. Exiting.")
            return

        while current_global_step < self.args.total_steps:
            # The epoch is relative to the start of this training run
            # but set_epoch needs to be consistent across restarts
            logical_epoch = current_global_step // len(self.dataloader)
            self.sampler.set_epoch(logical_epoch)
            
            if self.rank == 0 and logical_epoch > self._last_logical_epoch_printed:
                total_epochs = (self.args.total_steps + len(self.dataloader) - 1) // len(self.dataloader)
                print(f"\n--- Starting Logical Epoch {logical_epoch+1} / {total_epochs} (World Size: {self.world_size}) ---")
                self._last_logical_epoch_printed = logical_epoch

            for inputs, targets in self.dataloader:
                if current_global_step >= self.args.total_steps:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                current_global_step += 1

                if current_global_step % self.args.checkpoint_interval == 0:
                    dist.barrier()
                    self._save_checkpoint(current_global_step -1)
                    dist.barrier()

                if self.rank == 0 and current_global_step % 100 == 0:
                    print(f"Rank {self.rank} | Global Step {current_global_step}/{self.args.total_steps} | Loss: {loss.item():.4f}")

        if self.rank == 0:
            print(f"Rank {self.rank}: Training finished after {current_global_step} global steps.")

    def run(self):
        """Main execution method. No longer has a restart loop."""
        start_time = time.time()
        self._setup_distributed()
        self._setup_components()

        dist.barrier() # Ensure all ranks have components ready
        latest_checkpoint_path = self._find_and_share_latest_checkpoint()
        if latest_checkpoint_path:
            self._load_checkpoint(latest_checkpoint_path)
        dist.barrier() # Ensure all ranks have loaded checkpoint before starting

        self.train()

        if self.rank == 0:
            end_time = time.time()
            print(f"Total training time: {end_time - start_time:.2f} seconds")

    def cleanup(self):
        """Cleans up the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
            if self.rank == 0:
                print(f"Rank {self.rank}: Worker cleanup complete.")

def main():
    parser = argparse.ArgumentParser(description='Elastic PyTorch Distributed Worker using torchrun')
    parser.add_argument('--total_steps', type=int, default=10000, help='Total training steps to run.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset_size', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--backend', type=str, default='gloo', help='Distributed backend (gloo for CPU, nccl for GPU)')
    args = parser.parse_args()

    trainer = Trainer(args)
    try:
        print(f"Starting worker with args: {args}")
        trainer.run()
    except Exception as e:
        print(f"Rank {os.environ.get('RANK', 'N/A')}: Unhandled exception during training run: {str(e)}")
        traceback.print_exc()
        raise  # Re-raise to ensure Docker logs the error and exits non-zero
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()