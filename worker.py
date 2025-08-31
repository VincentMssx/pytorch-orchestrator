
import os
import sys
import time
import logging
import signal
import argparse
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from models.SimpleModel import SimpleModel

# --- Global Variables ---
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Flag to indicate if a shutdown signal has been received
SHUTDOWN_SIGNAL_RECEIVED = False


# --- Signal Handling ---
def signal_handler(signum, frame):
    """Sets a flag to indicate a shutdown signal was received."""
    global SHUTDOWN_SIGNAL_RECEIVED
    if not SHUTDOWN_SIGNAL_RECEIVED:
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown.")
        SHUTDOWN_SIGNAL_RECEIVED = True


class Trainer:
    """
    Encapsulates the distributed training process, including setup, training loop,
    and checkpointing.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = -1
        self.world_size = -1
        self.device = None
        self.model: Optional[DDP] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.dataloader: Optional[DataLoader] = None
        self.sampler: Optional[DistributedSampler] = None
        self.start_epoch = 0
        self.checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")

    def setup(self):
        """Initializes the distributed process group, device, model, and data."""
        self._setup_distributed()
        self._setup_device()
        self._setup_components()

    def _setup_distributed(self):
        """Initializes the distributed process group."""
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        # The master address and port are expected to be in the environment
        dist.init_process_group(
            backend=self.args.backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        logger.info(f"Rank {self.rank}/{self.world_size}: Distributed process group initialized.")

    def _setup_device(self):
        """Sets the device for training (CPU or GPU)."""
        if self.args.backend == "nccl" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self.device)
            logger.info(f"Rank {self.rank}: Using device: {self.device}")
        else:
            self.device = torch.device("cpu")
            logger.info(f"Rank {self.rank}: Using device: CPU")

    def _setup_components(self):
        """Sets up the model, optimizer, and data loaders."""
        model = SimpleModel(self.args.input_size, self.args.hidden_size, 1).to(self.device)
        self.model = DDP(model, device_ids=[self.device] if self.device.type == "cuda" else None)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)

        # Create a dummy dataset for demonstration
        torch.manual_seed(42)  # Ensure all ranks start with the same data
        dataset = TensorDataset(
            torch.randn(self.args.dataset_size, self.args.input_size),
            torch.randn(self.args.dataset_size, 1),
        )
        self.sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=self.sampler)

    def _load_checkpoint(self):
        """Loads a checkpoint from disk if it exists."""
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"Rank {self.rank}: No checkpoint found at {self.checkpoint_path}. Starting from scratch.")
            return

        logger.info(f"Rank {self.rank}: Loading checkpoint from {self.checkpoint_path}")
        map_location = self.device
        checkpoint = torch.load(self.checkpoint_path, map_location=map_location)

        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Rank {self.rank}: Resuming from Epoch {self.start_epoch}")

    def _save_checkpoint(self, current_epoch: int):
        """Saves a checkpoint to disk. Only rank 0 saves."""
        if self.rank != 0:
            return

        logger.info(f"Rank 0: Saving checkpoint for epoch {current_epoch} to {self.checkpoint_path}")
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': current_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        # Atomic save
        temp_filename = self.checkpoint_path + ".tmp"
        torch.save(state, temp_filename)
        os.rename(temp_filename, self.checkpoint_path)
        logger.info("Rank 0: Checkpoint saved successfully.")

    def train(self):
        """The main training loop."""
        criterion = nn.MSELoss()
        self._load_checkpoint()
        dist.barrier() # Ensure all ranks load checkpoint before starting

        for epoch in range(self.start_epoch, self.args.epochs):
            if SHUTDOWN_SIGNAL_RECEIVED:
                break

            self.sampler.set_epoch(epoch)
            if self.rank == 0:
                logger.info(f"--- Starting Epoch {epoch+1}/{self.args.epochs} ---")

            for i, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if self.rank == 0 and i % self.args.log_interval == 0:
                    logger.info(f"Epoch: {epoch+1} | Batch: {i}/{len(self.dataloader)} | Loss: {loss.item():.4f}")

            # Save checkpoint at the end of each epoch
            dist.barrier()
            self._save_checkpoint(epoch)
            dist.barrier()

        if self.rank == 0:
            if SHUTDOWN_SIGNAL_RECEIVED:
                logger.warning("Training interrupted by shutdown signal.")
            else:
                logger.info("Training finished successfully.")

    def cleanup(self):
        """Cleans up the distributed process group."""
        if dist.is_initialized():
            logger.info(f"Rank {self.rank}: Cleaning up distributed process group.")
            dist.destroy_process_group()


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(description='Production-Grade PyTorch Distributed Trainer')
    # Model/Data Args
    parser.add_argument('--input_size', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=2000)
    parser.add_argument('--dataset_size', type=int, default=4000)
    # Training Args
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--log_interval', type=int, default=10)
    # Infrastructure Args
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], help='Distributed backend.')
    # The --is_master flag is no longer needed as logic is determined by RANK
    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    trainer = Trainer(args)
    try:
        trainer.setup()
        trainer.train()
    except Exception as e:
        logger.error(f"Rank {os.environ.get('RANK', 'N/A')}: Unhandled exception during training.", exc_info=e)
        # Exit with a non-zero code to indicate failure
        sys.exit(1)
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    # MASTER_ADDR, MASTER_PORT, RANK, and WORLD_SIZE are expected to be set in the environment
    # by the Kubernetes Downward API and the service definition.
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.error("RANK and WORLD_SIZE environment variables are required.")
        sys.exit(1)
        
    main()
