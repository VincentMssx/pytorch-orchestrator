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
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SHUTDOWN_SIGNAL_RECEIVED = False

# --- Signal Handling ---
def signal_handler(signum, frame):
    global SHUTDOWN_SIGNAL_RECEIVED
    if not SHUTDOWN_SIGNAL_RECEIVED:
        logger.warning(f"Received signal {signum}. Initiating graceful shutdown.")
        SHUTDOWN_SIGNAL_RECEIVED = True

# --- Data Generation ---
def get_learnable_dataset(n_samples, input_size, device):
    logger.info(f"Generating a learnable dataset with {n_samples} samples.")
    true_weights = torch.randn(input_size, 1, device=device) * 2
    true_bias = torch.randn(1, device=device) * 5
    X = torch.randn(n_samples, input_size, device=device)
    noise = torch.randn(n_samples, 1, device=device) * 0.5
    y = X @ true_weights + true_bias + noise
    return TensorDataset(X, y)

class Trainer:
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
        self._setup_distributed()
        self._setup_device()
        self._setup_components()

    def _setup_distributed(self):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend=self.args.backend,
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )
        logger.info(f"Rank {self.rank}/{self.world_size}: Process group initialized.")

    def _setup_device(self):
        if self.args.backend == "nccl" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        logger.info(f"Rank {self.rank}: Using device: {self.device}")

    def _setup_components(self):
        model = SimpleModel(self.args.input_size, self.args.hidden_size, 1).to(self.device)
        self.model = DDP(model, device_ids=[self.device] if self.device.type == "cuda" else None)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr)
        dataset = get_learnable_dataset(self.args.dataset_size, self.args.input_size, self.device)
        self.sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        self.dataloader = DataLoader(dataset, batch_size=self.args.batch_size, sampler=self.sampler)

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path): return
        logger.info(f"Rank {self.rank}: Loading checkpoint...")
        map_location = self.device
        checkpoint = torch.load(self.checkpoint_path, map_location=map_location)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Rank {self.rank}: Resuming from Epoch {self.start_epoch}")

    def _save_checkpoint(self, current_epoch: int):
        if self.rank != 0: return
        logger.info(f"Rank 0: Saving checkpoint for epoch {current_epoch}")
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        state = {
            'epoch': current_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        temp_filename = self.checkpoint_path + ".tmp"
        torch.save(state, temp_filename)
        os.rename(temp_filename, self.checkpoint_path)

    def train(self):
        criterion = nn.MSELoss()
        self._load_checkpoint()
        dist.barrier()

        for epoch in range(self.start_epoch, self.args.epochs):
            if SHUTDOWN_SIGNAL_RECEIVED: break
            self.sampler.set_epoch(epoch)
            if self.rank == 0: logger.info(f"--- Starting Epoch {epoch+1}/{self.args.epochs} ---")

            for i, (inputs, targets) in enumerate(self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                if self.rank == 0 and i % self.args.log_interval == 0:
                    logger.info(f"Epoch: {epoch+1} | Batch: {i}/{len(self.dataloader)} | Loss: {loss.item():.4f}")

            dist.barrier()
            self._save_checkpoint(epoch)
            dist.barrier()

        if self.rank == 0:
            logger.info("Training finished.")

    def cleanup(self):
        if dist.is_initialized():
            dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Trainer')
    parser.add_argument('--input_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--dataset_size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'])
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    trainer = Trainer(args)
    try:
        trainer.setup()
        trainer.train()
    except Exception as e:
        logger.error(f"Rank {os.environ.get('RANK', 'N/A')}: Unhandled exception.", exc_info=e)
        sys.exit(1)
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.error("RANK and WORLD_SIZE env vars are required.")
        sys.exit(1)
    main()