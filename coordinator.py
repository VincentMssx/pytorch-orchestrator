import argparse
import subprocess
import os
import sys
import glob
import re

def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on step number."""
    latest_checkpoint = None
    latest_step = -1

    if not os.path.isdir(checkpoint_dir):
        print(f"Coordinator: Checkpoint directory '{checkpoint_dir}' not found.")
        return None

    pattern = re.compile(r"checkpoint_step_(\d+)\.pt")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))

    if not checkpoint_files:
        print(f"Coordinator: No checkpoint files found in '{checkpoint_dir}'.")
        return None

    for ckpt_file in checkpoint_files:
        match = pattern.search(os.path.basename(ckpt_file))
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = ckpt_file

    if latest_checkpoint:
        print(f"Coordinator: Found latest checkpoint '{latest_checkpoint}' at step {latest_step}.")
    else:
         print(f"Coordinator: Found checkpoint files, but couldn't parse step number from names.")

    return latest_checkpoint


def run_worker(rank, world_size, args, resume_checkpoint_path):
    """Function to be executed by each worker process."""
    command = [
        sys.executable,
        "worker.py",
        "--rank", str(rank),
        "--world_size", str(world_size),
        "--master_addr", args.master_addr,
        "--master_port", args.master_port,
        "--epochs", str(args.epochs),
        "--steps", str(args.steps),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--checkpoint_dir", args.checkpoint_dir,
        "--checkpoint_interval", str(args.checkpoint_interval),
    ]
    if resume_checkpoint_path:
        command.extend(["--resume_from_checkpoint", resume_checkpoint_path])

    print(f"Coordinator: Launching Rank {rank} with command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command)
        return process
    except Exception as e:
        print(f"Coordinator: Failed to launch Rank {rank}: {e}")
        return None


def main(args):
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)

    world_size = args.num_workers
    processes = []
    print(f"\nCoordinator: Launching {world_size} workers...")
    for rank in range(world_size):
        process = run_worker(rank, world_size, args, latest_checkpoint)
        if process:
            processes.append(process)
        else:
            print(f"Coordinator: Critical error launching Rank {rank}. Aborting.")
            for p in processes:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
            return

    print("\nCoordinator: All workers launched. Waiting for completion...")
    for rank, process in enumerate(processes):
        process.wait()
        if process.returncode != 0:
             print(f"Coordinator: Worker Rank {rank} exited with error code {process.returncode}.")
        else:
             print(f"Coordinator: Worker Rank {rank} finished successfully.")

    print("\nCoordinator: All workers have finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training Coordinator')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes (world size)')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Save checkpoint every N steps (passed to worker)')

    args = parser.parse_args()

    main(args)
