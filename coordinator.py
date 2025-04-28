import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import time
import types
from typing import Dict, Optional, List

# --- Constants ---
CHECKPOINT_FILENAME_PATTERN = re.compile(r"checkpoint_step_(\d+)\.pt")
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_MAX_RESTARTS = 3
DEFAULT_RESTART_WAIT_TIME = 5.0
DEFAULT_MASTER_ADDR = 'localhost'
DEFAULT_MASTER_PORT = '29500'

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
        print(f"Coordinator: Checkpoint directory '{checkpoint_dir}' not found or is not a directory.")
        return None

    try:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    except OSError as e:
        print(f"Coordinator: Error accessing checkpoint directory '{checkpoint_dir}': {e}")
        return None

    if not checkpoint_files:
        print(f"Coordinator: No files matching 'checkpoint_step_*.pt' found in '{checkpoint_dir}'.")
        return None

    print(f"Coordinator: Found {len(checkpoint_files)} potential checkpoint files. Parsing...")
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
                print(f"Coordinator: Warning - could not parse step number as integer from '{filename}'. Skipping.")
                continue

    if latest_checkpoint_path:
        print(f"Coordinator: Identified latest checkpoint '{latest_checkpoint_path}' at step {latest_step}.")
    else:
        print(f"Coordinator: Found potential checkpoint files, but couldn't identify the latest valid one based on step number pattern.")

    return latest_checkpoint_path


# --- Worker Process Management ---
def launch_single_worker(rank: int, world_size: int, args: argparse.Namespace, resume_checkpoint_path: Optional[str]) -> Optional[subprocess.Popen]:
    """
    Launches a single worker.py script as a subprocess.

    Args:
        rank: The rank to assign to this worker.
        world_size: The total number of workers in the distributed job.
        args: The coordinator's parsed arguments, containing settings to pass to the worker.
        resume_checkpoint_path: Path to the checkpoint file if resuming, otherwise None.

    Returns:
        A subprocess.Popen object representing the running worker process, or None if launch fails.
    """
    # Construct the command line arguments for the worker script
    command: List[str] = [
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

    # Add the resume checkpoint argument only if a path is provided
    if resume_checkpoint_path:
        command.extend(["--resume_from_checkpoint", resume_checkpoint_path])

    # Log the command being executed
    print(f"Coordinator: Launching Rank {rank}...")

    try:
        process = subprocess.Popen(command)
        print(f"Coordinator: Rank {rank} launched successfully with PID {process.pid}.")
        return process
    except FileNotFoundError:
        print(f"Coordinator: ERROR - 'worker.py' not found. Make sure it's in the same directory.")
        return None
    except Exception as e:
        print(f"Coordinator: ERROR - Failed to launch Rank {rank}: {e}")
        return None


def cleanup_processes(processes_dict: Dict[int, subprocess.Popen], reason: str = "cleanup"):
    """
    Attempts to gracefully terminate, and then forcefully kill, all processes
    in the provided dictionary.

    Args:
        processes_dict: A dictionary mapping rank (int) to Popen object.
                        The dictionary is modified in place.
        reason: A string indicating why cleanup is being performed.
    """
    print(f"Coordinator: Initiating {reason} for {len(processes_dict)} processes...")
    for rank in list(processes_dict.keys()):
        process = processes_dict[rank]
        if process.poll() is None:
            print(f"Coordinator: Terminating Rank {rank} (PID {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"Coordinator: Rank {rank} terminated gracefully.")
            except subprocess.TimeoutExpired:
                print(f"Coordinator: Rank {rank} did not terminate gracefully after 5s. Killing...")
                process.kill()
                try:
                    process.wait(timeout=2)
                    print(f"Coordinator: Rank {rank} killed.")
                except Exception as e:
                    print(f"Coordinator: Error waiting for killed Rank {rank} (PID {process.pid}): {e}")
            except Exception as e:
                print(f"Coordinator: Error waiting for terminated Rank {rank} (PID {process.pid}): {e}")
        else:
            print(f"Coordinator: Rank {rank} (PID {process.pid}) already exited with code {process.poll()} before {reason}.")

        del processes_dict[rank]

    print(f"Coordinator: {reason} complete.")


# --- Main Orchestration Logic ---
g_active_processes: Dict[int, subprocess.Popen] = {}

def signal_handler(sig: int, frame: Optional[types.FrameType]):
    """Gracefully handle termination signals (like Ctrl+C)."""
    signal_name = signal.Signals(sig).name
    print(f"\nCoordinator: Received signal {signal_name} ({sig}). Initiating shutdown...")
    cleanup_processes(g_active_processes, reason=f"{signal_name} signal")
    print("Coordinator: Shutdown complete.")
    sys.exit(1)


def main(args: argparse.Namespace):
    """
    Main coordinator function: launches, monitors, and restarts workers.
    """
    world_size = args.num_workers
    restart_count = 0
    max_restarts = args.max_restarts

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler) # Handle termination signals

    # Outer loop for handling restarts
    while restart_count <= max_restarts:
        print(f"\n===== Coordinator: Starting Training Attempt {restart_count + 1} / {max_restarts + 1} =====")

        # Find the latest checkpoint to resume from (if any)
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print(f"Coordinator: Will attempt to resume training from {latest_checkpoint}")
        else:
            print("Coordinator: No valid checkpoint found. Starting training from scratch.")

        # Launch worker processes for this attempt
        # Clear the global tracker before launching new processes
        g_active_processes.clear()
        processes_launched_successfully = True
        for rank in range(world_size):
            process = launch_single_worker(rank, world_size, args, latest_checkpoint)
            if process:
                g_active_processes[rank] = process
            else:
                print(f"Coordinator: CRITICAL ERROR - Failed to launch Rank {rank} on attempt {restart_count + 1}.")
                processes_launched_successfully = False
                break

        # If launch failed, clean up any workers that start and retry/abort
        if not processes_launched_successfully:
            cleanup_processes(g_active_processes, reason="launch failure cleanup")
            restart_count += 1
            if restart_count <= max_restarts:
                print(f"Coordinator: Launch failed. Waiting {args.restart_wait_time} seconds before next attempt...")
                time.sleep(args.restart_wait_time)
                continue
            else:
                print(f"Coordinator: Launch failed and maximum restart limit ({max_restarts}) reached. Aborting.")
                break

        # --- Monitoring Loop for the current attempt ---
        print(f"\nCoordinator: All {world_size} workers launched successfully for attempt {restart_count + 1}. Monitoring status...")
        failure_detected = False
        successful_ranks = set()

        # Continue monitoring as long as there are active processes for this attempt
        while g_active_processes:
            failed_rank = -1

            for rank, process in list(g_active_processes.items()):
                return_code = process.poll()

                if return_code is None:
                    continue
                elif return_code == 0:
                    print(f"Coordinator: Worker Rank {rank} (PID {process.pid}) finished successfully.")
                    successful_ranks.add(rank)
                    del g_active_processes[rank]
                else:
                    print(f"Coordinator: FAILURE DETECTED! Worker Rank {rank} (PID {process.pid}) exited unexpectedly with code {return_code}.")

                    failure_detected = True
                    failed_rank = rank
                    if rank in g_active_processes: 
                        del g_active_processes[rank]
                    break

            # If a failure was detected in the inner loop, break the monitoring loop to handle restart
            if failure_detected:
                break

            # If no failure detected in this poll cycle, check if all workers finished successfully
            if not g_active_processes and len(successful_ranks) == world_size:
                print("\nCoordinator: All workers completed successfully in this attempt!")
                cleanup_processes(g_active_processes, reason="Final cleanup")
                sys.exit(0)

            time.sleep(args.poll_interval)

        # --- Handle Failure or Completion of the Attempt ---
        if failure_detected:
            print(f"Coordinator: Failure in Rank {failed_rank} detected. Initiating cleanup and potential restart.")
            cleanup_processes(g_active_processes, reason=f"failure of Rank {failed_rank}")
            restart_count += 1

            # Check if we have exceeded the restart limit
            if restart_count <= max_restarts:
                print(f"Coordinator: Restart attempt {restart_count}/{max_restarts} will begin after {args.restart_wait_time} seconds...")
                time.sleep(args.restart_wait_time)
            else:
                print(f"Coordinator: Maximum restart limit ({max_restarts}) reached after failure. Aborting.")
                break # Exit the outer 'while' loop

        elif not g_active_processes and len(successful_ranks) < world_size:
            print("Coordinator: Monitoring loop ended, but not all workers succeeded and no failure was explicitly detected. This is unexpected. Aborting.")
            break

    # --- End of Outer Loop ---
    if restart_count > max_restarts:
        print("\n===== Coordinator: Training FAILED after maximum restarts. =====")
        sys.exit(1)
    elif len(successful_ranks) < world_size and not failure_detected:
         print("\n===== Coordinator: Training ended INCONCLUSIVELY. =====")
         sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fault-Tolerant PyTorch Distributed Training Coordinator (CPU)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Worker arguments (passed through to worker.py) ---
    worker_group = parser.add_argument_group('Worker Configuration (passed to worker.py)')
    worker_group.add_argument('--num_workers', type=int, default=2, help='Number of worker processes (world size)')
    worker_group.add_argument('--master_addr', type=str, default=DEFAULT_MASTER_ADDR, help='Master node address for workers')
    worker_group.add_argument('--master_port', type=str, default=DEFAULT_MASTER_PORT, help='Master node port for workers')
    worker_group.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    worker_group.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    worker_group.add_argument('--batch_size', type=int, default=16, help='Batch size per worker')
    worker_group.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    worker_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for saving/loading checkpoints')
    worker_group.add_argument('--checkpoint_interval', type=int, default=20, help='Save checkpoint every N global steps')

    # --- Coordinator specific arguments ---
    coord_group = parser.add_argument_group('Coordinator Configuration')
    coord_group.add_argument('--poll_interval', type=float, default=DEFAULT_POLL_INTERVAL, help='Seconds between checking worker status')
    coord_group.add_argument('--max_restarts', type=int, default=DEFAULT_MAX_RESTARTS, help='Maximum number of automatic restarts on failure')
    coord_group.add_argument('--restart_wait_time', type=float, default=DEFAULT_RESTART_WAIT_TIME, help='Seconds to wait before restarting workers after failure')

    args = parser.parse_args()

    # --- Basic Validation ---
    if args.num_workers <= 0:
        print("Coordinator: ERROR - Number of workers must be positive.")
        sys.exit(1)
    if args.checkpoint_interval <= 0:
        print("Coordinator: Warning - Checkpoint interval should be positive.")

    # --- Ensure Checkpoint Directory Exists ---
    try:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Coordinator: Ensured checkpoint directory exists: {args.checkpoint_dir}")
    except OSError as e:
        print(f"Coordinator: ERROR - Could not create checkpoint directory '{args.checkpoint_dir}': {e}")
        print("Coordinator: Please check path and permissions. Aborting.")
        sys.exit(1)

    # --- Run Main Coordinator Logic ---
    main(args)