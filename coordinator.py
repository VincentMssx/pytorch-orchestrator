import argparse
import subprocess
import os
import time
import glob
import re
import sys
import signal


def find_latest_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint file based on step number."""
    latest_checkpoint = None
    latest_step = -1
    if not os.path.isdir(checkpoint_dir):
        print(f"Coordinator: Checkpoint directory '{checkpoint_dir}' not found.")
        return None
    pattern = re.compile(r"checkpoint_step_(\d+)\.pt")
    try:
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    except OSError as e:
        print(f"Coordinator: Error accessing checkpoint directory '{checkpoint_dir}': {e}")
        return None

    if not checkpoint_files:
        print(f"Coordinator: No checkpoint files found in '{checkpoint_dir}'.")
        return None

    for ckpt_file in checkpoint_files:
        match = pattern.search(os.path.basename(ckpt_file))
        if match:
            try:
                step = int(match.group(1))
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = ckpt_file
            except ValueError:
                 print(f"Coordinator: Warning - could not parse step number from '{ckpt_file}'. Skipping.")
                 continue

    if latest_checkpoint:
        print(f"Coordinator: Found latest checkpoint '{latest_checkpoint}' at step {latest_step}.")
    else:
         print(f"Coordinator: Found checkpoint files, but couldn't parse step number from names.")

    return latest_checkpoint


def launch_single_worker(rank, world_size, args, resume_checkpoint_path):
    """Launches a single worker process and returns its Popen object."""
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

    print(f"Coordinator: Launching Rank {rank}...")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Coordinator: Rank {rank} launched with PID {process.pid}.")
        return process
    except Exception as e:
        print(f"Coordinator: Failed to launch Rank {rank}: {e}")
        return None


def cleanup_processes(processes_dict, reason="cleanup"):
    """Terminates or kills remaining processes."""
    print(f"Coordinator: Initiating {reason} for {len(processes_dict)} processes...")
    for rank, process in list(processes_dict.items()):
        if process.poll() is None:
            print(f"Coordinator: Terminating Rank {rank} (PID {process.pid})...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"Coordinator: Rank {rank} terminated gracefully.")
            except subprocess.TimeoutExpired:
                print(f"Coordinator: Rank {rank} did not terminate gracefully. Killing...")
                process.kill()
                try:
                    process.wait(timeout=2)
                    print(f"Coordinator: Rank {rank} killed.")
                except Exception as e:
                    print(f"Coordinator: Error waiting for killed Rank {rank}: {e}")
            except Exception as e:
                print(f"Coordinator: Error waiting for terminated Rank {rank}: {e}")
        else:
             print(f"Coordinator: Rank {rank} (PID {process.pid}) already exited with code {process.poll()}.")

        if rank in processes_dict:
            del processes_dict[rank]


def main(args):
    world_size = args.num_workers
    active_processes = {}
    restart_count = 0
    max_restarts = args.max_restarts


    def signal_handler(sig, frame):
        print("\nCoordinator: Ctrl+C detected. Initiating shutdown...")
        cleanup_processes(active_processes, reason="shutdown signal")
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler) # Register handler for Ctrl+C

    while restart_count <= max_restarts:
        print(f"\n===== Coordinator: Starting Training Attempt {restart_count + 1} =====")

        # --- Find latest checkpoint ---
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            print(f"Coordinator: Will attempt to resume from {latest_checkpoint}")
        else:
            print("Coordinator: No checkpoint found, starting from scratch.")

        # --- Launch all workers for this attempt ---
        processes_launched_successfully = True
        for rank in range(world_size):
            process = launch_single_worker(rank, world_size, args, latest_checkpoint)
            if process:
                active_processes[rank] = process
            else:
                print(f"Coordinator: Critical error launching Rank {rank} on attempt {restart_count + 1}. Aborting this attempt.")
                processes_launched_successfully = False
                break

        if not processes_launched_successfully:
            cleanup_processes(active_processes, reason="launch failure cleanup")
            restart_count += 1
            print(f"Coordinator: Waiting {args.restart_wait_time} seconds before next attempt...")
            time.sleep(args.restart_wait_time)
            continue

        print(f"\nCoordinator: All {world_size} workers launched for attempt {restart_count + 1}. Monitoring...")

        # --- Monitoring Loop ---
        failure_detected = False
        successful_ranks = set()

        while len(active_processes) > 0:
            failed_rank = -1
            for rank, process in list(active_processes.items()):
                return_code = process.poll()

                if return_code is None:
                    continue
                elif return_code == 0:
                    print(f"Coordinator: Worker Rank {rank} (PID {process.pid}) finished successfully.")
                    successful_ranks.add(rank)
                    del active_processes[rank]
                else:
                    print(f"Coordinator: FAILURE DETECTED! Worker Rank {rank} (PID {process.pid}) exited unexpectedly with code {return_code}.")
                    failure_detected = True
                    failed_rank = rank
                    del active_processes[rank]
                    break

            if failure_detected:
                break

            if not active_processes and len(successful_ranks) == world_size:
                print("\nCoordinator: All workers completed successfully!")
                return

            time.sleep(args.poll_interval)

        # --- Handle Failure or Completion ---
        if failure_detected:
            print(f"Coordinator: Failure in Rank {failed_rank} triggered cleanup and restart.")
            cleanup_processes(active_processes, reason=f"failure of Rank {failed_rank}")
            restart_count += 1
            if restart_count <= max_restarts:
                 print(f"Coordinator: Restart attempt {restart_count}/{max_restarts} will begin after {args.restart_wait_time} seconds...")
                 time.sleep(args.restart_wait_time)
            else:
                 print(f"Coordinator: Maximum restart limit ({max_restarts}) reached. Aborting.")
                 break

        elif not active_processes and len(successful_ranks) < world_size:
             print("Coordinator: Monitoring loop exited but not all workers succeeded and no failure was explicitly detected. This is unexpected.")
             break


    # End of outer loop
    if restart_count > max_restarts:
        print("\n===== Coordinator: Training FAILED after maximum restarts. =====")
        sys.exit(1)
    else:
        # This part should only be reached if successful exit happened via 'return' earlier
        print("\n===== Coordinator: Training finished. =====")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fault-Tolerant PyTorch Distributed Training Coordinator')
    # Worker args
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes (world size)')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory for checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=20, help='Save checkpoint every N global steps')
    # Coordinator args
    parser.add_argument('--poll_interval', type=float, default=2.0, help='Seconds between checking worker status')
    parser.add_argument('--max_restarts', type=int, default=3, help='Maximum number of automatic restarts on failure')
    parser.add_argument('--restart_wait_time', type=float, default=5.0, help='Seconds to wait before restarting workers after failure')

    args = parser.parse_args()

    # Ensure checkpoint directory exists before starting
    try:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Coordinator: Ensured checkpoint directory exists: {args.checkpoint_dir}")
    except OSError as e:
        print(f"Coordinator: Error creating checkpoint directory '{args.checkpoint_dir}': {e}")
        sys.exit(1)


    main(args)