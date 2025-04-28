# Simulated Fault-Tolerant PyTorch Training Orchestrator

## Project Goal

This project demonstrates a basic fault-tolerant orchestration system for simulated distributed PyTorch training.

*   Distributed training coordination (using PyTorch `distributed` with the `gloo` backend).
*   Stateful checkpointing (model, optimizer, training progress).
*   Fault detection of worker processes.
*   Automatic recovery and resumption from the last valid checkpoint.

This was built as a demonstration project targeting roles focused on ML infrastructure, reliability, and systems engineering, particularly within the context of PyTorch training workflows.

## Architecture

The system consists of two main Python scripts:

1.  **`coordinator.py`**:
    *   Acts as the master process.
    *   Responsible for launching a specified number (`--num_workers`) of worker processes.
    *   Continuously monitors the health of worker processes using non-blocking checks (`process.poll()`).
    *   **Fault Tolerance:** If a worker process exits unexpectedly (non-zero exit code), the coordinator:
        1.  Detects the failure.
        2.  Terminates all other *running* worker processes gracefully (with a kill fallback).
        3.  Scans the checkpoint directory (`--checkpoint_dir`) to find the *latest valid checkpoint* based on the step number in the filename (e.g., `checkpoint_step_100.pt`).
        4.  Increments a restart counter.
        5.  If the restart limit (`--max_restarts`) is not exceeded, it relaunches *all* worker processes, providing the path to the latest checkpoint so they can resume.
    *   Handles graceful completion if all workers finish successfully.
    *   Handles graceful shutdown on `Ctrl+C` (SIGINT).

2.  **`worker.py`**:
    *   Represents a single node/process in the distributed training job.
    *   Initializes `torch.distributed` using the `gloo` backend and parameters provided by the coordinator.
    *   Defines a simple `SimpleModel` for demonstration.
    *   Runs a basic training loop using synthetic data.
    *   Performs manual gradient averaging across all workers using `dist.all_reduce`.
    *   **Checkpoint Saving:** Periodically (every `--checkpoint_interval` global steps), Rank 0 saves a checkpoint containing the model state dict, optimizer state dict, current epoch, and completed global step number. Barriers ensure synchronization around saving.
    *   **Checkpoint Loading:** On startup, checks if a `--resume_from_checkpoint` path is provided. If so, loads the state from the checkpoint file before starting the training loop, resuming from the correct epoch and step. Includes barriers for synchronization.

The interaction simulates a scenario where worker failures can occur, and the coordinator ensures the overall training job can continue with minimal data loss (limited by checkpoint frequency).

## Tech Stack

*   **Language:** Python 3.12+
*   **ML Framework:** PyTorch
*   **Process Management:** Python `subprocess` module 

## Setup

Run this powershell script for setup
```powershell
.\setup.ps1
```

## How to Run

Execute the coordinator script from the project's root directory:

```powershell
python coordinator.py --num_workers 2 --epochs 20 --steps 30 --checkpoint_interval 10 --max_restarts 3 --poll_interval 1
```