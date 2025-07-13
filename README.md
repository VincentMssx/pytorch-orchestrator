# Fault-Tolerant PyTorch Training with Docker and Kubernetes

## Project Goal

This project demonstrates a robust, fault-tolerant orchestration system for distributed PyTorch training using standard containerization and orchestration tools.

- **Distributed Training**: Utilizes PyTorch `distributed` with the `gloo` backend.
- **Stateful Checkpointing**: Saves the model, optimizer, and training progress to a shared volume.
- **Automated Fault Tolerance**: Leverages container orchestrators (Kubernetes or Docker Compose) to automatically restart failed worker processes, resuming from the last valid checkpoint.

This updated version replaces the custom Python-based coordinator with a more production-ready approach, making it suitable for demonstrating skills in MLOps, reliability, and systems engineering for modern machine learning workflows.

## Architecture

The system consists of a single, containerized Python script:

1. **`worker.py`**:

   - Represents a single node/process in the distributed training job.
   - On startup, it reads its `RANK`, `WORLD_SIZE`, and `MASTER_ADDR` from environment variables, which are injected by the container orchestrator.
   - **Fault Tolerance**: Before training begins, it scans the shared checkpoint directory (`/mnt/checkpoints`) to find the latest valid checkpoint and automatically resumes from there.
   - Initializes `torch.distributed` using the `gloo` backend.
   - Runs a standard training loop with a simple `SimpleModel`.
   - **Checkpoint Saving**: Periodically, Rank 0 saves a checkpoint to the shared volume. Barriers ensure synchronization around the save operation.
2. **Container Orchestrator (Kubernetes or Docker Compose)**:

   - **Responsibilities**: Launching, monitoring, and ensuring the desired number of worker containers are running.
   - **Fault Tolerance**: If a container exits unexpectedly, the orchestrator automatically relaunches it. The new container picks up from the last checkpoint, ensuring the training job continues with minimal interruption.

## Tech Stack

- **Language**: Python 3.12+
- **ML Framework**: PyTorch
- **Containerization**: Docker
- **Orchestration**: Kubernetes, Docker Compose
