# Fault-Tolerant PyTorch Training with Docker and Kubernetes

## Project Goal

This project demonstrates a robust, fault-tolerant orchestration system for distributed PyTorch training using standard containerization and orchestration tools.

- **Distributed Training**: Utilizes PyTorch `distributed` with the `gloo` backend.
- **Stateful Checkpointing**: Saves the model, optimizer, and training progress to a shared volume.
- **Automated Fault Tolerance**: Leverages container orchestrators (Kubernetes or Docker Compose) to automatically restart failed worker processes, resuming from the last valid checkpoint.

This updated version replaces the custom Python-based coordinator with a more production-ready approach, making it suitable for demonstrating skills in MLOps, reliability, and systems engineering for modern machine learning workflows.

## Architecture

The system consists of a single, containerized Python script:

1.  **`worker.py`**:
    - Represents a single node/process in the distributed training job.
    - On startup, it reads its `RANK`, `WORLD_SIZE`, and `MASTER_ADDR` from environment variables, which are injected by the container orchestrator.
    - **Fault Tolerance**: Before training begins, it scans the shared checkpoint directory (`/mnt/checkpoints`) to find the latest valid checkpoint and automatically resumes from there.
    - Initializes `torch.distributed` using the `gloo` backend.
    - Runs a standard training loop with a simple `SimpleModel`.
    - **Checkpoint Saving**: Periodically, Rank 0 saves a checkpoint to the shared volume. Barriers ensure synchronization around the save operation.

2.  **Container Orchestrator (Kubernetes or Docker Compose)**:
    - **Responsibilities**: Launching, monitoring, and ensuring the desired number of worker containers are running.
    - **Fault Tolerance**: If a container exits unexpectedly, the orchestrator automatically relaunches it. The new container picks up from the last checkpoint, ensuring the training job continues with minimal interruption.

## Tech Stack

- **Language**: Python 3.12+
- **ML Framework**: PyTorch
- **Containerization**: Docker
- **Orchestration**: Kubernetes, Docker Compose

## How to Run

### Local Development (with Docker Compose)

This is the simplest way to run the project on a local machine.

**Prerequisites**:
- Docker
- Docker Compose

**Instructions**:

1.  **Build and Run**:

    Open a terminal in the project root and execute the run script:

    ```bash
    ./run-local.sh
    ```

    This script will:
    - Build the `pytorch-orchestrator` Docker image.
    - Start one `master` and one `worker-1` container.
    - Mount the local `./checkpoints` directory into the containers, allowing them to share progress.

2.  **Simulating a Failure**:

    To see the fault tolerance in action, open another terminal and run:

    ```bash
    docker kill pytorch-worker-1
    ```

    Docker Compose will automatically restart the container. You can observe in the logs that the new container loads the latest checkpoint and seamlessly resumes training.

### Kubernetes Deployment

For a production-style deployment, you can use the provided Kubernetes manifests.

**Prerequisites**:
- A running Kubernetes cluster (e.g., Minikube, Kind, or a cloud provider's EKS/GKE/AKS).
- `kubectl` configured to point to your cluster.
- A shared storage solution (e.g., an NFS provisioner) for the `ReadWriteMany` access mode required for checkpoints if running multiple pods on different cluster nodes. For a single-node cluster, the default `ReadWriteOnce` will suffice.

**Instructions**:

1.  **Build and Push the Docker Image**:

    Build the image and push it to a registry that your Kubernetes cluster can access (e.g., Docker Hub, GCR, ECR).

    ```bash
    docker build -t your-registry/pytorch-orchestrator:latest .
    docker push your-registry/pytorch-orchestrator:latest
    ```

    *Remember to update the image name in `kubernetes/worker-statefulset.yaml`.*

2.  **Deploy to Kubernetes**:

    Apply the Kubernetes manifests:

    ```bash
    # Create the headless service for the master
    kubectl apply -f kubernetes/master-service.yaml

    # Create the StatefulSet for the workers
    kubectl apply -f kubernetes/worker-statefulset.yaml
    ```

3.  **Monitor the Job**:

    Check the status of the pods and view their logs:

    ```bash
    kubectl get pods -l app=pytorch-distributed
    kubectl logs -f pytorch-worker-0
    ```

4.  **Cleanup**:

    To delete the resources from your cluster:

    ```bash
    kubectl delete statefulset pytorch-worker
    kubectl delete service pytorch-master-svc
    ```
