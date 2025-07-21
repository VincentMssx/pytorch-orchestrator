# PyTorch Orchestrator

This project demonstrates a fault-tolerant, distributed PyTorch training setup using Docker and Kubernetes. It is designed to be resilient to node failures, automatically resuming training from the last saved checkpoint.

## Key Features

- **Distributed Training**: Leverages `torchrun` and `torch.distributed` for elastic, distributed training.
- **Fault Tolerance**: Automatically resumes from the last checkpoint upon worker failure and restart.
- **Stateful Checkpointing**: Periodically saves model, optimizer, and training progress.
- **Orchestration**:
    - **Local Development**: Uses Docker Compose to simulate a multi-node environment.
    - **Production**: Provides Kubernetes manifests for deployment in a cluster.
- **CI/CD**: Includes a GitHub Actions workflow for continuous integration.

## Architecture

The core of the system is the `worker.py` script, which acts as a generic training worker. The role of "master" or "worker" is determined at runtime by the orchestrator.

- **`worker.py`**:
    - A single, containerized Python script that can run as any node in the distributed training job.
    - On startup, it initializes the distributed process group and loads the latest checkpoint from a shared volume.
    - It runs a standard PyTorch training loop, periodically saving checkpoints.
- **Docker Image**:
    - A single Docker image is built containing the `worker.py` script and all its dependencies.
- **Orchestrators**:
    - **Docker Compose**: For local development, it defines `master` and `worker` services that use the same Docker image but with different commands to establish the distributed environment.
    - **Kubernetes**: For a more production-like environment, `StatefulSet` and `Service` manifests are provided to manage the master and worker pods.

## How to Run

### Prerequisites

- Docker
- Docker Compose
- (Optional) A Kubernetes cluster (e.g., Kind, Minikube, or a cloud provider's)

### Local Development with Docker Compose

1.  **Build and Run the Services:**
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker image and start one `master` and one `worker` container. The `master` will act as the rendezvous point.

2.  **Scaling Workers:**
    To run with more workers, use the `--scale` flag:
    ```bash
    docker-compose up --build --scale worker=3
    ```

3.  **View Logs:**
    You can view the logs of the services to see the training progress:
    ```bash
    docker-compose logs -f
    ```

### Deployment with Kubernetes

1.  **Build and Load the Image:**
    First, build the Docker image. If you are using a local cluster like Kind, you'll need to load the image into the cluster.
    ```bash
    docker build -t pytorch-orchestrator:latest .
    kind load docker-image pytorch-orchestrator:latest
    ```

2.  **Deploy the Manifests:**
    Apply the Kubernetes manifests to your cluster:
    ```bash
    kubectl apply -f kubernetes/master-service.yaml
    kubectl apply -f kubernetes/master-statefulset.yaml
    kubectl apply -f kubernetes/worker-statefulset.yaml
    ```

3.  **Check the Pods:**
    You can monitor the status of the pods:
    ```bash
    kubectl get pods -l app=pytorch-distributed
    ```

## CI/CD

The project includes a CI workflow in `.github/workflows/ci-cd.yml`. This workflow is triggered on pushes and pull requests to the `main` and `feature/ci-cd` branches. It builds the Docker image to ensure that the project is always in a buildable state.