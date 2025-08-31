# app.py
import asyncio
import json
import logging
import re
import time
from typing import List, Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from kubernetes import client, config, watch

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")

# --- Kubernetes Client Setup ---
try:
    config.load_incluster_config()
    logger.info("Loaded in-cluster Kubernetes config.")
except config.ConfigException:
    try:
        config.load_kube_config()
    except config.ConfigException:
        logger.error("Could not configure Kubernetes client.")

v1 = client.CoreV1Api()
JOB_NAME = "master-job"
NAMESPACE = "default"
JOB_LABEL_SELECTOR = "app=pytorch-orchestrator"

@asynccontextmanager
async def lifespan(app: FastAPI):
    monitor_task = asyncio.create_task(monitor_training_job())
    yield
    monitor_task.cancel()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- FastAPI Routes ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles WebSocket connections from clients."""
    logger.info("WebSocket connection attempt from a client.")
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected.")

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        logger.info(f"Broadcasting message to {len(self.active_connections)} clients.")
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# --- Data Fetching Logic ---
def get_master_pod_name() -> str | None:
    """Finds the master pod using its specific role label."""
    try:
        # Look for a pod with the 'role=master' label
        pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=f"app=pytorch-orchestrator,role=master"
        )
        running_pods = [p for p in pods.items if p.status.phase == "Running"]
        if running_pods:
            running_pods.sort(key=lambda p: p.metadata.creation_timestamp)
            return running_pods[0].metadata.name
        return None
    except client.ApiException as e:
        logger.error(f"K8s API error getting master pod: {e}")
        return None

def get_worker_count() -> int:
    """Counts the number of currently running worker pods for the job."""
    try:
        pods = v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=JOB_LABEL_SELECTOR
        )
        # Count pods that are actively running or have not yet failed
        active_pods = [p for p in pods.items if p.status.phase in ["Running", "Pending"]]
        return len(active_pods)
    except client.ApiException as e:
        logger.error(f"K8s API error getting worker count: {e}")
        return 0

def parse_logs_for_loss(log_lines: List[str]) -> float | None:
    """Parses a list of log lines to find the most recent loss value."""
    loss_pattern = re.compile(r"Loss: (\d+\.\d+)")
    last_loss = None
    for line in reversed(log_lines):
        match = loss_pattern.search(line)
        if match:
            try:
                last_loss = float(match.group(1))
                return last_loss
            except (ValueError, IndexError):
                continue
    return last_loss

def get_logs_and_loss(pod_name: str) -> (List[str], float | None):
    """Fetches recent logs from the master pod and parses the latest loss."""
    try:
        # Fetch the last 50 lines to keep it efficient
        logs_str = v1.read_namespaced_pod_log(
            name=pod_name, namespace=NAMESPACE, tail_lines=50
        )
        log_lines = logs_str.strip().split('\n')
        loss = parse_logs_for_loss(log_lines)
        return log_lines, loss
    except client.ApiException:
        # This can happen if the pod is terminating
        return ["Log stream unavailable..."], None


# --- Background Monitoring Task ---
async def monitor_training_job():
    """Periodically checks the K8s job and broadcasts updates to clients."""
    while True:
        await asyncio.sleep(1)  # Poll every 3 seconds
        logger.info("--- Monitor Task: Starting new check ---")
        master_pod = get_master_pod_name()
        worker_count = get_worker_count()
        current_timestamp = time.time()
        logger.info(f"Monitor Task: Found master_pod={master_pod}, worker_count={worker_count}")
        
        log_lines = [f"[INFO] Looking for job '{JOB_NAME}'..."]
        loss_value = None

        if master_pod:
            log_lines, loss_value = get_logs_and_loss(master_pod)
        elif worker_count > 0:
            log_lines = ["[INFO] Workers are starting up, waiting for logs..."]
        else:
            log_lines = [f"[INFO] No running workers found for job '{JOB_NAME}'. Waiting for orchestrator to start one."]

        # Bundle the data in the format expected by the frontend
        bundle = {
            "type": "bundle",
            "data": {
                "workers": {"timestamp": current_timestamp, "count": worker_count},
                "loss": {"timestamp": current_timestamp, "value": loss_value},
                "logs": log_lines,
            },
        }
        logger.info(f"Monitor Task: Prepared bundle. Broadcasting to {len(manager.active_connections)} clients.")
        
        await manager.broadcast(json.dumps(bundle))
