import time
import logging
import subprocess

# --- Configuration ---
KUBE_MANIFEST_PATH = "kubernetes/job.yaml"
JOB_NAME = "master-job"
NAMESPACE = "default"
WAIT_TIMEOUT_SECONDS = 3600  # 1 hour
SLEEP_INTERVAL_SECONDS = 20

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def run_command(command: list[str]):
    """Runs a command and logs its output."""
    try:
        logging.info(f"Running command: {' '.join(command)}")
        # Using synchronous subprocess.run
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise

def cleanup():
    """Deletes the Kubernetes resources defined in the manifest."""
    logging.info("Cleaning up Kubernetes resources...")
    run_command(["kubectl", "delete", "-f", KUBE_MANIFEST_PATH, "--ignore-not-found=true"])
    logging.info("Cleanup complete.")

def main_loop():
    """Main orchestration loop."""
    while True:
        try:
            logging.info("Applying Kubernetes manifests to start a new training job...")
            run_command(["kubectl", "apply", "-f", KUBE_MANIFEST_PATH])

            logging.info(f"Waiting for job '{JOB_NAME}' to complete...")
            wait_command = [
                "kubectl", "wait",
                f"--for=condition=complete",
                f"job/{JOB_NAME}",
                f"--namespace={NAMESPACE}",
                f"--timeout={WAIT_TIMEOUT_SECONDS}s",
            ]
            run_command(wait_command)
            logging.info(f"Job '{JOB_NAME}' completed successfully.")

        except Exception as e:
            logging.error(f"An error occurred during the training job: {e}")
            logging.error("Continuing to the next cycle after cleanup.")
        finally:
            cleanup()
            logging.info(f"Waiting for {SLEEP_INTERVAL_SECONDS} seconds before starting the next job...")
            time.sleep(SLEEP_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.warning("Orchestrator stopped by user. Cleaning up before exit.")
        cleanup()