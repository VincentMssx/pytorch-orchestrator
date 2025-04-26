import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import argparse
import os
import time

from models.SimpleModel import SimpleModel

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    print(f"Rank {rank}: Initializing process group...")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized.")

def cleanup():
    dist.destroy_process_group()
    print("Cleaned up process group.")

def average_gradients(model, world_size):
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

def run_training(rank, world_size, epochs=2, steps_per_epoch=50, batch_size=16, lr=0.01):
    print(f"Rank {rank}: Starting training...")
    dist.barrier()
    print(f"Rank {rank}: Barrier passed, setting up model and data.")

    # Model setup
    input_size = 10
    output_size = 1
    model = SimpleModel(input_size=input_size, output_size=output_size)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        if rank == 0: # Print epoch status only from rank 0
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Wait all processes before starting epoch
        dist.barrier()

        start_time = time.time()
        total_loss = 0.0
        for step in range(steps_per_epoch):
            # Generate synthetic data for each step
            inputs = torch.randn(batch_size, input_size)
            targets = torch.randn(batch_size, output_size)

            # Forward step
            optimizer.zero_grad()
            outputs = model(inputs)

            # Backward step
            loss = criterion(outputs, targets)
            loss.backward()
            average_gradients(model, world_size)
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and (step + 1) % 10 == 0:
                print(f"Rank {rank} | Epoch {epoch+1} | Step {step+1}/{steps_per_epoch} | Loss: {loss.item():.4f}")

        # Sync after each epoch completes
        dist.barrier()
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / steps_per_epoch
        if rank == 0:
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

    print(f"Rank {rank}: Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, required=True, help='Rank of the process')
    parser.add_argument('--world_size', type=int, required=True, help='Number of processes participating')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')

    args = parser.parse_args()

    print(f"Starting worker with Rank: {args.rank}, World Size: {args.world_size}")

    try:
        setup(args.rank, args.world_size, args.master_addr, args.master_port)
    except Exception as e:
        print(f"Rank {args.rank}: Error during setup - {e}")

    try:
        run_training(
            args.rank,
            args.world_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps
        )
    except Exception as e:
        print(f"Rank {args.rank}: Error during training - {e}")
    finally:
        dist.barrier()
        print(f"Rank {args.rank}: Reached final barrier.")
        cleanup()