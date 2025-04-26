import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import argparse
import os
import time
import traceback

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
    size = float(world_size)
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def save_checkpoint(state, filename="checkpoint.pt"):
    # Saves checkpoint only from rank 0
    print(f"Rank 0: Attempting to save checkpoint to {filename}")

    checkpoint_dir = os.path.dirname(filename)
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Rank 0: Created checkpoint directory {checkpoint_dir}")
        except OSError as e:
            print(f"Rank 0: Error creating checkpoint directory: {e}")
            return

    try:
        torch.save(state, filename)
        print(f"Rank 0: Checkpoint saved successfully to {filename}")
    except Exception as e:
        print(f"Rank 0: Error saving checkpoint: {e}")


def run_training(rank, world_size, epochs, steps_per_epoch, batch_size, lr,
                 checkpoint_dir, checkpoint_interval, resume_from_checkpoint):
    print(f"Rank {rank}: Starting training...")
    dist.barrier()
    print(f"Rank {rank}: Barrier passed, setting up model and data.")

    # Model setup
    input_size = 10
    output_size = 1
    model = SimpleModel(input_size=input_size, output_size=output_size)

    # Optimizer and Loss
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Loading Checkpoint
    start_epoch = 0
    next_step = 0
    if resume_from_checkpoint and os.path.isfile(resume_from_checkpoint):
        print(f"Rank {rank}: Attempting to load checkpoint '{resume_from_checkpoint}'")
        try:
            checkpoint = torch.load(resume_from_checkpoint, map_location='cpu')

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            completed_step = checkpoint['step']
            next_step = completed_step + 1

            print(f"Rank {rank}: Loaded checkpoint. Resuming from Epoch {start_epoch + 1}, Step {next_step}")

        except Exception as e:
            print(f"Rank {rank}: Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0
            next_step = 0
            
    else:
        if resume_from_checkpoint:
             print(f"Rank {rank}: Checkpoint file not found at '{resume_from_checkpoint}'. Starting from scratch.")
        else:
             print(f"Rank {rank}: No checkpoint provided. Starting from scratch.")

    # Ensure all workers have loaded the checkpoint before proceeding
    dist.barrier()

    # Training loop
    global_step = 0
    for epoch in range(start_epoch, epochs):
        current_epoch_step = 0
        epoch_start_step = 0

        if epoch == start_epoch and next_step > 0:
             epoch_start_step = next_step
             print(f"Rank {rank}: Resuming epoch {epoch+1} from step {epoch_start_step}")

             if epoch_start_step >= steps_per_epoch:
                 print(f"Rank {rank}: Checkpoint was at end of epoch {epoch+1}, skipping to next epoch.")
                 continue

        if rank == 0:
            print(f"\n--- Starting Epoch {epoch+1}/{epochs} ---")
        dist.barrier()

        start_time = time.time()
        total_loss = 0.0

        for step in range(epoch_start_step, steps_per_epoch):
            global_step = epoch * steps_per_epoch + step

            # Generate synthetic data
            inputs = torch.randn(batch_size, input_size)
            targets = torch.randn(batch_size, output_size)

            # Training step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Average gradients
            average_gradients(model, world_size)
            optimizer.step()

            total_loss += loss.item()
            current_epoch_step += 1

            # Save Checkpoint
            current_global_step_completed = global_step
            if (current_global_step_completed + 1) % checkpoint_interval == 0:
                dist.barrier()
                if rank == 0:
                    checkpoint_filename = os.path.join(checkpoint_dir, f"checkpoint_step_{current_global_step_completed + 1}.pt")
                    state = {
                        'epoch': epoch,
                        'step': current_global_step_completed,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }
                    save_checkpoint(state, checkpoint_filename)
                dist.barrier()

            if rank == 0 and (step + 1) % 10 == 0:
                print(f"Rank {rank} | Epoch {epoch+1} | Step {step+1}/{steps_per_epoch} | Global Step: {current_global_step_completed + 1} | Loss: {loss.item():.4f}")

        dist.barrier()
        epoch_time = time.time() - start_time
        
        avg_loss = total_loss / current_epoch_step if current_epoch_step > 0 else 0 # Avoid division by zero if epoch was skipped or had no steps run
        if rank == 0:
            print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Steps Run: {current_epoch_step}, Time: {epoch_time:.2f}s")

    print(f"Rank {rank}: Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Distributed Worker with Checkpointing')
    parser.add_argument('--rank', type=int, required=True, help='Rank of the process')
    parser.add_argument('--world_size', type=int, required=True, help='Number of processes participating')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='29500', help='Master node port')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--steps', type=int, default=50, help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per worker')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Save checkpoint every N steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint file to resume from')


    args = parser.parse_args()

    print(f"Starting worker with Rank: {args.rank}, World Size: {args.world_size}, Port: {args.master_port}")

    if args.rank == 0:
         os.makedirs(args.checkpoint_dir, exist_ok=True)

    setup(args.rank, args.world_size, args.master_addr, args.master_port)

    try:
        run_training(
            rank=args.rank,
            world_size=args.world_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    except Exception as e:
        print(f"Rank {args.rank}: Error during training - {e}")
        traceback.print_exc()
    finally:
        dist.barrier()
        print(f"Rank {args.rank}: Reached final barrier.")
        cleanup()