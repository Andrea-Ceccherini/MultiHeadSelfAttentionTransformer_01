"""
This script (mh_sat_phase_1_fast_memory_map.py) is the core "Brain Builder". It performs the heavy lifting of teaching the
neural network how to understand and generate English.

The train_fast function is a classic Deep Learning training loop.

    Forward Pass: It feeds numbers (tokens) into the model (output = model(...)).

    Loss Calculation: It calculates how wrong the model's guess was compared to the actual text (loss = criterion(...)).

    Backward Pass: It calculates the gradients (loss.backward()).

    Optimization: It updates the model's internal numbers (weights) to make it smarter (optimizer.step()).

This process happens thousands of times per hour. This IS the learning phase.

create the Model
    In Memory: The line model = CustomTransformer(...) creates the "empty brain" (randomly initialized neural network) inside your RAM.

    On Disk: The code save_file(model.state_dict(), ... "latest_checkpoint.safetensors") takes that "brain," which is getting smarter every minute, and saves it to your hard drive.

The Result

    When this script finishes (or when you stop it), you will have a file named latest_checkpoint.safetensors in
    unsupervised_model_weights/latest_checkpoint.safetensors.

    The script only creates the folder and the file when it hits Step 1000.

    Your script is configured to save a checkpoint every 1,000 steps.

    Last Save: Step 1,000

    Next Save: Step 2,000

    Following Saves: Step 3,000, 4,000, 5,000, etc.

    Target Loss: ~3.5 to 4.0 that could around step 50000 of epoch 1. You can stop manually when target loos reaches
    ~3.5 to 4.0
    If you don't stop manually it will run till epoch 1 is finishes that will happen around step 525,690
    The Math

    Total Tokens: 4,306,458,971
    Tokens consumed per Step:
        Sequence Length: 256
        Physical Batch: 4
        Accumulation: 8
        Total: 256×4×8=8,192256×4×8=8,192 tokens per step.

        4,306,458,971 tokens/8,192 tokens/step ≈ 525,690 Steps

    You do not need to reach Step 525,690. The model will likely be fully trained and smart enough between Step 50,000 and Step 100,000.
    If you stop script at step 3000 and then you re-lunch it after 4 hours it will start from the saved latest_checkpoint.safetensors stopped step 3000?


That file IS your model. You can load that file later to chat with it, fine-tune it on liver data, or use it for any other English task.

"""


import os
import sys
import numpy as np
from datetime import datetime

# --- SYSTEM CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file

from mh_sat_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_gpt2_tokenizer
)

# --- CONFIGURATION ---
DATA_BIN_PATH = "../wiki_books_dataset/wiki_books_dataset.bin"
TOKENIZATION_MAX_LENGTH = 256
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2

# --- STABLE TRAINING PARAMS ---
BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4


# --- MEMORY MAPPED DATASET ---
class MemmapDataset(Dataset):
    def __init__(self, bin_path, block_size):
        self.block_size = block_size

        # Load the binary file as a memory-mapped array
        # This is INSTANT and uses almost 0 RAM (OS handles paging)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        print(f"Dataset loaded. Total Tokens: {len(self.data):,}")

    def __len__(self):
        # Number of possible chunks
        return len(self.data) // self.block_size - 1

    def __getitem__(self, idx):
        # Grab a chunk of data
        # We start at idx * block_size
        # But for better randomness, usually we do random sampling.
        # Since DataLoader shuffles 'idx', this is fine.
        start = idx * self.block_size
        end = start + self.block_size

        # Convert uint16 back to long for PyTorch
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {'input_ids': chunk}


def train_fast(epochs_, dataloader_, device_, optimizer_, criterion_, model_, save_dir_):
    print("--- Phase 1 Training (Fast Memmap Mode - BULLETPROOF) BEGIN ---")
    model_.to(device_)
    model_.train()

    scaler = torch.amp.GradScaler("cuda")
    steps = 0
    total_loss = 0
    optimizer_.zero_grad(set_to_none=True)

    for epoch in range(epochs_):
        print(f"\n--- Epoch {epoch + 1}/{epochs_} ---")

        for i, batch in enumerate(dataloader_):
            src_data = batch['input_ids'].to(device_)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            with torch.autocast("cuda", dtype=torch.float16):
                output = model_(src_data, decoder_input)
                loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

                # Check for NaN immediately
                if torch.isnan(loss):
                    print(f"\n⚠️ WARNING: NaN detected at batch {i}. Skipping update.")
                    optimizer_.zero_grad(set_to_none=True)
                    continue

                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            current_loss_val = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss_val

            if (i + 1) % ACCUMULATION_STEPS == 0:
                # --- SAFETY BRAKE: Gradient Clipping ---
                # Unscale the gradients so we can check their size
                scaler.unscale_(optimizer_)
                # Clip gradients to max norm 1.0 (Prevents explosion)
                torch.nn.utils.clip_grad_norm_(model_.parameters(), 1.0)

                scaler.step(optimizer_)
                scaler.update()
                optimizer_.zero_grad(set_to_none=True)
                steps += 1

                if steps % 100 == 0:
                    avg = total_loss / (i + 1)
                    print(f"\rStep {steps} | Loss: {current_loss_val:.4f} | Avg: {avg:.4f}", end="", flush=True)

                if steps % 1000 == 0:
                    print(f"\n   💾 Saving Checkpoint at Step {steps}...")
                    os.makedirs(save_dir_, exist_ok=True)
                    save_file(model_.state_dict(), os.path.join(save_dir_, "latest_checkpoint.safetensors"))
                    torch.cuda.empty_cache()

        print(f"\nEpoch {epoch + 1} Complete.")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    begin_time = datetime.now()

    if not os.path.exists(DATA_BIN_PATH):
        print(f"❌ Error: {DATA_BIN_PATH} not found. Run the preprocessor script first!")
        sys.exit(1)

    tokenizer, vocab_size = load_gpt2_tokenizer()

    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Check if a checkpoint exists - Resume Logic if latest_checkpoint.safetensors has been stopped at certain number of nx1000 steps with n = 1, 2, 3, ...
    checkpoint_path = os.path.join("../mh_sat_01_01/unsupervised_model_weights", "latest_checkpoint.safetensors")

    if os.path.exists(checkpoint_path):
        print(f"🔄 Found checkpoint: {checkpoint_path}")
        print("   Loading weights to RESUME training...")
        try:
            # Load the weights into the model
            from safetensors.torch import load_file

            model.load_state_dict(load_file(checkpoint_path))
            print("   ✅ Resume successful! The model is smart again.")
        except Exception as e:
            print(f"   ⚠️ Error loading checkpoint: {e}")
            print("   Starting from scratch.")
    else:
        print("🆕 No checkpoint found. Starting training from scratch.")

    # Dataset (Instant Load)
    dataset = MemmapDataset(DATA_BIN_PATH, TOKENIZATION_MAX_LENGTH)

    # DataLoader (Standard Map-Style allows SHUFFLE=TRUE which is better for learning)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,  # You can try increasing this to 4 or 8 since we read from binary
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train
    train_fast(1, dataloader, device, optimizer, criterion, model, "../mh_sat_01_01/unsupervised_model_weights")

    print(f"Elapsed: {datetime.now() - begin_time}")