import os
import sys
import glob
from datetime import datetime

# --- SYSTEM CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Silence inductor logs
import logging
import torch._inductor.config

torch._inductor.config.verbose_progress = False
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from safetensors.torch import save_file

from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, load_gpt2_tokenizer
)

# --- CONFIGURATION ---
TOKENIZATION_MAX_LENGTH = 256
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2

# --- OPTIMIZED TRAINING PARAMS ---
BATCH_SIZE = 4  # Small to prevent Error 700
ACCUMULATION_STEPS = 8  # Effective Batch Size = 32 (Much better for learning grammar)
LEARNING_RATE = 1e-4


# --- DATASET ---
class RobustIterableDataset(IterableDataset):
    def __init__(self, folders, tokenizer, max_length):
        self.files = []
        for folder in folders:
            # Recursively find all txt files
            found = sorted(glob.glob(os.path.join(folder, "*.txt")))
            if not found:
                # Try recursive search if simple glob fails
                found = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
            self.files.extend(found)

        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Dataset initialized with {len(self.files)} files.")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        for idx, file_path in enumerate(self.files):
            # Print progress every file
            filename = os.path.basename(file_path)
            print(f"\n📂 [Progress] Opening file {idx + 1}/{len(self.files)}: {filename}")

            try:
                # FIX: errors='replace' allows reading Books with weird characters
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    buffer_text = ""
                    for line in f:
                        text = line.strip()
                        if not text: continue

                        buffer_text += " " + text

                        # Process when buffer is full enough
                        if len(buffer_text) > (self.max_length * 4):
                            tokenized_ids = self.process_text(buffer_text)
                            for seq in tokenized_ids:
                                yield {"input_ids": torch.tensor(seq, dtype=torch.long)}
                            buffer_text = ""

                    # Process remainder
                    if buffer_text:
                        tokenized_ids = self.process_text(buffer_text)
                        for seq in tokenized_ids:
                            yield {"input_ids": torch.tensor(seq, dtype=torch.long)}

            except Exception as e:
                print(f"⚠️ Skipped {filename}: {e}")
                continue

    def process_text(self, text):
        encodings = self.tokenizer(text, truncation=False, padding=False, return_attention_mask=False)['input_ids']
        sequences = []
        for i in range(0, len(encodings), self.max_length):
            chunk = encodings[i: i + self.max_length]
            if len(chunk) < self.max_length:
                pad_needed = self.max_length - len(chunk)
                chunk = chunk + [self.tokenizer.eos_token_id] * pad_needed
            sequences.append(chunk)
        return sequences


# --- TRAINING LOOP ---
def train_phase_1(epochs, dataloader, device, optimizer, criterion, model, save_dir):
    print("--- Phase 1 Training (Robust Mode) BEGIN ---")
    model.to(device)
    model.train()

    scaler = torch.amp.GradScaler("cuda")
    steps = 0
    total_loss = 0

    # Initialize gradients
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for i, batch in enumerate(dataloader):
            src_data = batch['input_ids'].to(device)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            with torch.autocast("cuda", dtype=torch.float16):
                output = model(src_data, decoder_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

                # Normalize loss for accumulation
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            current_loss_val = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss_val

            # Only update weights every N steps
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1

                # Print status
                if steps % 50 == 0:
                    avg = total_loss / (i + 1)
                    print(f"\rStep {steps} | Loss: {current_loss_val:.4f} | Avg: {avg:.4f}", end="", flush=True)

                # Checkpoint every 1000 UPDATES (not batches)
                if steps % 1000 == 0:
                    print(f"\n   💾 Saving Checkpoint at Step {steps}...")
                    os.makedirs(save_dir, exist_ok=True)
                    save_file(model.state_dict(), os.path.join(save_dir, "latest_checkpoint.safetensors"))
                    torch.cuda.empty_cache()

        print(f"\nEpoch {epoch + 1} Complete.")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    begin_time = datetime.now()

    # --- DATA FOLDERS ---
    # We include EVERYTHING now. The 'errors=replace' fix will handle the books.
    folders = [
        "../../../Datasets/Txt_Books/",
        "../../../Datasets/Au_Books/",
        "../../../Datasets/WikipediaDump/Final_Training_Data"
    ]

    tokenizer, vocab_size = load_gpt2_tokenizer()
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    dataset = RobustIterableDataset(folders, tokenizer, TOKENIZATION_MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Train for 1 full epoch (It will take time, but it's necessary)
    train_phase_1(1, dataloader, device, optimizer, criterion, model, "unsupervised_model_weights")

    print(f"Elapsed: {datetime.now() - begin_time}")