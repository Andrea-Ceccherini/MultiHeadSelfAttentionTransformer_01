import os
import sys
import glob
import csv
import random
from datetime import datetime

# --- SYSTEM CONFIGURATION ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Keep this allocator setting, it helps prevent Error 700
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file

from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    TOKENIZATION_MAX_LENGTH, CustomTransformer,
    create_model_configuration, NUM_LAYERS, D_MODEL,
    NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer
)

# --- CONFIGURATION (STABLE MODE) ---
BATCH_SIZE = 4  # REDUCED from 8 to 4 to prevent Error 700
ACCUMULATION_STEPS = 2  # ACCUMULATE gradients to simulate Batch Size 8
EPOCHS = 4
LEARNING_RATE = 1e-5
PATIENCE = 2


# --- CUSTOM BALANCED DATASET ---
class BalancedFineTuningDataset(Dataset):
    def __init__(self, liver_csv, wiki_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        # 1. Load Liver Data
        print("Loading Liver Data...")
        liver_count = 0
        try:
            with open(liver_csv, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        text = f"Question: {row[0]} Answer: {row[1]}"
                        self.samples.append(text)
                        liver_count += 1
        except FileNotFoundError:
            print(f"❌ Error: Liver CSV not found at {liver_csv}")
            sys.exit(1)

        print(f"   -> Found {liver_count} liver samples.")

        # 2. Load Wikipedia Data
        print("Loading and Slicing Wikipedia Data...")
        wiki_samples = []
        try:
            with open(wiki_file, 'r', encoding='utf-8', errors='replace') as f:
                buffer = ""
                for line in f:
                    line = line.strip()
                    if not line: continue
                    buffer += " " + line

                    if len(buffer.split()) > 50:
                        wiki_samples.append(buffer.strip())
                        buffer = ""
                        if len(wiki_samples) >= liver_count * 2:
                            break
        except FileNotFoundError:
            print(f"❌ Error: Wiki file not found at {wiki_file}")
            sys.exit(1)

        print(f"   -> Sliced {len(wiki_samples)} Wikipedia segments.")

        # Combine
        self.samples.extend(wiki_samples)
        random.shuffle(self.samples)
        print(f"Total Training Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {'input_ids': encodings['input_ids'].squeeze(0)}


def train_loop(epochs, train_dl, device, optimizer, criterion, model, save_dir):
    model.to(device)
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        steps = 0
        optimizer.zero_grad(set_to_none=True)  # Initialize gradients

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        for i, batch in enumerate(train_dl):
            src_data = batch['input_ids'].to(device)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            with torch.autocast("cuda", dtype=torch.float16):
                output = model(src_data, decoder_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))
                # Normalize loss for accumulation
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            # Step optimizer only after accumulating enough gradients
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Revert loss scaling for printing
            current_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss
            steps += 1

            # Print progress
            if steps % 50 == 0:
                print(f"\rStep {steps} | Loss: {current_loss:.4f}", end="")
                # --- STABILITY FIX: Clear cache periodically ---
                torch.cuda.empty_cache()

        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"\n   Avg Loss: {avg_loss:.4f}")

        # Save every epoch
        os.makedirs(save_dir, exist_ok=True)
        save_file(model.state_dict(), os.path.join(save_dir, "fine_tuned_best.safetensors"))

        # --- STABILITY FIX: Clear cache at end of epoch ---
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("MAIN (Balanced Fine-Tuning STABLE) - BEGIN")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- PATHS ---
    liver_csv = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    wiki_txt = "../../../Datasets/WikipediaDump/Final_Training_Data/train_chunk_001.txt"

    pretrained_path = os.path.join("unsupervised_model_weights", "latest_checkpoint.safetensors")
    save_dir = "supervised_qa_model_files"

    tokenizer, vocab_size = load_gpt2_tokenizer()

    # --- LOAD DATASET ---
    dataset = BalancedFineTuningDataset(liver_csv, wiki_txt, tokenizer, TOKENIZATION_MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # --- LOAD MODEL ---
    print("Loading Phase 1 Model...")
    model = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    if os.path.exists(pretrained_path):
        try:
            model.load_state_dict(load_file(pretrained_path))
            print("✅ Phase 1 Weights Loaded.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            exit()
    else:
        print(f"❌ Error: Pretrained weights not found at {pretrained_path}")
        exit()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- TRAIN ---
    train_loop(EPOCHS, dataloader, device, optimizer, criterion, model, save_dir)

    # Save Config
    create_model_configuration(save_dir, vocab_size, vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                               TOKENIZATION_MAX_LENGTH, DROPOUT)
    print("✅ Done. Now run the inference script.")