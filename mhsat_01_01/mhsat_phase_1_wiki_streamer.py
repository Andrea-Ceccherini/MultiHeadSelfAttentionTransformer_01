import os
import sys
import glob
from datetime import datetime

# --- SYSTEM CONFIGURATION & STABILITY FIXES ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Silence logs
import logging
import torch._inductor.config

torch._inductor.config.verbose_progress = False
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from safetensors.torch import save_file

# Import architecture
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, load_gpt2_tokenizer
)

# --- MODEL CONFIGURATION ---
TOKENIZATION_MAX_LENGTH = 256
NUM_LAYERS = 12
D_MODEL = 768
NUM_HEADS = 12
D_FF = 3072
DROPOUT = 0.2
BATCH_SIZE = 8  # Keep low for stability


# --- THE NEW SMART DATASET (Solves RAM & Truncation issues) ---
class WikiIterableDataset(IterableDataset):
    def __init__(self, folders, tokenizer, max_length):
        self.files = []
        for folder in folders:
            self.files.extend(sorted(glob.glob(os.path.join(folder, "*.txt"))))

        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Dataset initialized with {len(self.files)} files.")

    def __iter__(self):
        # Loop through files
        for idx, file_path in enumerate(self.files):

            # --- NEW CODE START ---
            # Print which file we are working on (Progress Bar)
            filename = os.path.basename(file_path)
            print(f"\n📂 [Progress] Opening file {idx + 1}/{len(self.files)}: {filename}")
            # --- NEW CODE END ---

            try:
                # errors='replace' fixes the "utf-8" crash you saw earlier
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:

                    buffer_text = ""
                    for line in f:
                        text = line.strip()
                        if not text: continue

                        buffer_text += " " + text

                        if len(buffer_text) > (self.max_length * 4):
                            tokenized_ids = self.process_text(buffer_text)
                            for seq in tokenized_ids:
                                yield {"input_ids": torch.tensor(seq, dtype=torch.long)}
                            buffer_text = ""

                    if buffer_text:
                        tokenized_ids = self.process_text(buffer_text)
                        for seq in tokenized_ids:
                            yield {"input_ids": torch.tensor(seq, dtype=torch.long)}

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

    def process_text(self, text):
        # (This method remains the same as before)
        encodings = self.tokenizer(text, truncation=False, padding=False, return_attention_mask=False)['input_ids']
        sequences = []
        for i in range(0, len(encodings), self.max_length):
            chunk = encodings[i: i + self.max_length]
            if len(chunk) < self.max_length:
                pad_needed = self.max_length - len(chunk)
                chunk = chunk + [self.tokenizer.eos_token_id] * pad_needed
            sequences.append(chunk)
        return sequences


def calculate_elapsed_time(start, end):
    return str(end - start)


def print_model_stats(model, vocab_size):
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = (total_params * 4) / (1024 * 1024)
    print("\n" + "=" * 50)
    print(f"📊 MODEL CONFIGURATION")
    print(f"Params: {total_params:,} | Size: {size_mb:.2f} MB")
    print("=" * 50 + "\n")


def model_training_unsupervised(epochs_, dataloader_, device_, optimizer_, criterion_, model_, model_save_dir_):
    print("model_training_unsupervised() - BEGIN")
    print("epochs_ =", epochs_)
    print("device_ =", device_)
    print("optimizer_ =", optimizer_)
    print("criterion_ =", criterion_)
    print("model_ =", type(model_))
    print("model_save_dir_ =", model_save_dir_)

    model_.to(device_)
    model_.train()

    scaler = torch.amp.GradScaler("cuda")

    # Checkpointing setup
    steps = 0
    i = 0
    save_every_n_steps = 1000  # Save every 1000 batches because 1 epoch is now HUGE

    for epoch in range(epochs_):
        print(f"\n--- Epoch {epoch + 1}/{epochs_} ---")
        train_loss = 0

        # Create the iterator
        dataloader_enum_ = enumerate(dataloader_)

        # NOTE: The "📂 [Progress] Opening file..." messages will appear
        # automatically inside this loop because the dataset triggers them.
        for i, batch in dataloader_enum_:
            src_data = batch['input_ids'].to(device_)

            # Create inputs and targets (Shifted right)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            optimizer_.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16):
                output = model_(src_data, decoder_input)
                loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer_)
            scaler.update()

            current_loss = loss.item()
            train_loss += current_loss
            steps += 1

            if i % 100 == 0:
                print(f"\rStep {i} | Loss: {current_loss:.4f}", end="", flush=True)

            # Save intermediate checkpoints
            if steps % save_every_n_steps == 0:
                print(f"\n   [Checkpoint] Saving at step {steps}...")
                os.makedirs(model_save_dir_, exist_ok=True)
                save_file(model_.state_dict(), os.path.join(model_save_dir_, "latest_checkpoint.safetensors"))
                torch.cuda.empty_cache()

        # Final stats for the epoch
        num_batches = i + 1
        if num_batches > 0:
            print(f"\nEpoch {epoch + 1} Finished. Avg Loss: {train_loss / num_batches:.4f}")
        else:
            print(f"\nEpoch {epoch + 1} Finished. No data processed.")

        torch.cuda.empty_cache()
    print("model_training_unsupervised() - END")


if __name__ == "__main__":
    print("MAIN - BEGIN")
    begin_time = datetime.now()

    # Point to your folders
    folders = [
        "../../../Datasets/WikipediaDump/Final_Training_Data_Partial",
        "../../../Datasets/Txt_Books/",
        "../../../Datasets/WikipediaData/",
        "../../../Datasets/Au_Books/"
        # You can add others back if you want, but Wiki is the big one
    ]

    # Load Tokenizer
    # Note: Ideally you should load your CUSTOM tokenizer trained in the previous step
    # But for now we use GPT2 tokenizer as per your imports to keep it simple.
    tokenizer, vocab_size = load_gpt2_tokenizer()
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = CustomTransformer(
        input_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=TOKENIZATION_MAX_LENGTH,
        dropout=DROPOUT
    )

    print_model_stats(model, vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Internal PyTorch Device: {device}")  # This will always say 'cuda' on GPU

    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Physical GPU Name: {gpu_name}")

        # Check for AMD specifically
        if "AMD" in gpu_name or "Radeon" in gpu_name:
            print("Backend: ROCm (AMD)")
        else:
            print("Backend: CUDA (NVIDIA)")
    else:
        print("Backend: CPU Only")

    # --- INITIALIZE THE STREAMING DATASET ---
    dataset = WikiIterableDataset(folders, tokenizer, TOKENIZATION_MAX_LENGTH)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Must be 0 for IterableDataset simplicity
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    epochs = 1 # Note: epochs set to 1 because passing through Wikipedia ONCE is already a lot of training!
    model_training_unsupervised(epochs, dataloader, device, optimizer, criterion, model, "unsupervised_model_weights")


    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")
    print("MAIN - END")