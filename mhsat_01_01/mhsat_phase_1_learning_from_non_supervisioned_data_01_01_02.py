import os

# --- FIX 1: Prevent Tokenizer Deadlocks ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- RDNA 4 OPTIMIZATION: Clean Environment ---
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
from torch.utils.data import DataLoader, Dataset

import torch._inductor.config
import logging
# SILENCE TRITON/INDUCTOR LOGS
torch._inductor.config.verbose_progress = False
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


from safetensors.torch import save_file
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    CustomTransformer, NUM_LAYERS, TOKENIZATION_MAX_LENGTH,
    D_MODEL, NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer
)
from datetime import datetime


class UnsupervisedTextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def calculate_elapsed_time(start, end):
    return str(end - start)


def model_training_unsupervised(epochs, dataloader_, device_, optimizer_, criterion_, model_, model_save_dir_,
                                patience_=3):
    print("Unsupervised Training - BEGIN")
    model_.to(device_)

    print("Compiling model for RX 9070 XT (Default mode)...")
    # 'default' is much more stable than 'max-autotune' and compiles faster
    # It still provides good speedups for RDNA 4
    model_ = torch.compile(model_)

    model_.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        train_loss = 0
        model_.train()

        print(f"Epoch {epoch + 1}/{epochs} ", end="")

        # DataLoader iteration starts here
        for i, batch in enumerate(dataloader_):
            src_data = batch['input_ids'].to(device_)

            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            optimizer_.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16):
                output = model_(src_data, decoder_input)
                loss = criterion_(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer_)
            scaler.update()

            train_loss += loss.item()
            if i % 50 == 0: print(".", end="", flush=True)

        avg_loss = train_loss / len(dataloader_)
        print(f" Loss: {avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            epochs_no_improve = 0
            os.makedirs(model_save_dir_, exist_ok=True)
            # Save original state dict (compatible with future loads)
            # Accessing ._orig_mod if compiled, or fallback
            if hasattr(model_, "_orig_mod"):
                save_file(model_._orig_mod.state_dict(),
                          os.path.join(model_save_dir_, "unsupervised_model_best.safetensors"))
            else:
                save_file(model_.state_dict(), os.path.join(model_save_dir_, "unsupervised_model_best.safetensors"))
            print("Saved Best Model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_:
                print("Early Stopping.")
                break


def load_text_data_from_folder(folder_path, data_list):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return 0
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding='latin-1') as f:
                    text = f.read().strip()
                    if len(text) > 10:
                        data_list.append(text)
                        count += 1
            except Exception:
                pass
    return count


if __name__ == "__main__":
    print("MAIN - BEGIN")
    begin_time = datetime.now()

    folders = [
        "../../../Datasets/Txt_Books/",
        "../../../Datasets/WikipediaData/",
        "../../../Datasets/Au_Books/"
    ]

    tokenizer, vocab_size = load_gpt2_tokenizer()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    raw_text = []
    for folder in folders:
        c = load_text_data_from_folder(folder, raw_text)
        print(f"Loaded {c} docs from {folder}")

    if not raw_text:
        print("No data found. Creating dummy data.")
        raw_text = ["This is a test sentence for the GPU." for _ in range(1000)]

    print(f"Tokenizing {len(raw_text)} documents...")
    tokenized_data = tokenizer(
        raw_text, return_tensors='pt', padding='max_length', truncation=True, max_length=TOKENIZATION_MAX_LENGTH
    )

    dataset = UnsupervisedTextDataset(tokenized_data)

    # --- FIX 2: Set num_workers=0 to prevent crash ---
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # Must be 0 to avoid fork issues with torch.compile
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    model_training_unsupervised(5, dataloader, device, optimizer, criterion, model, "unsupervised_model_weights")

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")

    print("MAIN - END")