import os
import sys
import glob
from datetime import datetime

# --- STABILITY & MEMORY FIXES (CRITICAL FOR AMD) ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file

# Import architecture config from library
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    TOKENIZATION_MAX_LENGTH, CustomTransformer, get_dataloaders,
    create_model_configuration, NUM_LAYERS, D_MODEL,
    NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer
)


def calculate_elapsed_time(start, end):
    return str(end - start)


# --- LOCAL STABLE TRAINING LOOP ---
# We define this HERE to ensure it uses Mixed Precision and empty_cache
def model_training_fine_tuning(epochs, train_dl, val_dl, device, optimizer, criterion, model, save_dir, patience):
    print("\n--- Fine-Tuning Started ---")
    model.to(device)

    scaler = torch.amp.GradScaler("cuda")
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Training Phase
        for batch in train_dl:
            src_data = batch['input_ids'].to(device)
            decoder_input = src_data[:, :-1]
            labels = src_data[:, 1:]

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16):
                output = model(src_data, decoder_input)
                loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            steps += 1

            if steps % 10 == 0:
                print(f"\r   Batch {steps} | Train Loss: {loss.item():.4f}", end="", flush=True)

        avg_train_loss = train_loss / steps if steps > 0 else 0
        print(f"\n   Avg Train Loss: {avg_train_loss:.4f}")

        # Validation Phase
        model.eval()
        val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in val_dl:
                src_data = batch['input_ids'].to(device)
                decoder_input = src_data[:, :-1]
                labels = src_data[:, 1:]

                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(src_data, decoder_input)
                    loss = criterion(output.reshape(-1, output.shape[-1]), labels.reshape(-1))

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
        print(f"   Avg Val Loss:   {avg_val_loss:.4f}")

        # Checkpointing & Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs(save_dir, exist_ok=True)
            save_file(model.state_dict(), os.path.join(save_dir, "fine_tuned_best.safetensors"))
            print("   ✅ Saved New Best Model")
        else:
            epochs_no_improve += 1
            print(f"   ⚠️ No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print("   ⏹️ Early Stopping Triggered.")
                break

        torch.cuda.empty_cache()


if __name__ == "__main__":
    print("MAIN (Fine-Tuning) - BEGIN")
    begin_time = datetime.now()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- PATH CONFIGURATION ---
    liver_dataset_path = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    dictionary_file_path = "../../../Datasets/English_Dictionary/english_dictionary_questions_and_answers_44.csv"

    # UPDATED: Point to one of your clean chunks for the replay buffer
    # If this path doesn't exist, change it to "../../../Datasets/WikipediaDump/Final_Training_Data/*.txt"
    general_data_path = "../../../Datasets/WikipediaDump/Final_Training_Data/train_chunk_001.txt"

    model_save_dir = "supervised_qa_model_files"

    # UPDATED: Points to the file created by Phase 1
    pre_trained_model_path = os.path.join("unsupervised_model_weights", "latest_checkpoint.safetensors")

    # --- INITIALIZATION ---
    tokenizer, vocab_size = load_gpt2_tokenizer()

    print(f"Initializing Model: {NUM_LAYERS} Layers, {D_MODEL} Dim")
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

    # --- WEIGHT LOADING ---
    if os.path.exists(pre_trained_model_path):
        print(f"Loading Phase 1 weights from: {pre_trained_model_path}")
        try:
            state_dict = load_file(pre_trained_model_path)
            model.load_state_dict(state_dict)
            print("✅ Weights loaded successfully.")
        except Exception as e:
            print(f"❌ FATAL ERROR loading weights: {e}")
            print("Ensure NUM_LAYERS/D_MODEL in library match the Phase 1 configuration.")
            exit(1)
    else:
        print(f"❌ ERROR: Pre-trained file not found at {pre_trained_model_path}")
        print("Please finish Phase 1 training first.")
        exit(1)

    # --- HYPERPARAMETERS ---
    batch_size = 8
    learning_rate = 2e-5
    epochs = 35
    patience = 5

    print(f"Loading datasets (Replay Buffer: {os.path.basename(general_data_path)})...")

    # NOTE: Ensure get_dataloaders in your library handles the paths correctly.
    # If general_data_path is a single file, your get_dataloaders needs to support that.
    train_dl, val_dl, test_dl = get_dataloaders(
        liver_dataset_path,
        dictionary_file_path,
        general_data_path,
        tokenizer,
        batch_size
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- START TRAINING ---
    model_training_fine_tuning(
        epochs, train_dl, val_dl, device, optimizer, criterion, model, model_save_dir, patience
    )

    # --- SAVE FINAL CONFIG ---
    create_model_configuration(model_save_dir, vocab_size, vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                               TOKENIZATION_MAX_LENGTH, DROPOUT)
    tokenizer.save_pretrained(model_save_dir)

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")
    print("MAIN (Fine-Tuning) - END")