import os
import sys
from datetime import datetime

# --- STABILITY & MEMORY FIXES ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file

# Import from library (Make sure you updated the Global Parameters there!)
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import (
    TOKENIZATION_MAX_LENGTH, CustomTransformer, get_dataloaders, model_training,
    save_model_weights, create_model_configuration, NUM_LAYERS, D_MODEL,
    NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer
)


def calculate_elapsed_time(start, end):
    return str(end - start)


if __name__ == "__main__":
    print("MAIN (Fine-Tuning with Replay Buffer) - BEGIN")
    begin_time = datetime.now()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- PATH CONFIGURATION ---
    liver_dataset_path = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    dictionary_file_path = "../../../Datasets/English_Dictionary/english_dictionary_questions_and_answers_44.csv"
    general_data_path = "../../../Datasets/WikipediaData/*.txt"

    model_save_dir = "supervised_qa_model_files"
    pre_trained_model_path = os.path.join("unsupervised_model_weights", "unsupervised_model_best.safetensors")

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
        print(f"Loading pre-trained weights from: {pre_trained_model_path}")
        try:
            state_dict = load_file(pre_trained_model_path)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR loading weights: {e}")
            print("Did you update the library file parameters (NUM_LAYERS, D_MODEL) to match Phase 1?")
            exit()
    else:
        print("WARNING: Pre-trained weights not found!")

    # --- STABILITY FIX: DISABLE COMPILATION ---
    print("✅ Running in Native Eager Mode (Stable)...")
    # model = torch.compile(model) # DISABLED for stability on 314M model

    # --- HYPERPARAMETERS ---
    # Batch Size 8 is crucial for 12 Layers on 16GB VRAM
    batch_size = 8
    learning_rate = 2e-5
    epochs = 35
    patience = 5

    # --- DATA PREPARATION ---
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
    # Note: Ensure model_training in the library also has torch.cuda.empty_cache() added if possible,
    # otherwise we rely on the small batch size to stay safe.
    model_training(epochs, train_dl, val_dl, device, optimizer, criterion, model, model_save_dir, patience)

    # --- SAVE ---
    save_model_weights(model, model_save_dir)
    create_model_configuration(model_save_dir, vocab_size, vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                               TOKENIZATION_MAX_LENGTH, DROPOUT)
    tokenizer.save_pretrained(model_save_dir)

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")
    print("MAIN (Fine-Tuning) - END")