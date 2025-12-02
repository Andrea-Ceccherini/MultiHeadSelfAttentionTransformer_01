import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from safetensors.torch import load_file

# --- RDNA 4 OPTIMIZATION: Clean Environment ---
# Ensure the GPU is not forced into serial mode
os.environ.pop("AMD_SERIALIZE_KERNEL", None)
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Import functions from the library file
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

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- PATH CONFIGURATION ---
    # 1. Specific data (Target Domain)
    liver_dataset_path = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    dictionary_file_path = "../../../Datasets/English_Dictionary/english_dictionary_questions_and_answers_44.csv"

    # 2. General Data (Replay Buffer to Avoid Catastrophic Forgetting)
    general_data_path = "../../../Datasets/WikipediaData/*.txt"

    # 3. Model Paths
    model_save_dir = "supervised_qa_model_files"
    pre_trained_model_path = os.path.join("unsupervised_model_weights", "unsupervised_model_best.safetensors")

    # --- INITIALIZATION ---
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

    # --- WEIGHT LOADING PHASE 1 ---
    # IMPORTANT: Load weights BEFORE compiling
    if os.path.exists(pre_trained_model_path):
        print(f"Loading pre-trained weights: {pre_trained_model_path}")
        try:
            state_dict = load_file(pre_trained_model_path)
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights (mismatch?): {e}")
    else:
        print("WARNING: Pre-trained weights not found. Training from scratch (High risk of overfitting).")

    # --- RDNA 4 OPTIMIZATION: Compile Model ---
    print("Compiling model for RX 9070 XT (this may take ~60s)...")
    # This optimizes the specific graph for your fine-tuning run
    # model = torch.compile(model, mode="max-autotune")
    model = torch.compile(model, mode="default")

    # --- HYPERPARAMETERS ---
    # Increased Batch Size because FP16 (used in library) uses half the VRAM
    batch_size = 64
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

    # --- TRAINING SETUP ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- START TRAINING ---
    # Note: The mixed precision scaler is handled INSIDE this function now (in the library file)
    model_training(epochs, train_dl, val_dl, device, optimizer, criterion, model, model_save_dir, patience)

    # --- FINAL RESCUE ---
    # Note: When saving a compiled model, we usually want to save the original state dict.
    # save_file usually handles the state_dict() call which extracts weights from the compiled wrapper.
    save_model_weights(model, model_save_dir)
    create_model_configuration(model_save_dir, vocab_size, vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                               TOKENIZATION_MAX_LENGTH, DROPOUT)
    tokenizer.save_pretrained(model_save_dir)

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")

    print("MAIN (Fine-Tuning) - END")