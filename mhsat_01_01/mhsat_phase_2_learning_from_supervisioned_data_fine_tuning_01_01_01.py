import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from safetensors.torch import load_file

# Import functions from the library file (which we just updated with Data Mixing)
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

    # --- PATH CONFIGURATION ---
    # 1. Specific data (Target Domain)
    liver_dataset_path = "../../../Datasets/LiverDataset/liver_questions_and_answers_999.csv"
    dictionary_file_path = "../../../Datasets/English_Dictionary/english_dictionary_questions_and_answers_44.csv"
    
    # 2. General Data (Replay Buffer to Avoid Catastrophic Forgetting)
    # Points to the folder used in Phase 1. The *.txt asterisk is important for the glob.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device = ", device)

    # --- HYPERPARAMETERS ---
    batch_size = 32
    learning_rate = 2e-5
    epochs = 35
    patience = 5

    # --- DATA PREPARATION (Now with Data Mixing)) ---
    # We pass general_data_path to the updated function
    train_dl, val_dl, test_dl = get_dataloaders(
        liver_dataset_path, 
        dictionary_file_path, 
        general_data_path,  # <--- New parameter added
        tokenizer, 
        batch_size
    )
    
    # --- TRAINING SETUP ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # --- START TRAINING ---
    model_training(epochs, train_dl, val_dl, device, optimizer, criterion, model, model_save_dir, patience)

    # --- FINAL RESCUE ---
    save_model_weights(model, model_save_dir)
    create_model_configuration(model_save_dir, vocab_size, vocab_size, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, TOKENIZATION_MAX_LENGTH, DROPOUT)
    tokenizer.save_pretrained(model_save_dir)

    print(f"Elapsed: {calculate_elapsed_time(begin_time, datetime.now())}")

    print("MAIN (Fine-Tuning) - END")