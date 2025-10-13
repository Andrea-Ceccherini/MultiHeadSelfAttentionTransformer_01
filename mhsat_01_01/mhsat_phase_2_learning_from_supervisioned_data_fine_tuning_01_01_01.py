import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from safetensors.torch import load_file

from mhsat_01_01.mhsat_algorithms_for_custom_transformer_model_01_01_01 import TOKENIZATION_MAX_LENGTH, \
    CustomTransformer, get_dataloaders, model_training, save_model_weights, create_model_configuration, NUM_LAYERS, \
    D_MODEL, NUM_HEADS, D_FF, DROPOUT, load_gpt2_tokenizer

def calculate_elapsed_time(begin_time_, end_time_):
    elapsed_time = end_time_ - begin_time_
    days = elapsed_time.days
    seconds = elapsed_time.seconds
    milliseconds = elapsed_time.microseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)
    formatted_elapsed_time = f"{years:04}:{months:02}:{days:02}:{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"
    return formatted_elapsed_time

if __name__ == "__main__":
    print("__main__() - BEGIN")
    begin_time_ = datetime.now()

    # Paths and Hyperparameters
    #dataset_file_path = "../../../Datasets/LiverDataset/liver_questions_and_answers_14931.csv"
    dataset_file_path = "../../../Datasets/LiverDataset/liver_questions_and_answers_499.csv"

    if not os.path.exists(dataset_file_path):
        print(f"__main__() - Error: The file '{dataset_file_path}' was not found.")
        exit()

    model_save_dir = "supervised_qa_model_files"
    pre_trained_model_path = os.path.join("unsupervised_model_weights", "unsupervised_model_best.safetensors")

    # Load the tokenizer

    tokenizer, tokenizer_len = load_gpt2_tokenizer()
    #tokenizer_len = len(tokenizer)
    print(f"__main__() - tokenizer_len = {tokenizer_len}")


    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # # FIX: Add the same special token to match the pre-trained model's vocabulary.
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Define hyperparameters to match the pre-trained model
    d_model = D_MODEL
    num_heads = NUM_HEADS
    d_ff = D_FF
    num_layers = NUM_LAYERS
    max_len = TOKENIZATION_MAX_LENGTH
    dropout = DROPOUT
    input_vocab_size = tokenizer_len
    target_vocab_size = tokenizer_len

    print(f"__main__() - d_model: {d_model}")
    print(f"__main__() - num_heads: {num_heads}")
    print(f"__main__() - d_ff: {d_ff}")
    print(f"__main__() - num_layers: {num_layers}")
    print(f"__main__() - max_len: {max_len}")
    print(f"__main__() - dropout: {dropout}")
    print(f"__main__() - input_vocab_size: {input_vocab_size}")
    print(f"__main__() - target_vocab_size: {target_vocab_size}")

    # Istanzia il modello
    model = CustomTransformer(
        input_vocab_size=input_vocab_size,
        target_vocab_size=target_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout
    )

    # --- Print the number of model parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n__main__() - Model contains {total_params:,} parameters.")
    # -------------------------------------------

    # --- CRUCIAL STEP: LOAD PRE-TRAINED WEIGHTS ---
    if os.path.exists(pre_trained_model_path):
        print(f"__main__() - Loading pre-trained weights from: {pre_trained_model_path}")
        state_dict = load_file(pre_trained_model_path)
        model.load_state_dict(state_dict)
    else:
        print(f"__main__() - Warning: Pre-trained weights not found at '{pre_trained_model_path}'. Training from scratch.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Hyperparameters for fine-tuning
    batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.001
    epochs = 35
    patience = 5


    # Prepare data
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset_file_path, tokenizer, batch_size)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Begin the supervised training
    model_training(
        epochs=epochs,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        model=model,
        model_save_dir=model_save_dir,
        patience=patience
    )

    # After training, load the best model saved by early stopping for inference or final evaluation
    best_model_path = os.path.join(model_save_dir, "model_best.safetensors")
    if os.path.exists(best_model_path):
        print(f"\n__main__() - Loading best model weights from '{best_model_path}' for final saving/evaluation.")
        best_state_dict = load_file(best_model_path)
        model.load_state_dict(best_state_dict)
        model.eval()
    else:
        print(f"\n__main__() - No best model found at '{best_model_path}'. Using the last trained model state.")

    # Save final model state and configuration
    save_model_weights(model, model_save_dir)
    create_model_configuration(
        model_save_dir_=model_save_dir,
        input_vocab_size_=len(tokenizer),
        target_vocab_size_=len(tokenizer),
        d_model_=d_model,
        num_heads_=num_heads,
        d_ff_=d_ff,
        num_layers_=num_layers,
        max_len_=max_len,
        dropout_=dropout
    )

    # Save the tokenizer
    print("__main__() - Saving model and tokenizer to the specified directory - STARTED")
    tokenizer.save_pretrained(model_save_dir)
    print(
        f"__main__() - merges.txt, special_tokens_map.json, tokenizer.json, tokenizer_config.json, vocab.json saved to {model_save_dir}")
    print("__main__() - Saving model and tokenizer to the specified directory - COMPLETED")

    end_time_ = datetime.now()
    elapsed_ = calculate_elapsed_time(begin_time_, end_time_)
    print("__main__() - Elapsed Time =", elapsed_)
    print("__main__() - END")
