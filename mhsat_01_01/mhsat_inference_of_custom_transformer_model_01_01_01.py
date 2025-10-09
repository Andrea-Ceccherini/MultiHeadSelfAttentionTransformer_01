
import os
import json
from safetensors.torch import load_file
from transformers import AutoTokenizer

from mhsat_01_01.mhsat_algorithms_for_custom_transformer_model_01_01_01 import \
    generate_text_with_beam, CustomTransformer

def create_tokenizer(model_directory_path_):
    """
    Creates and returns a tokenizer from the specified directory.
    Handles adding the padding token to ensure consistency.
    """
    tokenizer_ = AutoTokenizer.from_pretrained(model_directory_path_)
    if tokenizer_.pad_token is None:
        tokenizer_.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer_


if __name__ == "__main__":
    print("__main__() - BEGIN")
    # FIX: Use the correct directory path from the supervised training script
    model_directory_path = "supervised_qa_model_files"

    if not os.path.exists(model_directory_path):
        print(f"__main__() - Error: The model directory '{model_directory_path}' was not found. Please ensure your fine-tuning script has been run and the files are saved correctly.")
        exit()

    tokenizer = create_tokenizer(model_directory_path)

    print(f"\n__main__() - Loading config.json to get hyperparameters - BEGIN")
    config_path = os.path.join(model_directory_path, "config.json")
    if not os.path.exists(config_path):
        print(f"__main__() - Error: config.json not found in '{model_directory_path}'. Cannot load model architecture. ❌")
        exit()

    with open(config_path, "r") as f:
        model_config = json.load(f)
    print(f"__main__() - model_config = {model_config}")
    print(f"__main__() - Loading config.json to get hyperparameters - END\n")

    print("\n__main__() - Instantiating model architecture from config.json - BEGIN")
    input_vocab_size = model_config["input_vocab_size"]
    print(f"__main__() - input_vocab_size: {input_vocab_size}")

    dropout = model_config.get("dropout", 0.1)
    print(f"__main__() - dropout: {dropout}")
    try:
        model = CustomTransformer(
            input_vocab_size=input_vocab_size,
            target_vocab_size=model_config["target_vocab_size"],
            d_model=model_config["d_model"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            num_layers=model_config["num_layers"],
            max_len=model_config["max_len"],
            dropout=dropout
        )
        print("__main__() - Model architecture instantiated.")
    except KeyError as e:
        print(f"__main__() - Error: Missing hyperparameter '{e}' in config.json. Please check your config file.")
        exit()
    except Exception as e:
        print(f"__main__() - Error instantiating model: {e}")
        exit()
    print("__main__() - Instantiating model architecture from config.json - END\n")

    print("\n__main__() - Getting Weights from file 'model_best.safetensors' and load them into the model - BEGIN")
    # Load the best model weights
    model_weights_path = os.path.join(model_directory_path, "model_best.safetensors")
    if not os.path.exists(model_weights_path):
        print(f"__main__() - Error: model_best.safetensors not found in '{model_directory_path}'. Cannot load weights. ❌")
        exit()

    try:
        state_dict = load_file(model_weights_path)
        model.load_state_dict(state_dict)
        model.eval()
        print("__main__() - Model weights loaded successfully and model set to evaluation mode.")
    except Exception as e:
        print(f"__main__() - Error loading model weights: {e}")
        exit()
    print("\n__main__() - Getting Weights from file 'model_best.safetensors' and load them into the model - END\n")

    print("\n__main__() - Performing Inference - BEGIN")
    question = ""
    print(f"__main__() - question: '{question}'")

    temperature = 0.8
    max_output_length = 50
    beam_width = 3

    while True:
        user_input = input("__main__() - Enter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("__main__() Exiting the inference loop.")
            break
        question = user_input
        answer = generate_text_with_beam(model, tokenizer, question, max_output_length, beam_width, temperature)
        print(f"__main__() - answer = '{answer}'")

    # FIX: Moved this print statement outside the loop.
    print("\n__main__() - Performing Inference - END\n")

    print("__main__() - END")
