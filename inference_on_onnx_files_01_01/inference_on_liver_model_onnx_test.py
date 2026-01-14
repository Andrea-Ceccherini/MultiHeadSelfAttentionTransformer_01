import onnxruntime as ort
import numpy as np
import os
import sys

# Standard library for tokenizers (pip install tokenizers)
# We use this instead of your custom function to simulate a real user environment
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# --- CONFIGURATION ---
MODEL_FILE = "../mh_sat_01_01/onnx_model_distribution_format/my_liver_model.onnx"
# We need the tokenizer.json file. It should be in your supervised_model_weights folder
# or hf_model_export folder. Adjust path if needed.
TOKENIZER_FILE = "../mh_sat_01_01/onnx_model_distribution_format/tokenizer.json"
TOKENIZER_DIR = "../mh_sat_01_01/onnx_model_distribution_format/"
TOKENIZATION_MAX_LENGTH = 256


def run_inference():
    # 1. Verify Files
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: Model not found at {MODEL_FILE}")
        return

    # Fallback to load your custom tokenizer if json isn't exported yet
    try:
        # tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    except:
        print("⚠️ tokenizer.json not found, loading via custom function...")
        from mh_sat_01_01.mh_sat_algorithms_for_custom_transformer_model import load_gpt2_tokenizer
        tokenizer_obj, _ = load_gpt2_tokenizer()

        # Create a wrapper to make it look like the standard Tokenizer for this script
        class Wrapper:
            def encode(self, text):
                return tokenizer_obj.encode(text)

            def decode(self, ids):
                return tokenizer_obj.decode(ids)

            @property
            def eos_token_id(self):
                return tokenizer_obj.eos_token_id

            @property
            def pad_token_id(self):
                return tokenizer_obj.eos_token_id

        tokenizer = Wrapper()

    print("Loading ONNX Session (This runs on CPU)...")
    # This automatically loads the .data file too
    session = ort.InferenceSession(MODEL_FILE)

    # 2. Define Tests
    tests = [
        "Question: What is the function of the liver? Answer:",
        "Question: The capital of France is Answer:"
    ]

    for question in tests:
        print(f"\n🔍 Input: {question}")
        print("🤖 Generating...", end="", flush=True)

        # 3. Encode
        # Note: Depending on which tokenizer object we got, api might differ slightly.
        # Assuming HuggingFace/GPT2 tokenizer style:
        if hasattr(tokenizer, 'encode') and hasattr(tokenizer.encode(question), 'ids'):
            input_ids = tokenizer.encode(question).ids  # Tokenizers library
        else:
            input_ids = tokenizer.encode(question)  # Transformers library

        # 4. Generation Loop (Greedy)
        current_ids = list(input_ids)

        for _ in range(50):
            # Prepare Inputs
            # Pad to fixed length (ONNX usually expects fixed shapes unless dynamic axes worked perfectly)
            padding_len = TOKENIZATION_MAX_LENGTH - len(current_ids)
            if padding_len < 0: break

            # Use EOS as pad
            pad_id = tokenizer.eos_token_id
            # pad_id = tokenizer.token_to_id("<|endoftext|>")
            # if pad_id is None:
            #     pad_id = 50256

            padded_input = current_ids + [pad_id] * padding_len

            # Create ONNX Inputs (Batch Size 1)
            input_tensor = np.array([padded_input], dtype=np.int64)

            # Run Math
            outputs = session.run(
                None,
                {
                    "encoder_input": input_tensor,
                    "decoder_input": input_tensor
                }
            )

            # Get next token
            logits = outputs[0]  # Shape: [1, 256, 50258]
            last_token_logits = logits[0, len(current_ids) - 1, :]
            next_token_id = np.argmax(last_token_logits)

            # Stop condition
            if next_token_id == tokenizer.eos_token_id:
                break

            current_ids.append(next_token_id)
            print(".", end="", flush=True)

        # 5. Decode
        if hasattr(tokenizer, 'decode'):
            # Handle list vs tensor nuances
            output_text = tokenizer.decode(current_ids)
        else:
            output_text = tokenizer.decode(current_ids)

        print(f"\n💡 Result: {output_text}")


if __name__ == "__main__":
    run_inference()