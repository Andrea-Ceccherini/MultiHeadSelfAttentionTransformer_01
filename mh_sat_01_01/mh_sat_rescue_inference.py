import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mh_sat_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_gpt2_tokenizer,
    TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
)

# Point to your Phase 2 model
MODEL_PATH = "supervised_model_weights/fine_tuned_best.safetensors"
DEVICE = "cpu"


def load_model():
    print(f"Loading Tokenizer...")
    tokenizer_, vocab_size = load_gpt2_tokenizer()

    print(f"Loading Model Architecture...")
    model_ = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit()

    print(f"Loading Weights from {MODEL_PATH}...")
    state_dict = load_file(MODEL_PATH)
    model_.load_state_dict(state_dict)
    model_.to(DEVICE)
    model_.eval()
    return model_, tokenizer_


def generate_strict(model_, tokenizer_, prompt, max_new_tokens=100):
    input_ids = tokenizer_.encode(prompt, return_tensors='pt').to(DEVICE)

    print(f"\n🔍 Input: '{prompt}'")

    for i in range(max_new_tokens):
        cond_ids = input_ids[:, -TOKENIZATION_MAX_LENGTH:]

        with torch.no_grad():
            outputs = model_(cond_ids, cond_ids)
            next_token_logits = outputs[:, -1, :]

            # --- STRICT GREEDY DECODING (Temp 0 equivalent) ---
            # We just pick the token with the highest score. No randomness.
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Check EOS
            if next_token_id.item() == tokenizer_.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # Stop if it generates "Question:" again
            if "Question:" in tokenizer_.decode(input_ids[0])[len(prompt):]:
                break

    return tokenizer_.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("✅ Model Loaded.\n")

    tests = [
        "Question: What is the function of the liver? Answer:",
        "Question: The capital of France is Answer:",
    ]

    for t in tests:
        response = generate_strict(model, tokenizer, t)
        clean_response = response.replace(t, "").strip()
        print(f"🤖 Output: {clean_response}")
        print("-" * 30)