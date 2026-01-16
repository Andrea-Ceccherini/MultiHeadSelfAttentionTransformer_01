import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# --- FORCE CPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mh_sa_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_liver_tokenizer,
    TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
)

# --- PATHS ---
# Point to your new CPU-trained Phase 2 model
MODEL_PATH = "../mh_sa_custom_transformer_01_01/supervised_model_weights/fine_tuned_best.safetensors"
DEVICE = "cpu"


def load_model():
    print(f"Loading Tokenizer...")
    tokenizer_, vocab_size = load_liver_tokenizer()

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


def generate_text(model_, tokenizer_, prompt, max_new_tokens=100, temperature=0.4):
    # Prepare Prompt
    input_ids = tokenizer_.encode(prompt, return_tensors='pt').to(DEVICE)

    # Generate
    for i in range(max_new_tokens):
        cond_ids = input_ids[:, -TOKENIZATION_MAX_LENGTH:]

        with torch.no_grad():
            outputs = model_(cond_ids, cond_ids)
            next_token_logits = outputs[:, -1, :]

            # Repetition Penalty (Soft)
            for token_id in set(input_ids[0].tolist()):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= 1.2
                else:
                    next_token_logits[0, token_id] *= 1.2

            # Sampling
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # STOPPING CONDITION (The fix from Phase 2)
            if next_token_id.item() == tokenizer_.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # Secondary safety stop
            if "Question:" in tokenizer_.decode(input_ids[0])[len(prompt):]:
                break

    return tokenizer_.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("✅ Model Loaded. Running Tests...\n")

    tests = [
        # Liver Questions (The Specialist)
        "Question: What is the function of the liver? Answer:",
        "Question: What are the symptoms of hepatitis? Answer:",

        # General Knowledge (The Foundation)
        "Question: The capital of France is Answer:",

        # Raw Completion (Checking flexibility)
        "Once upon a time there was"
    ]

    for t in tests:
        # Generate
        response = generate_text(model, tokenizer, t)

        # Clean display
        clean_response = response.replace(t, "").strip()

        print(f"🔍 Input: {t}")
        print(f"🤖 Output: {clean_response}")
        print("-" * 30)