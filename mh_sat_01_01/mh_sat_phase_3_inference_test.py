import torch
import torch.nn.functional as F
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.pop("AMD_SERIALIZE_KERNEL", None)

from mh_sat_algorithms_for_custom_transformer_model import (
    CustomTransformer, load_gpt2_tokenizer,
    TOKENIZATION_MAX_LENGTH, NUM_LAYERS, D_MODEL, NUM_HEADS, D_FF, DROPOUT
)
from safetensors.torch import load_file

# --- POINT TO PHASE 1 MODEL ---
# MODEL_PATH = "unsupervised_model_weights/phase1_FINAL_3.77.safetensors"
MODEL_PATH = "../mh_sat_01_01/supervised_model_weights/fine_tuned_best.safetensors"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("MODEL_PATH =", MODEL_PATH)
print("DEVICE =", DEVICE)


def load_model():
    print(f"Loading Tokenizer...")
    tokenizer_, vocab_size = load_gpt2_tokenizer()

    print(f"Loading Model Architecture...")
    model_ = CustomTransformer(
        input_vocab_size=vocab_size, target_vocab_size=vocab_size,
        d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
        num_layers=NUM_LAYERS, max_len=TOKENIZATION_MAX_LENGTH, dropout=DROPOUT
    )

    print(f"Loading Weights from {MODEL_PATH}...")
    state_dict = load_file(MODEL_PATH)
    model_.load_state_dict(state_dict)
    model_.to(DEVICE)
    model_.eval()
    return model_, tokenizer_


def generate_text(model_, tokenizer_, prompt, max_new_tokens=60, temperature=0.5, repetition_penalty=1.2):
    # RAW PROMPT (No "Question:" wrapper)
    input_ids = tokenizer_.encode(prompt, return_tensors='pt').to(DEVICE)

    print(f"\n🔍 Input: '{prompt}'")

    for i in range(max_new_tokens):
        cond_ids = input_ids[:, -TOKENIZATION_MAX_LENGTH:]

        with torch.no_grad():
            outputs = model_(cond_ids, cond_ids)
            next_token_logits = outputs[:, -1, :]

            # --- REPETITION PENALTY (Critical Fix) ---
            # This penalizes tokens that have already appeared
            for token_id in set(input_ids[0].tolist()):
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty

            # Temperature
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample
            next_token_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # Stop at EOS
            if next_token_id.item() == tokenizer_.eos_token_id:
                break

    return tokenizer_.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("✅ Model Loaded.\n")

    # --- LIST OF TESTS ---
    tests = [
        # 1. Questions (Need formatting)
        {"text": "What is the function of the liver?", "type": "qa"},
        {"text": "What are the symptoms of hepatitis?", "type": "qa"},

        # 2. General Knowledge / Completions (Raw is okay, but QA format is safer for Phase 2 models)
        {"text": "The capital of France is", "type": "qa"},

        # 3. Storytelling (Raw format to test Phase 1 retention)
        {"text": "Once upon a time there was", "type": "raw"}
    ]

    for t in tests:
        raw_input = t["text"]

        # Format the prompt based on type
        if t["type"] == "qa":
            # Match the training format EXACTLY
            final_prompt = f"Question: {raw_input} Answer:"
        else:
            final_prompt = raw_input

        # Generate
        response = generate_text(model, tokenizer, final_prompt)

        # Clean up output for display (Remove the prompt part)
        # If the model echoes the prompt, we strip it to see just the new text
        if response.startswith(final_prompt):
            clean_response = response[len(final_prompt):].strip()
        else:
            clean_response = response

        print(f"🔍 Input: {final_prompt}")
        print(f"🤖 Output: {clean_response}")
        print("-" * 30)