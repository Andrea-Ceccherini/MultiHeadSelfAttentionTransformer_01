from llama_cpp import Llama
import sys
import os

# --- CONFIGURATION ---
# Point this to where your converted model is
MODEL_PATH = "my_liver_model.gguf"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return

    try:
        print(f"Loading custom model from {MODEL_PATH}...")
        # Initialize the model
        # n_ctx=256 matches your training context length
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, # Try to use GPU
            n_ctx=256,       # Your custom model was trained with max_len=256
            verbose=False
        )
        print("✅ Model loaded successfully.\n")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    print("--- Liver QA Bot (Type 'exit' to quit) ---")
    print("Model trained on format: 'Question: ... Answer: ...'")

    while True:
        try:
            user_input = input("\nQuestion: ")
            if user_input.lower() in ["quit", "exit"]:
                break

            # Format the prompt EXACTLY how the model learned it in Phase 2
            # The model expects "Question: [Your Text] Answer:"
            prompt = f"Question: {user_input} Answer:"

            # Generate response
            # stop=["Question:"]: Tells the model to stop if it tries to ask a new question
            output = llm(
                prompt,
                max_tokens=64,
                stop=["Question:", "\n\n"],
                echo=False, # Set to True if you want to see the prompt too
                temperature=0.7
            )

            # Extract text
            response_text = output['choices'][0]['text'].strip()
            print(f"Answer: {response_text}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()