"""
/home/andrea/Large_Language_Models/mistral-7B-Instruct-v0_3-Q8_0_GGUF/Mistral-7B-Instruct-v0.3-Q8_0.gguf

"""

from llama_cpp import Llama
import sys

# Path to your specific model file
MODEL_PATH = "/home/andrea/Large_Language_Models/mistral-7B-Instruct-v0_3-Q8_0_GGUF/Mistral-7B-Instruct-v0.3-Q8_0.gguf"


def main():
    try:
        # Initialize the model
        # n_gpu_layers=-1 attempts to offload all layers to GPU. Set to 0 for CPU only.
        # n_ctx=4096 sets the context window size. Mistral supports up to 32k, but higher values use more RAM.
        print(f"Loading model from {MODEL_PATH}...")
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False  # Set to True to see detailed loading logs
        )
        print("Model loaded successfully.\n")

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("--- Mistral 7B Instruct Chat (Type 'quit' to exit) ---")

    # Interactive chat loop
    history = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break

            # Add user message to chat history
            history.append({"role": "user", "content": user_input})

            # Create chat completion
            # stream=True allows printing the token as they are generated
            stream = llm.create_chat_completion(
                messages=history,
                max_tokens=1024,
                temperature=0.7,
                stream=True
            )

            print("Bot: ", end="", flush=True)

            response_text = ""
            for chunk in stream:
                if 'content' in chunk['choices'][0]['delta']:
                    text_chunk = chunk['choices'][0]['delta']['content']
                    print(text_chunk, end="", flush=True)
                    response_text += text_chunk

            print()  # Newline after response

            # Add assistant response to history to maintain context
            history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()