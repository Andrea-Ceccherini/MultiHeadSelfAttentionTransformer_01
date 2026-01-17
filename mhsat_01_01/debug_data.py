import numpy as np
from mhsat_algorithms_for_custom_transformer_model_01_01_01 import load_gpt2_tokenizer

DATA_PATH = "../mh_sat_01_01/wiki_books_dataset/wiki_books_dataset.bin"

def inspect_data():
    if not os.path.exists(DATA_PATH):
        print("File not found.")
        return

    print(f"Loading {DATA_PATH}...")
    # Load as uint16 (how we saved it)
    data = np.memmap(DATA_PATH, dtype=np.uint16, mode='r')
    print(f"Total tokens: {len(data):,}")

    tokenizer, _ = load_gpt2_tokenizer()

    print("\n--- CHECK 1: The First 100 Tokens ---")
    chunk1 = data[:100].astype(np.int64)
    print("Raw IDs:", chunk1[:20]) # Print first 20 numbers
    text1 = tokenizer.decode(chunk1)
    print(f"Decoded Text:\n{text1}")

    print("\n--- CHECK 2: Random Middle Chunk ---")
    # Pick a random spot in the middle
    mid = len(data) // 2
    chunk2 = data[mid : mid+100].astype(np.int64)
    text2 = tokenizer.decode(chunk2)
    print(f"Decoded Text:\n{text2}")

    print("\n--- CHECK 3: Zeros check ---")
    # Check if we have too many zeros (padding)
    zeros = np.sum(data[:10000] == 0)
    print(f"Zeros in first 10,000 tokens: {zeros}")

if __name__ == "__main__":
    import os
    inspect_data()