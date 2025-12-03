import os
import glob
from tqdm import tqdm  # Install with: pip install tqdm

# --- Configuration ---
# Where the messy WikiExtractor folders are
INPUT_DIR = r"../../../Datasets/WikipediaDump/Extracted_Text_Corpus"

# Where you want the final clean .txt files
TRAIN_DIR = r"../../../Datasets/WikipediaDump/Final_Training_Data"


def clean_and_consolidate():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Error: Directory {INPUT_DIR} does not exist.")
        return

    os.makedirs(TRAIN_DIR, exist_ok=True)

    # 1. Identify all chunk folders (e.g., chunk_001, chunk_002)
    chunk_folders = sorted([f for f in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, f))])

    print(f"Found {len(chunk_folders)} chunk folders to process.")

    # 2. Process each chunk folder
    for folder_name in tqdm(chunk_folders, desc="Consolidating Chunks"):

        source_chunk_path = os.path.join(INPUT_DIR, folder_name)

        # Define output filename: "train_chunk_001.txt"
        output_filename = f"train_{folder_name}.txt"
        output_file_path = os.path.join(TRAIN_DIR, output_filename)

        # Skip if already exists (resume capability)
        if os.path.exists(output_file_path):
            continue

        # Open the single output file for this chunk
        with open(output_file_path, 'w', encoding='utf-8') as outfile:

            # Walk through subfolders (AA, AB, etc.) inside the chunk
            for root, dirs, files in os.walk(source_chunk_path):
                for file in files:
                    # WikiExtractor produces files named 'wiki_00', 'wiki_01', etc.
                    if file.startswith("wiki_"):
                        file_path = os.path.join(root, file)

                        # Read and clean the content
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                # REMOVE <doc> TAGS
                                if line.startswith('<doc') or line.startswith('</doc>'):
                                    continue

                                # Write clean text
                                # Optional: Filter out empty lines if you want tight text
                                if line.strip():
                                    outfile.write(line)

        # (Optional) If you want to delete the messy source folder to save space:
        # import shutil
        # shutil.rmtree(source_chunk_path)

    print(f"\n✅ Done! Clean .txt files are in: {TRAIN_DIR}")


if __name__ == "__main__":
    clean_and_consolidate()