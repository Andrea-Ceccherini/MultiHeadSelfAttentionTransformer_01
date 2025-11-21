import subprocess
import os
import sys
import json
from datetime import datetime

"""
This script defines 10 blocks of IDs, executes the first unfinished block, and updates the state file.
If the run fails, it restarts from the incomplete block.
Since we cannot know the exact IDs of the $18.7$ million securities without analyzing the dump, we define a reasonably large range
of IDs for the split. We use a chunking system and
save the state in a JSON file.

Initial Run: Run the script: python your_script.py.

The script will create the file wikiextractor_progress.json and run Block 1 (ID 0 to approximately 9 million).

Next Run (Continued): Run the script a second time.

The script will read the JSON file, see that Block 1 is completed: true, and automatically launch Block 2.

Failure to Launch (Recovery): If Block 3 fails, the exception is caught, but the JSON file is not updated.

On the next launch, the script will see that Block 3 is still completed: false and will retry executing Block 3
from the beginning.

This method ensures that, even in the event of an interruption, your expensive parsing process will resume exactly
from where the problem occurred.
"""



# --- Global Configuration ---
INPUT_FILE = r"../../../Datasets/WikipediaDump/enwiki-latest-pages-articles.xml.bz2"
OUTPUT_DIR = r"../../../Datasets/WikipediaDump/Extracted_Text_Corpus"
STATE_FILE = os.path.join(os.path.dirname(INPUT_FILE), "wikiextractor_progress.json")
NUM_CHUNKS = 10  # The number of parts into which to divide the work

# Estimate the maximum IDs of the English Wikipedia (they are in the tens of millions)
# Rough estimate of the maximum ID of the English Wikipedia (about 90 million for recent versions)
MAX_WIKI_ID = 90000000


# --- State Management Functions (Checkpoint) ---

def load_progress():
    """Load progress from a JSON file or create an initial status."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            print(f"Loading status from: {STATE_FILE}")
            return json.load(f)

    print("Creating initial state.")
    # Create work blocks (Chunks) based on ID.
    chunk_size = MAX_WIKI_ID // NUM_CHUNKS
    chunks = []

    for i in range(NUM_CHUNKS):
        min_id = i * chunk_size
        max_id = (i + 1) * chunk_size - 1
        if i == NUM_CHUNKS - 1:
            max_id = MAX_WIKI_ID  # Ensures that the last block covers the maximum ID

        chunks.append({
            "id": i + 1,
            "min_id": min_id,
            "max_id": max_id,
            "completed": False
        })

    return {"chunks": chunks}


def save_progress(state):
    """Save progress to JSON file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"State saved: {STATE_FILE}")


# --- Running WikiExtractor ---

def run_wikiextractor_chunked():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
        sys.exit(1)

    state = load_progress()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Find the next block to execute
    chunk_to_run = next((chunk for chunk in state["chunks"] if not chunk["completed"]), None)

    if not chunk_to_run:
        print("✅ All blocks have already been completed. Finished.")
        return

    chunk_id = chunk_to_run["id"]
    min_id = chunk_to_run["min_id"]
    max_id = chunk_to_run["max_id"]

    print(f"\n--- START Block {chunk_id}/{NUM_CHUNKS} ---")
    print(f"ID Range: fom {min_id} a {max_id}")

    # 2. Set the specific output for this block
    # Let's create a subfolder for each block to avoid confusion
    chunk_output_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}")
    os.makedirs(chunk_output_dir, exist_ok=True)

    start_time = datetime.now()

    # 3. Building the Command with Interrupt Parameters
    command = [
        "wikiextractor",
        INPUT_FILE,
        "--output", chunk_output_dir,
        "--min-id", str(min_id),  # Start from this ID
        "--max-id", str(max_id),  # Ends at this ID
        "--processes", str(os.cpu_count() or 1),
        "--bytes", "10M",
        "--quiet"
    ]

    try:
        # Execution
        subprocess.run(command, check=True, capture_output=False)

        # 4. Success: Update status as completed
        chunk_to_run["completed"] = True
        save_progress(state)

        end_time = datetime.now()
        elapsed_time = end_time - start_time

        print(f"\n✅ Block {chunk_id} successfully completed in {elapsed_time}.")
        print(f"The next launch will restart from the block {chunk_id + 1}.")

    except subprocess.CalledProcessError:
        print(f"\n❌ ERROR: Block {chunk_id} failed. The status was NOT updated.")
        print("On the next roll, it will try to execute the same block again.")
    except FileNotFoundError:
        print("\n❌ ERROR: Please make sure 'wikiextractor' is installed.")


if __name__ == "__main__":
    run_wikiextractor_chunked()