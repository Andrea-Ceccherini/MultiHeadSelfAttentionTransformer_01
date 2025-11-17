import subprocess
import os
import sys
from datetime import datetime

# --- Configurazione dei Percorsi ---
# Percorso del file di input (dump compresso)
INPUT_FILE = r"C:\Users\andre\Datasets\WikipediaDump\enwiki-latest-pages-articles.xml.bz2"

# Cartella di output dove verranno salvati i file di testo pulito
# (Verrà creata automaticamente se non esiste)
OUTPUT_DIR = r"C:\Users\andre\Datasets\WikipediaDump\Extracted_Text_Corpus"


# --- Funzione Principale di Esecuzione ---

def run_wikiextractor():
    """
    Esegue WikiExtractor utilizzando il modulo subprocess per pulire il dump.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERRORE: File di input non trovato: {INPUT_FILE}")
        print("Assicurati che il percorso sia corretto e che il file sia presente.")
        sys.exit(1)

    # 1. Preparazione dell'Output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Avvio WikiExtractor ---\n")
    print(f"File di Input: {INPUT_FILE}")
    print(f"Cartella di Output: {OUTPUT_DIR}")

    start_time = datetime.now()

    # 2. Comando di Esecuzione (Parametri Chiave)
    # --json: Salva ogni articolo come oggetto JSON (strutturato, ma più grande)
    # --links: Mantiene i link wiki (generalmente utile)
    # --extract_all: Elabora tutte le pagine, non solo quelle di dimensioni "ragionevoli"
    # --output: Specifica la cartella di output

    # Per il tuo LLM, è spesso meglio un formato semplice, useremo un'opzione base:
    command = [
        "wikiextractor",
        INPUT_FILE,
        "--output", OUTPUT_DIR,
        "--bytes", "10M",  # Opzionale: limita la dimensione dei file di output a 10MB per gestirli meglio
        "--processes", str(os.cpu_count() or 1),  # Usa tutti i core disponibili per velocizzare
        "--quiet"  # Opzionale: riduce l'output a schermo
    ]

    try:
        # Esegue il comando e attende il completamento
        process = subprocess.run(command, check=True, capture_output=False)

        end_time = datetime.now()
        elapsed_time = end_time - start_time

        print(f"\n--- Elaborazione Completata ---\n")
        print(f"Tempo Totale Trascorso: {elapsed_time}")
        print(f"I file puliti si trovano in: {OUTPUT_DIR}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERRORE: WikiExtractor ha fallito.")
        print(f"Codice di Uscita: {e.returncode}")
        print("Verifica i permessi o la corruzione del file di input.")
    except FileNotFoundError:
        print("\n❌ ERRORE: Assicurati che 'wikiextractor' sia installato e nel tuo PATH.")
        print("Esegui: pip install wikiextractor")


if __name__ == "__main__":
    run_wikiextractor()