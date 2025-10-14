import wikipedia
import os


def collect_and_save_articles_wikipedia(subjects_to_find_, destination_folder_="dati_wikipedia"):
    """
    Search and download Wikipedia articles by topic,
    saving them in separate text files.

    Args:
        subjects_to_find_ (list): List of strings containing the names of the arguments to search for.
        destination_folder_ (str): The name of the folder where to save the files.
    """
    print("collect_and_save_articles_wikipedia() - BEGIN")
    # 1. Configurazione Iniziale
    print(f"Tentativo di creare la cartella: {destination_folder_}")
    os.makedirs(destination_folder_, exist_ok=True)
    wikipedia.set_lang("eg")  # Imposta la lingua di Wikipedia (italiano)

    downloaded_articles_ = 0

    for subject_ in subjects_to_find_:
        print(f"\n--- Elaborazione di: '{subject_}' ---")

        try:
            # 2. Ottenere la Pagina (Potrebbe sollevare un'eccezione se non trovata)
            page_ = wikipedia.page(subject_, auto_suggest=False)

            # 3. Pulizia del Nome del File
            # Rimuove caratteri non validi per i nomi di file
            file_name_ = "".join(c for c in subject_ if c.isalnum() or c in (' ', '_')).rstrip()
            file_path_ = os.path.join(destination_folder_, f"{file_name_}.txt")

            # 4. Salvare il Contenuto nel File
            with open(file_path_, 'w', encoding='utf-8') as f:
                # La proprietà 'content' restituisce l'intero testo della page_
                f.write(page_.content)

            print(f"✔ Articolo salvato in: {file_path_}")
            downloaded_articles_ += 1

        except wikipedia.exceptions.PageError:
            print(f"❌ Errore: Pagina di Wikipedia non trovata per l'subject_ '{subject_}'. Saltato.")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"⚠️ Attenzione: L'subject_ '{subject_}' è ambiguo. Opzioni: {e.options[:5]}. Saltato.")
        except Exception as e:
            print(f"collect_and_save_articles_wikipedia() - Generic error while processing '{subject_}': {e}. Saltato.")

    print(f"\n--- Finito. {downloaded_articles_} articoli salvati. ---")
    print("collect_and_save_articles_wikipedia() - END")



if __name__ == "__main__":
    print("__main__() - BEGIN")

    subjects_to_find = [
        "liver",
        "Liver cirrhosis",
        "Bile",
        "Liver functions",
        "Human anatomy",
        "Small intestine"
    ]

    destination_folder = "../../../Datasets/WikipediaData/"

    collect_and_save_articles_wikipedia(subjects_to_find, destination_folder)

    print("__main__() - END")
