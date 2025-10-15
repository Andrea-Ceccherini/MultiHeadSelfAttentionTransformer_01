import wikipedia
import os
from datetime import datetime



def calculate_elapsed_time(begin_time_, end_time_):
    elapsed_time_ = end_time_ - begin_time_
    days = elapsed_time_.days
    seconds = elapsed_time_.seconds
    milliseconds = elapsed_time_.microseconds // 1000
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    years, days = divmod(days, 365)
    months, days = divmod(days, 30)
    formatted_elapsed_time_ = f"{years:04}:{months:02}:{days:02}:{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"
    return formatted_elapsed_time_


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
    wikipedia.set_lang("en")  # Imposta la lingua di Wikipedia (italiano)

    downloaded_articles_ = 0

    for subject_ in subjects_to_find_:
        print(f"\n--- Elaborazione di: '{subject_}' ---")

        try:
            # 2. Ottenere la Pagina (Potrebbe sollevare un'eccezione se non trovata)
            page_ = wikipedia.page(subject_, auto_suggest=False)

            # 3. Pulizia del Nome del File
            # Rimuove caratteri non validi per i nomi di file
            file_name_ = "".join(c for c in subject_ if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
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

    begin_time = datetime.now()

    subjects_to_find = [
        "Liver",
        "Liver cirrhosis",
        "Bile",
        "Liver disease",
        "Human anatomy",
        "Hepatotoxicity",
        "Liver failure",
        "Liver support system",
        "Elevated transaminases",
        "Metabolic dysfunction–associated steatotic liver disease",
        "Liver regeneration",
        "Fatty liver disease",
        "Acute liver failure",
        "Hepatomegaly",
        "Lobules of liver",
        "Chronic liver disease",
        "Alanine transaminase",
        "AST/ALT ratio",
        "Liver transplantation",
        "Hepatorenal syndrome",
        "Hepatocellular carcinoma",
        "Liver metastasis",
        "Jaundice",
        "Bilirubin",
        "Pre-eclampsia",
        "Alkaline phosphatase",
        "Kupffer cell",
        "Paracetamol",
        "Shark liver oil",
        "HepaRG",
        "Liver cancer",
        "Alpha-1 antitrypsin deficiency",
        "Schistosomiasis",
        "Gallbladder",
        "Paracetamol poisoning",
        "Aspartate transaminase",
        "Lorazepam",
        "Cholangiocarcinoma",
        "Organ (biology)",
        "Urea-to-creatinine ratio",
        "Sarcoidosis",
        "Foie gras",
        "Alcoholic hepatitis",
        "Interventional radiology",
        "Polycystic liver disease",
        "Sulfamethoxazole",
        "Hypoalbuminemia",
        "Human serum albumin",
        "Blood test",
        "Anorexia nervosa",
        "Hepatitis",
        "Adult-onset Still's disease",
        "Intrahepatic cholestasis of pregnancy",
        "Alcoholic liver disease",
        "TNT",
        "Reye syndrome",
        "Primary sclerosing cholangitis",
        "Cholestasis",
        "Autoimmune hepatitis",
        "Liver cytology",
        "Bone tumor",
        "Basic metabolic panel",
        "Lipotropic",
        "Hepatic encephalopathy",
        "Liver biopsy",
        "Elevated alkaline phosphatase",
        "Hyperbilirubinemia in adults",
        "Hepatitis C",
        "Primary biliary cholangitis",
        "Sanford Rosenthal",
        "Wilson's disease",
        "Bromsulfthalein",
        "Ezetimibe",
        "Congestive hepatopathy",
        "Agaricus subrufescens",
        "Organ-on-a-chip",
        "Valproate",
        "Rodenticide",
        "Small intestine"
    ]

    destination_folder = "../../../Datasets/WikipediaData/"

    collect_and_save_articles_wikipedia(subjects_to_find, destination_folder)

    end_time = datetime.now()
    elapsed_time = calculate_elapsed_time(begin_time, end_time)
    print("__main__() - Elapsed Time =", elapsed_time)

    print("__main__() - END")
