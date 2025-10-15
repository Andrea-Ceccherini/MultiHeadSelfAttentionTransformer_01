
import requests
import os
import re

def download_gutenberg_australia_text(download_url_, output_dir_):
    """
    Scarica un eBook specifico in formato plain text da Project Gutenberg Australia.

    Project Gutenberg Australia (gutenberg.net.au) non ha un'API di massa per il download
    come il sito principale. È necessario costruire l'URL del file di testo esatto.

    :param download_url_: L'ID numerico del libro nel catalogo (es. 1000 per 'The Iliad').
    :param output_dir_: La directory in cui salvare il file.
    """
    # base_url = "https://gutenberg.net.au"




    txt_file_name_split_ = download_url_.split("/")
    txt_file_name_ = txt_file_name_split_[3] + "_" + txt_file_name_split_[4]

    # Create output directory if it does not exist
    os.makedirs(output_dir_, exist_ok=True)
    output_path = os.path.join(output_dir_, txt_file_name_)

    print(f"Attempting to download from: {download_url_}")

    try:
        # Use stream=True for efficient downloading, although for .txt files it is not strictly necessary
        response = requests.get(download_url_, stream=True)
        response.raise_for_status()  # Throws an exception for error status codes (4xx or 5xx)

        # The content is decoded as text
        raw_text = response.content.decode('latin-1')


        # 2. Salvataggio del testo
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        print(f"✅ Success: '{txt_file_name_}' downloaded and saved in '{output_dir_}'.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Errore durante il download di {txt_file_name_}: {e}")
        print("Ciò potrebbe indicare che l'ID del libro non esiste o che il formato del file è diverso.")
    except Exception as e:
        print(f"❌ Si è verificato un errore inatteso: {e}")


if __name__ == "__main__":

    folder_document_path = "../../../Datasets/Au_Books/"

    download_url = "https://gutenberg.net.au/ebooks03/0301501.txt"

    download_gutenberg_australia_text(
        download_url_=download_url,
        output_dir_=folder_document_path
    )

