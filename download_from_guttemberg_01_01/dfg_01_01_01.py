
import requests
import os

def download_gutenberg_australia_text(download_url_, output_dir_):
    """
    Download a specific eBook in plain text format from Project Gutenberg Australia.

    Project Gutenberg Australia (gutenberg.net.au) does not have a bulk download API.
    Like the main site. You need to construct the exact URL of the text file.

    :param download_url_:
    :param output_dir_: The directory to save the file in.
    """
    # base_url = "https://gutenberg.net.au"

    txt_file_name_split_ = download_url_.split("/")
    txt_file_name_ = txt_file_name_split_[3] + "_" + txt_file_name_split_[4]

    # Create output directory if it does not exist
    os.makedirs(output_dir_, exist_ok=True)
    output_file_name_path = os.path.join(output_dir_, txt_file_name_)

    print(f"Attempting to download from: {download_url_}")

    try:
        # Use stream=True for efficient downloading, although for .txt files it is not strictly necessary
        response = requests.get(download_url_, stream=True)
        response.raise_for_status()  # Throws an exception for error status codes (4xx or 5xx)

        # The content is decoded as text
        raw_text = response.content.decode('latin-1')


        # Saving text
        with open(output_file_name_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        print(f"✅ Success: '{txt_file_name_}' downloaded and saved in '{output_dir_}'.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error while downloading {txt_file_name_}: {e}")
        print("This could mean that the book ID doesn't exist or that the file format is different.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


if __name__ == "__main__":

    folder_document_path = "../../../Datasets/Au_Books/"

    download_url = "https://gutenberg.net.au/ebooks03/0301501.txt"

    download_gutenberg_australia_text(
        download_url_=download_url,
        output_dir_=folder_document_path
    )

