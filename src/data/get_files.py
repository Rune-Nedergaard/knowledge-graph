import pandas as pd
from bs4 import BeautifulSoup
import requests
import regex as re
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import PyPDF2
from io import BytesIO

Fil = pd.read_pickle('data/raw/Fil.pkl')
bad_files = defaultdict()
pdf_files = defaultdict()


output_folder = 'data/processed/'

def extract_text_from_html(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    return soup.get_text()

def extract_text_from_pdf(pdf_doc):
    # Create a PdfFileReader object
    pdf_reader = PyPDF2.PdfFileReader(BytesIO(pdf_doc))

    # Initialize an empty string to store the extracted text
    extracted_text = ""

    # Iterate through all the pages in the PDF and extract the text
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        extracted_text += page.extractText()

    return extracted_text

def process_text(text):
    #removing the sentence "PDF to HTML - Convert PDF files to HTML files"
    text = re.sub(r'PDF to HTML - Convert PDF files to HTML files', '', text)
    #removing newlines when there are more than 2 in a row
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def download_and_save(row, output_dir):
    try:
        id = row['id']
        #check if the file is already downloaded
        file_path = os.path.join(output_dir, f"{str(id)}.txt")
        
        if os.path.exists(file_path):
            return

        url = row['filurl']
        

        url = url[:-4]+'/index.htm'
        response = requests.get(url)
        if response.status_code == 200:
            html_doc = response.text
            raw_text = extract_text_from_html(html_doc)
            text = process_text(raw_text)
            
            #saving to output folder
            #file_path = os.path.join(output_dir, f"{str(id)}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        elif response.status_code == 403: #if pdf exists but not htm
            url = row['filurl']

            #download the pdf file
            response = requests.get(url)
            if response.status_code == 200:

                pdf_doc = response.content
                raw = extract_text_from_pdf(pdf_doc)
                text = process_text(raw)
                pdf_files[id] = url

                #saving to output folder
                #file_path = os.path.join(output_dir, f"{str(id)}.txt")

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                #saving to bad_files dict along with the status code
                bad_files[id] = response.status_code
                #print(f"Failed to download {url} with status code {response.status_code}")
                #print(f"There have been {count} bad files so far")
                


        else:
            #print(f"Failed to download {url} with status code {response.status_code}")
            #saving to bad_files dict along with the status code
            bad_files[id] = response.status_code
            #print(f"Failed to download {url} with status code {response.status_code}")
            #print(f"There have been {count} bad files so far")        
    except Exception as e:
        print(f"Error processing {url}: {e}")
        bad_files[id] = e


def download_and_save_all(df, output_dir, num_workers=16):
    os.makedirs(output_dir, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_and_save, row, output_dir): row for _, row in df.iterrows()}
        progress_bar = tqdm(total=len(futures), desc="Downloading and processing", unit="file")

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                row = futures[future]
                print(f"Error processing {row['filurl']}: {e}")
            finally:
                progress_bar.update(1)

        progress_bar.close()

# Call the function with DataFrame and output folder
download_and_save_all(Fil, output_folder)
#save the bad_files dict to a pickle file
pd.to_pickle(bad_files, 'data/raw/bad_files.pkl')
