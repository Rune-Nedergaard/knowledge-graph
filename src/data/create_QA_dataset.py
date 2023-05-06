import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

# Load the required DataFrames
Fil = pd.read_pickle('data/raw/Fil.pkl')
EmneordDokument = pd.read_pickle('data/raw/EmneordDokument.pkl')
Emneord = pd.read_pickle('data/raw/Emneord.pkl')
Emneordstype = pd.read_pickle('data/raw/Emneordstype.pkl')
DokumentAktør = pd.read_pickle('data/raw/DokumentAktør.pkl')
Aktør = pd.read_pickle('data/raw/Aktør.pkl')

def extract_qa_pairs(text):
    qa_pairs = []
    """
    To handle some minor inconsistencies in the data, we have to consider three cases:
    """
    # Case 1: Questions are explicitly marked with "Spørgsmål" and "Svar" - this accounts for 99% of the data
    questions = re.findall(r'Spørgsmål \d+: (.+)', text)
    answers = re.findall(r'Svar \d+: (.+)', text)
    
    if len(questions) == len(answers):
        for q, a in zip(questions, answers):
            qa_pairs.append((q, a))
        return qa_pairs

    # Case 2: Questions are not explicitly marked with "Spørgsmål"
    # Check if the number of "Svar" instances is 5
    svar_count = len(re.findall(r'Svar \d+:', text))

    if svar_count == 5:
        questions = re.findall(r'^(.+)\nSvar \d+:', text, flags=re.MULTILINE)

        if len(questions) == len(answers):
            for q, a in zip(questions, answers):
                qa_pairs.append((q, a))
            return qa_pairs

    # Case 3: There is a problem, return an empty list
    return []


def process_files(input_folder):
    data = []
    files = os.listdir(input_folder)

    # Initialize max_chunk dictionary
    max_chunks = {}

    # Find the maximum chunk value for each filename
    for file in tqdm(files, desc='Finding max chunk values'):
        if file.endswith('.txt'):
            basename, chunk = os.path.splitext(file)[0].rsplit('_', 1)
            chunk = int(chunk)
            if basename in max_chunks:
                max_chunks[basename] = max(max_chunks[basename], chunk)
            else:
                max_chunks[basename] = chunk

    # Process files and update the "chunk" value with "chunk/max_chunk"
    for file in tqdm(files, desc='Processing files'):
        if file.endswith('.txt'):
            basename, chunk = os.path.splitext(file)[0].rsplit('_', 1)
            chunk = int(chunk)
            max_chunk = max_chunks[basename]
            chunk_ratio = f"{chunk}/{max_chunk}"

            # Retrieve additional information
            dokumentid = Fil.loc[Fil['id'] == int(basename), 'dokumentid'].values
            titel = Fil.loc[Fil['id'] == int(basename), 'titel'].values
            if dokumentid.size > 0 and titel.size > 0:
                dokumentid = dokumentid[0]
                titel = titel[0]

                emneordids = EmneordDokument.loc[EmneordDokument['dokumentid'] == dokumentid, 'emneordid'].values
                emneord_info = [(Emneord.loc[Emneord['id'] == emneordid, 'emneord'].values[0],
                                 Emneordstype.loc[Emneordstype['id'] == Emneord.loc[Emneord['id'] == emneordid, 'typeid'].values[0], 'type'].values[0])
                                for emneordid in emneordids]

                filurl = Fil.loc[Fil['id'] == int(basename), 'filurl'].values[0]
                date = Fil.loc[Fil['id'] == int(basename), 'versionsdato'].values[0]
                if date is not None:
                    date = np.datetime_as_string(date, unit='D')

                aktørids = DokumentAktør.loc[DokumentAktør['dokumentid'] == dokumentid, 'aktørid'].values
                entities = [Aktør.loc[Aktør['id'] == aktørid, 'navn'].values[0] for aktørid in aktørids]
            else:
                dokumentid = None
                titel = None
                emneord_info = []
                filurl = None
                date = None
                entities = None

            # Read file content
            with open(os.path.join(input_folder, file), 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract questions and answers
            qna_pairs = extract_qa_pairs(content)

            # Add to data list
            for q, a in qna_pairs:
                for theme, type_ in emneord_info:
                    data.append({
                        'Spørgsmål': q,
                        'Svar': a,
                        'Filename': basename,
                        'Chunk': chunk_ratio,
                        'titel': titel,
                        'emneord': theme,
                        'type': type_,
                        'filurl': filurl,
                        'date': date,
                        'dokumentid': dokumentid,
                        'entities': entities
                    })

    return data




def create_QA_dataset(input_folder):
    data = process_files(input_folder)
    df = pd.DataFrame(data, columns=['Spørgsmål', 'Svar', 'Filename', 'Chunk', 'titel', 'emneord', 'type', 'date', 'filurl'])
    return df


if __name__ == '__main__':
    input_folder = 'data/output_responses'
    df = create_QA_dataset(input_folder)
    #save to pickle
    df.to_pickle('data/QA_dataset.pkl')
    print(df.head())