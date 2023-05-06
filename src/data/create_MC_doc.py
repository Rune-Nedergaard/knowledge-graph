import os
import glob

def numerical_sort_key(file_path):
    return int(os.path.splitext(os.path.basename(file_path))[0])

folder_path = 'data/2020results05constrained'
output_file = os.path.join(folder_path, 'combined.txt')

# Get all the txt files in the folder
txt_files = sorted(glob.glob(os.path.join(folder_path, '*.txt')), key=numerical_sort_key)

# Open the output file for writing
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Loop through each txt file
    for txt_file in txt_files:
        # Get the file number (basename) from the filename
        file_number = os.path.splitext(os.path.basename(txt_file))[0]
        
        # Write the "Multiple choice spørgsmål X:" line to the output file
        outfile.write(f"################## Spørgsmål {file_number} ##################:\n\n")
        
        # Read the content of the txt file and write it to the output file
        with open(txt_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
            outfile.write(content)
            
        # Write a newline to separate the contents of the files
        outfile.write('\n')


print("All text files have been combined in 'combined.txt'.")
