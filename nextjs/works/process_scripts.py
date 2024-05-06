# process_script.py
import sys
import os
# Ensure you import your processing function correctly
from main.main import process_pdf_or_folder
from main.validate import finalVal

if __name__ == "__main__":
    file_path = sys.argv[1]
    fileID = os.path.basename(file_path)
    result = process_pdf_or_folder(file_path, is_file=True)
    print(result)
    finalVal(f'output_{fileID}.json', file_path)
