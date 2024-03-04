# TE-AI-Cup

These files processes scanned PDF documents, ensuring they are in landscape orientation before performing Optical Character Recognition (OCR) using the DocTR library. It consists of two main components: `format_correction.py` for adjusting document orientation, and `test_doctr.py` for OCR processing.

## Setup Instructions

Follow these steps to set up the project environment and run the scripts.

### Prerequisites

- Python 3.6+
- Anaconda or Miniconda (recommended for managing environments)

### Environment Setup

1. **Clone the Repository:**
   If the project is hosted on GitHub, clone it to your local machine. Otherwise, ensure you have both `format_correction.py` and `test_doctr.py` scripts.

2. **Create a Conda Environment:**
   Open a terminal and navigate to the project directory. Create a new Conda environment:

   ```bash
   conda create -n ocrProject python=3.8
   ```

   Activate the environment:

   ```bash
   conda activate ocrProject
   ```

3. **Install Dependencies:**
   Install the required Python packages:
   ```bash
   pip install opencv-python-headless pdf2image python-doctr[torch] numpy
   ```
   Install Poppler (required by `pdf2image` for PDF processing):
   - **macOS:**
     ```bash
     brew install poppler
     ```
   - **Linux:**
     ```bash
     sudo apt-get install poppler-utils
     ```

### Running the Code

1. **Prepare Your Documents:**
   Place the PDF documents you wish to process in a known directory.

2. **Configure the Scripts:**

   - Edit `test_doctr.py` to set `pdf_path` to the path of your PDF document and `output_text_path` to the desired output text file location.

3. **Run the OCR Process:**
   Execute `test_doctr.py` to process your document:
   ```bash
   python test_doctr.py
   ```
   This script automatically adjusts the document orientation and performs OCR, saving the recognized text to the specified output file.

## Additional Notes

- This project is configured for documents that need orientation adjustment before OCR. For directly processing well-oriented documents, modifications to the script may be required.
- The OCR performance can vary based on the quality of the input documents and the complexity of their layouts.
