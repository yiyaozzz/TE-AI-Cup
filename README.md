# TE-AI-Cup: ML Automation for Lot History Record Digitization

This project delivers an intelligent, end-to-end digitization system developed to automate the recognition of handwritten data on Lot History Record (LHR) sheets. The system combines advanced machine learning techniques with a user-friendly web-based UI, providing seamless data extraction and integration with SAP systems. It achieves 98% detection accuracy, cutting processing time from 8 hours to just 2 minutes.

## Key Features

- UI-Driven Workflow: An intuitive web interface streamlines user interaction for uploading files and viewing results.
- High Accuracy: Powered by YOLO and OpenCV, achieving 98% accuracy in handwritten data recognition.
- Time Efficiency: Reduces manual processing from hours to minutes.
- SAP Integration: Outputs data in an SAP-compatible format, simplifying enterprise adoption.
- Scalability: Deployed on AWS Cloud for reliable and scalable performance.

## System Workflow

### Backend Workflow

1. Input Handling:
   - Accepts single PDF files or folders via the UI or CLI.
2. PDF Conversion:
   - Converts PDFs into PNG images using convert_pdf_to_png.
3. Pre-Processing:
   - Enhances images using OpenCV for optimal OCR performance.
4. OCR:
   - Extracts handwritten data using DocTR and processes specific table columns.
5. Image Classification:
   - YOLO-v8m models classify and detect objects in specified columns.
6. Validation:
   - Ensures the data extracted is accurate and formatted correctly via validation scripts.
7. Output Generation:
   - Consolidates and formats results into Excel files compatible with SAP.

### User Interface (UI)

- File Upload:
  - Users can upload PDFs or folders directly from the browser.
- Real-Time Feedback:
  - Displays processing status and logs.
- Result Download:
  - Outputs can be downloaded directly in Excel format.

## File Structure

```bash
main/
│
├── pdf_converter.py     # Converts PDFs into images
├── ocr_det.py           # Handles OCR detection and data extraction
├── tt.py                # Processes images for object tracking
├── firstWord.py         # Detects key words in specific table columns
├── yolo_pred.py         # Implements YOLO-based object detection
├── popSheet.py          # Consolidates and generates Excel output
├── validate.py          # Validates extracted data for accuracy
├── ui/                  # Contains code for the web interface
│   ├── app.py           # Main UI logic
│   ├── templates/       # HTML templates for the web interface
│   └── static/          # CSS and JavaScript for the frontend
```

## Setup Instructions

### Prerequisites

- Python: Version 3.6+ is required.
- Anaconda/Miniconda: Recommended for environment management.
- Poppler: Necessary for PDF-to-image conversion.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Set Up Virtual Environment:**
   Open a terminal and navigate to the project directory. Create a new Conda environment:

   ```bash
   conda create -n lhr-digitization python=3.8
   conda activate lhr-digitization
   ```

3. **Install Required Dependencies:**
   ```bash
   pip install opencv-python-headless pdf2image python-doctr[torch] flask numpy
   ```
4. **Install Poppler:**
   macOS:
   ```bash
   brew install poppler
   ```
   Linux:
   ```bash
   sudo apt-get install poppler-utils
   ```
5. **Run the Web Interface:**
   ```bash
   python ui/app.py
   ```
   Access the UI at http://127.0.0.1:5000 in your browser.

## Running the Backend

### Command-Line Interface (CLI)

- **Process a Single PDF:**
  ```bash
  python main.py --input <file_path> --is_file True
  ```
- **Process a Folder:**
  ```bash
  python main.py --input <folder_path>
  ```

## Web Interface

1. Upload PDF files or folders via the UI.
2. Monitor the real-time progress on the web interface.
3. Download processed results in Excel format.

## Workflow Example

1. PDF Input:
   - A folder containing multiple LHR PDFs is uploaded via the UI.
2. Processing:
   - PDFs are converted into images and processed for OCR and object detection.
3. Validation:
   - Extracted data is validated for accuracy.
4. Excel Output:
   - Processed data is consolidated into Excel files and made available for download.

## Additional Notes

- Temp Tables:
  - Temporary files are created during processing and stored in tempTables\_<fileID>.
- Validation:
  - The validate.py script ensures the extracted data meets quality requirements.
- Customizations:
  - Adjust the backend scripts to handle additional use cases or different document layouts.
