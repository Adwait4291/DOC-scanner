# Document Segmentation and Information Extraction Tool

## Overview

This project provides an interactive web application for extracting text and segmenting information from document images (receipts, invoices, forms, etc.). Users can upload document files (PNG, JPG, PDF) or use their internal camera to capture documents. The application utilizes Optical Character Recognition (OCR) to extract text and a pre-trained LayoutLM model to classify words into predefined categories like Address, Date, Total, Item, Price, Name, and Other. The results, including the original image, segmented visualizations, and extracted text, are displayed in a user-friendly interface built with Streamlit. Users can also download the extracted text in various formats (TXT, PDF, Excel).

## Features

* **Multiple Input Methods:**
    * Upload document images (PNG, JPG, JPEG) or PDF files[cite: 2].
    * Scan documents directly using the internal webcam[cite: 2].
    * *(Experimental in `main.py`)* Option to guide users for scanning with an external phone camera via QR code[cite: 2].
* **OCR Text Extraction:** Employs Tesseract OCR to extract raw text from the document image[cite: 2].
* **Layout-Aware Information Segmentation:** Uses the `microsoft/layoutlm-base-uncased` model to classify extracted words based on their content and position within the document[cite: 2].
* **Entity Classification:** Segments words into categories: Address, Date, Total, Item, Price, Name, Other, using a combination of LayoutLM predictions and heuristics[cite: 2].
* **Visualizations:**
    * Displays the original uploaded/scanned document[cite: 2].
    * Shows the document with colored bounding boxes overlaid on segmented entities (both filled and outlined versions)[cite: 2].
* **Text Display & Download:**
    * Presents the full extracted OCR text in a text area[cite: 2].
    * Allows downloading the extracted text as a `.txt` file, a generated `.pdf` file, or an `.xlsx` spreadsheet[cite: 2].
* **PDF Handling:** Processes single-page (`app.py` [cite: 1]) or multi-page PDFs (`main.py` [cite: 2]), extracting the first page or all pages respectively for analysis.
* **Fallback Image Processing:** Includes basic image pre-processing (grayscale, thresholding) if initial OCR fails (`main.py` [cite: 2]).

## Technologies Used

This project leverages several powerful Python libraries and external tools:

1.  **[Streamlit](https://streamlit.io/):**
    * **Why:** Used as the primary framework for building the interactive web application UI.
    * **Benefit:** Enables rapid development of data-centric applications with simple Python scripting, handling widgets, state management, and data display efficiently.

2.  **[Pillow (PIL Fork)](https://python-pillow.org/):**
    * **Why:** Essential for image loading, manipulation (like format conversion), and drawing annotations (bounding boxes, text labels) on images.
    * **Benefit:** It's the standard Python imaging library, providing comprehensive image handling capabilities required for processing uploads and generating visualizations.

3.  **[Pytesseract](https://github.com/madmaze/pytesseract):**
    * **Why:** Acts as a Python wrapper for Google's Tesseract-OCR Engine. It extracts text content and bounding box information from images.
    * **Benefit:** Provides easy Python access to the powerful and widely used Tesseract OCR engine, forming the foundation for text extraction. *Requires external Tesseract installation.*

4.  **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index):**
    * **Why:** Used to load and run the pre-trained `microsoft/layoutlm-base-uncased` model (`LayoutLMForTokenClassification`).
    * **Benefit:** LayoutLM is specifically designed for document understanding, integrating textual and layout information, making it superior to text-only models for tasks like form/receipt analysis. The Transformers library simplifies downloading, caching, and using state-of-the-art NLP models.

5.  **[PyTorch](https://pytorch.org/):**
    * **Why:** The deep learning framework underpinning the LayoutLM model used via the Transformers library.
    * **Benefit:** Provides the necessary tensor computations and neural network infrastructure for running the LayoutLM model inference efficiently.

6.  **PDF Processing:**
    * **[PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF):** (Used in `app.py`[cite: 1], listed in `requirements.txt` [cite: 2])
        * **Why:** Converts PDF pages into images for OCR processing.
        * **Benefit:** Known for being fast and efficient in rendering PDF pages to various image formats directly from file streams or paths.
    * **[pdf2image](https://github.com/Belval/pdf2image):** (Used in `main.py`[cite: 2], listed in `requirements.txt` [cite: 2])
        * **Why:** An alternative library for converting PDF pages into PIL Images. Often relies on the external `poppler-utils` dependency.
        * **Benefit:** Provides a straightforward API for PDF-to-image conversion.

7.  **[Streamlit-WebRTC](https://github.com/whitphx/streamlit-webrtc):**
    * **Why:** Integrates real-time video streaming capabilities into the Streamlit app, enabling document capture via the internal webcam.
    * **Benefit:** Simplifies the complex process of accessing camera streams within a web application framework like Streamlit.

8.  **[OpenCV-Python (cv2)](https://pypi.org/project/opencv-python/):**
    * **Why:** Used internally by `streamlit-webrtc` for handling video frames and potentially for more advanced image pre-processing (though current fallback is basic).
    * **Benefit:** The standard library for computer vision tasks, offering a vast array of functions for image manipulation and analysis.

9.  **[Pandas](https://pandas.pydata.org/):**
    * **Why:** Used to structure the extracted text for export into an Excel (.xlsx) file format.
    * **Benefit:** Simplifies data manipulation and provides robust I/O tools, including easy creation of Excel files via the `to_excel` method.

10. **[Openpyxl](https://openpyxl.readthedocs.io/en/stable/):**
    * **Why:** Required by Pandas to write data to `.xlsx` Excel files[cite: 2].
    * **Benefit:** A dedicated library for reading and writing modern Excel file formats.

11. **[ReportLab](https://www.reportlab.com/opensource/):**
    * **Why:** Used to generate PDF documents containing the extracted text for download.
    * **Benefit:** A powerful and versatile library for programmatically creating PDFs in Python, allowing for text formatting and layout control.

12. **[qrcode](https://github.com/lincolnloop/python-qrcode):** (Used in `main.py` [cite: 2])
    * **Why:** Generates QR codes, used in the "External Camera" feature to display the app's URL for access on a mobile device.
    * **Benefit:** Simple and effective library focused solely on QR code generation.

13. **[Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract):**
    * **Why:** The core engine performing the OCR. `pytesseract` is just the Python interface to this engine[cite: 1].
    * **Benefit:** A highly accurate, open-source OCR engine supporting multiple languages. **Note:** This must be installed separately on the system where the application runs.

## Usage

1.  Ensure Tesseract (and Poppler if needed) is correctly installed and accessible in your PATH.
2.  Make sure you are in the project directory and the virtual environment is activated.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    # OR if using the main.py script
    # streamlit run main.py
    ```
4.  The application will open in your default web browser.
5.  Use the sidebar options to either "Upload Document" or "Scan with Internal Camera"[cite: 2].
6.  If uploading, select one or more image/PDF files. If scanning, allow camera access and click "Capture Document"[cite: 2].
7.  The application will process the document(s) and display the original image, segmented visualizations, and extracted text.
8.  Use the download buttons to save the extracted text in your preferred format (TXT, PDF, XLSX)[cite: 2].

## Code Snippets

**Loading the LayoutLM Model and Tokenizer (`app.py`):** [cite: 1]
```python
import streamlit as st
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import torch

LAYOUTLM_MODEL = "microsoft/layoutlm-base-uncased"

@st.cache_resource # Cache model loading
def load_model_and_tokenizer():
    try:
        model = LayoutLMForTokenClassification.from_pretrained(LAYOUTLM_MODEL)
        tokenizer = LayoutLMTokenizer.from_pretrained(LAYOUTLM_MODEL)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading LayoutLM model ({LAYOUTLM_MODEL}): {e}")
        st.stop()

model, tokenizer = load_model_and_tokenizer()
