# app.py (Corrected - Replaced use_column_width with use_container_width)

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract # NOTE: Requires Tesseract OCR engine installed on the system
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import torch     # Ensure torch is imported
import fitz      # Import the PyMuPDF library
import pandas as pd
import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2       # Still needed for internal camera processing
import numpy as np # Still needed for internal camera processing
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# --- Workaround for Streamlit/Torch __path__ error ---
# This attempts to prevent a known issue where Streamlit's file watcher
# conflicts with torch.classes inspection.
try:
    torch.classes.__path__ = []
    # st.write("Applied torch.classes.__path__ workaround.") # Optional debug message
except Exception as e_workaround:
    st.warning(f"Could not apply torch.classes workaround: {e_workaround}")
# --- End Workaround ---

# --- SET PAGE CONFIG MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(layout="wide")
# ----------------------------------------------------------

# --- Configuration ---
LAYOUTLM_MODEL = "microsoft/layoutlm-base-uncased"
# --- End Configuration ---


# --- Model Loading ---
@st.cache_resource # Cache model/tokenizer loading for performance
def load_model_and_tokenizer():
    try:
        model = LayoutLMForTokenClassification.from_pretrained(LAYOUTLM_MODEL)
        tokenizer = LayoutLMTokenizer.from_pretrained(LAYOUTLM_MODEL)
        # Ignore the warning about classifier weights not being initialized, it's expected
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading LayoutLM model ({LAYOUTLM_MODEL}): {e}")
        st.error("Please check your internet connection and model name.")
        st.stop() # Stop the app if the model can't load

model, tokenizer = load_model_and_tokenizer()
# --- End Model Loading ---


# --- Helper Functions ---

# Function to classify entities based on custom rules
def classify_entity(word, label):
    """Classifies a word based on its LayoutLM label and simple heuristics."""
    label_upper = label.upper() if label else ""
    word_clean = word if word else ""

    if "ADDR" in label_upper:
        return "Address"
    elif "DATE" in label_upper or any(char.isdigit() for char in word_clean):
        if len(word_clean) > 4 and any(c in '/-:.' for c in word_clean): # Heuristic for date format
             return "Date"
    elif "TOTAL" in label_upper:
        return "Total"
    elif "PRICE" in label_upper or word_clean.startswith(('$', '£', '€')) or \
         (word_clean.replace('.', '', 1).replace(',', '', 1).isdigit() and ('.' in word_clean or ',' in word_clean)):
        if any(c.isdigit() for c in word_clean):
             return "Price"
    elif "NAME" in label_upper or (word_clean.istitle() and len(word_clean) > 2): # Check for title case names
        return "Name"
    elif "ITEM" in label_upper:
        return "Item"

    if word_clean.isdigit() and len(word_clean) < 4: # Likely quantity or part of date/code
        return "Other"
    if word_clean.isupper() and len(word_clean) > 1: # All caps might be headers, titles, codes
         return "Other"

    return "Other"


# Function to extract and segment entities
def extract_and_segment_entities(image):
    """Performs OCR, runs LayoutLM, and classifies entities."""
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = data['text']
        n_boxes = len(words)

        if n_boxes == 0 or all(word.strip() == '' for word in words):
            st.warning("No text detected in the document.")
            return {}, []

        valid_indices = [i for i, word in enumerate(words) if word.strip()]
        valid_words = [words[i] for i in valid_indices]

        if not valid_words:
            st.warning("No valid words found after filtering.")
            return {}, []

        encoding = tokenizer(valid_words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        token_type_ids = torch.zeros_like(input_ids)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).tolist()[0]

        segments = {
            "Address": [], "Date": [], "Total": [], "Item": [],
            "Price": [], "Name": [], "Other": []
        }

        word_index = 0
        for i, token_pred_index in enumerate(predictions):
            if input_ids[0, i].item() in tokenizer.all_special_ids: continue

            if word_index < len(valid_indices):
                original_data_index = valid_indices[word_index]
                word = words[original_data_index]
                label_id = token_pred_index
                label = model.config.id2label.get(label_id, "O")

                category = classify_entity(word, label)
                entity = {
                    "word": word,
                    "label": category,
                    "box": (data['left'][original_data_index], data['top'][original_data_index],
                            data['width'][original_data_index], data['height'][original_data_index])
                }
                segments[category].append(entity)
                word_index += 1
            else:
                 break

        return segments, valid_words

    except pytesseract.TesseractNotFoundError:
        st.error("TesseractNotFoundError: Tesseract is not installed or not in your PATH.")
        st.error("Please install Tesseract OCR engine and ensure it's added to your system PATH.")
        st.info("See Tesseract installation guide: https://tesseract-ocr.github.io/tessdoc/Installation.html")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during entity extraction: {e}")
        return {}, []


# Function to draw boxes on image with color coding and transparency
def draw_segmented_boxes(image, segments, fill=False):
    """Draws bounding boxes on the image based on segmented entities."""
    if not isinstance(image, Image.Image):
         st.error("Invalid image type passed to draw_segmented_boxes.")
         return None

    vis_image = image.copy()
    if vis_image.mode != "RGBA":
        vis_image = vis_image.convert("RGBA")

    draw = ImageDraw.Draw(vis_image, "RGBA")
    try:
        font = ImageFont.load_default()
    except IOError:
        st.warning("Default font not found. Text labels on boxes might be missing.")
        font = None

    color_map = {
        "Address": (0, 0, 255, 100), "Date": (0, 128, 0, 100), "Total": (255, 0, 0, 100),
        "Item": (255, 165, 0, 100), "Price": (0, 255, 255, 100), "Name": (255, 0, 255, 100),
        "Other": (128, 128, 128, 100)
    }

    for segment, entities in segments.items():
        color = color_map.get(segment, (0, 0, 0, 100))
        outline_color = color[:-1] + (255,)

        for entity in entities:
            try:
                 box_coords = entity['box']
                 x, y, w, h = map(int, [box_coords[0], box_coords[1], box_coords[2], box_coords[3]])
                 box = (x, y, x + w, y + h)

                 if fill:
                     draw.rectangle(box, fill=color, outline=outline_color, width=2)
                     if font:
                          text_position = (x + 2, y + 2)
                          draw.text(text_position, entity['word'], fill="black", font=font)
                 else:
                     draw.rectangle(box, outline=outline_color, width=2)
                     if font:
                          try:
                              if hasattr(draw, 'textbbox'):
                                   text_bbox = draw.textbbox((0,0), entity['word'], font=font)
                                   text_width = text_bbox[2] - text_bbox[0]; text_height = text_bbox[3] - text_bbox[1]
                              else: # Fallback
                                   text_width = draw.textlength(entity['word'], font=font)
                                   text_height = font.size if hasattr(font, 'size') else 10
                          except AttributeError:
                                text_width = len(entity['word']) * 6; text_height = 10

                          text_position = (x, y - text_height - 2)
                          label_bg_box = (text_position[0] - 1, text_position[1] - 1,
                                          text_position[0] + text_width + 1, text_position[1] + text_height + 1)
                          draw.rectangle(label_bg_box, fill=(255, 255, 255, 180))
                          draw.text(text_position, entity['word'], fill=outline_color, font=font)

            except (IndexError, ValueError, TypeError) as box_err:
                 st.warning(f"Skipping invalid box data for word '{entity.get('word', 'N/A')}': {box_err}")
                 continue

    return vis_image


# Function to load PDF as images using PyMuPDF
def load_image_from_pdf(uploaded_pdf_file):
    """Converts an uploaded PDF file object into a list of PIL Images using PyMuPDF."""
    images = []
    try:
        pdf_bytes = uploaded_pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150) # Adjust DPI as needed
            img_bytes = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_bytes))
            images.append(pil_image)

        doc.close()

    except Exception as e:
        st.error(f"Failed to load or convert PDF using PyMuPDF: {e}")
        return []

    return images


# Function to convert text to PDF using ReportLab
def convert_text_to_pdf(text):
    """Generates a PDF file containing the extracted text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text_object = c.beginText(30, height - 40)
    text_object.setFont("Helvetica", 10)
    text_object.textLine("Extracted OCR Text:")
    text_object.moveCursor(0, 14)
    lines = text.split('\n')
    for line in lines:
        text_object.textLine(line)
    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# Video processing class for camera input
class VideoTransformer(VideoTransformerBase):
    """Captures frames from the WebRTC stream."""
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img


# --- Streamlit App UI Starts Here ---
st.title("📄 Document Information Extraction (Single Document)")

# Sidebar for Input Options
st.sidebar.title("Input Options")
option = st.sidebar.selectbox(
    "Choose how to provide the document:",
    ("Upload Document", "Scan with Internal Camera") # Removed External Camera option
)

# Initialize variables
image = None
extracted_words = []
segments = {}

if option == "Upload Document":
    uploaded_file = st.sidebar.file_uploader("Upload PNG, JPG, JPEG, or PDF file", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        image = None; extracted_words = []; segments = {} # Reset
        with st.spinner("Processing uploaded file..."):
            if uploaded_file.type == "application/pdf":
                pdf_images = load_image_from_pdf(uploaded_file) # Use PyMuPDF version
                if pdf_images:
                    image = pdf_images[0] # Use only the first page
                    st.sidebar.info(f"Loaded first page of PDF: {uploaded_file.name}")
                else:
                    st.sidebar.error("PDF processing failed.")
            else:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.sidebar.success(f"Image loaded: {uploaded_file.name}")
                except Exception as e:
                    st.sidebar.error(f"Failed to load image file: {e}")

elif option == "Scan with Internal Camera":
    st.sidebar.write("Use the internal camera to scan.")
    webrtc_ctx = webrtc_streamer(
        key="internal-camera",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    )

    if webrtc_ctx.video_transformer:
        if st.sidebar.button("📸 Capture Document"):
            image = None; extracted_words = []; segments = {} # Reset
            if webrtc_ctx.video_transformer.frame is not None:
                img_array = webrtc_ctx.video_transformer.frame
                image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                st.sidebar.success("Image captured!")
            else:
                st.sidebar.warning("No frame captured from camera.")


# --- Main Area for Processing and Displaying Results ---
if image is not None:
    st.header("Analysis Result")
    with st.spinner("Analyzing Document..."):
        segments, extracted_words = extract_and_segment_entities(image)

    if not segments and not extracted_words:
        # Check if image exists before trying to display it on failure
        if image:
            st.error("Could not extract text from the document. Please check image quality or Tesseract setup.")
            st.image(image, caption='Document Analysis Failed', use_container_width=True) # Changed parameter here
        else:
             st.error("Image could not be loaded or processed.")
    else:
        # Display Results in Columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original Document")
            st.image(image, caption='Uploaded/Scanned Document', use_container_width=True) # Changed parameter here
        with col2:
            st.subheader("Segmented (Filled)")
            if segments:
                filled_annotated_image = draw_segmented_boxes(image, segments, fill=True)
                if filled_annotated_image:
                     st.image(filled_annotated_image, caption="Segmented Document", use_container_width=True) # Changed parameter here
                else: st.warning("Could not draw filled segmentation.")
            else: st.info("No segments found to display.")
        with col3:
            st.subheader("Segmented (Outlined)")
            if segments:
                annotated_image = draw_segmented_boxes(image, segments, fill=False)
                if annotated_image:
                     st.image(annotated_image, caption="Segmented Details", use_container_width=True) # Changed parameter here
                else: st.warning("Could not draw outlined segmentation.")
            else: st.info("No segments found to display.")

        # Display Extracted Text using st.text_area
        st.subheader("Extracted OCR Text")
        if extracted_words:
            formatted_text = " ".join(extracted_words)
            st.text_area(
                label="Extracted Text", # Label for the text area
                value=formatted_text,   # The text to display
                height=300,             # Set desired height
                key="ocr_text_area",     # Add a unique key
                disabled=True           # Make it read-only
            )

            # Download Options
            st.subheader("Download Extracted Text")
            col_dl_1, col_dl_2, col_dl_3 = st.columns(3)
            with col_dl_1:
                text_bytes = formatted_text.encode('utf-8')
                st.download_button(label="📥 Download as Text (.txt)", data=text_bytes, file_name="extracted_text.txt", mime="text/plain", key="txt_dl")
            with col_dl_2:
                try:
                    pdf_buffer = convert_text_to_pdf(formatted_text)
                    st.download_button(label="📄 Download as PDF", data=pdf_buffer, file_name="extracted_text.pdf", mime="application/pdf", key="pdf_dl")
                except Exception as pdf_err: st.error(f"PDF Error: {pdf_err}")
            with col_dl_3:
                try:
                    df = pd.DataFrame({"Extracted Text": [formatted_text]})
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer: df.to_excel(writer, index=False, sheet_name='Extracted_Text')
                    excel_buffer.seek(0)
                    st.download_button(label="📊 Download as Excel (.xlsx)", data=excel_buffer, file_name="extracted_text.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="excel_dl")
                except Exception as excel_err: st.error(f"Excel Error: {excel_err}")
        else:
            st.info("No text was extracted to display or download.")

elif not option.startswith("Scan"):
    st.info("Upload a document or use the internal camera option in the sidebar to start.")

# --- End Streamlit App ---