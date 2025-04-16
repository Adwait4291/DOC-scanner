# main.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract # NOTE: Requires Tesseract OCR engine installed on the system
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer
import torch
from pdf2image import convert_from_path
import pandas as pd
import io
import os
from tempfile import NamedTemporaryFile
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import qrcode

# --- Configuration ---
LAYOUTLM_MODEL = "microsoft/layoutlm-base-uncased"
HEADER_IMAGE_PATH = "image1.png"  # Path to your header image (optional)
# IMPORTANT: Replace with your actual app URL if deploying or using ngrok locally for external camera.
# For purely local testing without external camera, the default localhost is usually fine.
APP_PUBLIC_URL = "http://localhost:8501" # Default Streamlit local URL, change if needed
# --- End Configuration ---

# --- Model Loading ---
# Add error handling in case the model download fails
@st.cache_resource # Cache model/tokenizer loading for performance
def load_model_and_tokenizer():
    try:
        model = LayoutLMForTokenClassification.from_pretrained(LAYOUTLM_MODEL)
        tokenizer = LayoutLMTokenizer.from_pretrained(LAYOUTLM_MODEL)
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
    # Normalize label for easier checking
    label_upper = label.upper() if label else ""
    word_clean = word if word else ""

    if "ADDR" in label_upper:
        return "Address"
    # Check for date patterns or digits alongside DATE label
    elif "DATE" in label_upper or any(char.isdigit() for char in word_clean):
         # Basic check to avoid classifying all numbers as dates
        if len(word_clean) > 4 and any(c in '/-:.' for c in word_clean): # Heuristic for date format
             return "Date"
    elif "TOTAL" in label_upper:
        return "Total"
    # Check for price patterns (currency symbols, digits with decimals)
    elif "PRICE" in label_upper or word_clean.startswith(('$', '£', '€')) or \
         (word_clean.replace('.', '', 1).replace(',', '', 1).isdigit() and ('.' in word_clean or ',' in word_clean)):
        # Further refine to avoid simple numbers being tagged as Price
        if any(c.isdigit() for c in word_clean):
             return "Price"
    elif "NAME" in label_upper or (word_clean.istitle() and len(word_clean) > 2): # Check for title case names
        return "Name"
    elif "ITEM" in label_upper:
        return "Item"

    # Fallback based on content if no strong signal from label
    if word_clean.isdigit() and len(word_clean) < 4: # Likely quantity or part of date/code
        return "Other" # Avoid classifying simple digits as Date/Price without context
    if word_clean.isupper() and len(word_clean) > 1: # All caps might be headers, titles, codes
         return "Other" # Or could refine to "Header", "Code" etc.

    return "Other"


# Function to extract and segment entities
def extract_and_segment_entities(image):
    """Performs OCR, runs LayoutLM, classifies entities, and handles potential errors."""
    try:
        # Use pytesseract to get detailed data including bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = data['text']
        n_boxes = len(words)

        # Check if any text was detected
        if n_boxes == 0 or all(word.strip() == '' for word in words):
            st.warning("No text detected by initial OCR pass. Trying fallback processing...")
            try:
                 processed_image = process_with_fallback(image)
                 data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
                 words = data['text']
                 n_boxes = len(words)
                 if n_boxes == 0 or all(word.strip() == '' for word in words):
                      st.error("No text detected even after fallback processing.")
                      return {}, [] # Return empty if still no text
                 st.success("Fallback processing yielded text.")
                 image_to_process = processed_image # Use the processed image's data
            except Exception as fallback_error:
                 st.error(f"Fallback processing failed: {fallback_error}")
                 return {}, [] # Return empty if fallback fails
        else:
             image_to_process = image # Use original image's data

        # Filter out empty words and prepare for LayoutLM
        valid_indices = [i for i, word in enumerate(words) if word.strip()]
        valid_words = [words[i] for i in valid_indices]

        if not valid_words:
            st.warning("No valid words found after filtering.")
            return {}, []

        # Tokenize and run LayoutLM
        encoding = tokenizer(valid_words, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        # Ensure token_type_ids are provided if the model expects them (base LayoutLM does)
        # For LayoutLM, token_type_ids are typically all zeros for single sequence tasks.
        token_type_ids = torch.zeros_like(input_ids)

        with torch.no_grad(): # Disable gradient calculation for inference
             outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).tolist()[0] # Get predicted class index for each token

        # Initialize segments dictionary
        segments = {
            "Address": [], "Date": [], "Total": [], "Item": [],
            "Price": [], "Name": [], "Other": []
        }

        word_index = 0
        for i, token_pred_index in enumerate(predictions):
            # Skip special tokens like [CLS], [SEP], [PAD]
            if input_ids[0, i].item() in tokenizer.all_special_ids:
                continue

            # Map token prediction back to original word
            if word_index < len(valid_indices):
                original_data_index = valid_indices[word_index]
                word = words[original_data_index]
                label_id = token_pred_index
                label = model.config.id2label.get(label_id, "O") # Get label name, default to "O" (Other)

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
                 # This condition handles cases where predictions might exceed words due to tokenization artifacts
                 # Although with is_split_into_words=True, it should align well.
                 break


        # Return all words detected by OCR (including those filtered before LayoutLM if needed)
        # For simplicity here, return the valid words processed by LayoutLM.
        return segments, valid_words

    except pytesseract.TesseractNotFoundError:
        st.error("Tesseract is not installed or not in your PATH. Please install Tesseract.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during entity extraction: {e}")
        # Attempt fallback if not already done
        if 'processed_image' not in locals(): # Check if fallback was tried
             st.warning("Trying fallback processing due to error...")
             try:
                processed_image = process_with_fallback(image)
                # Rerun the extraction on the processed image
                return extract_and_segment_entities(processed_image)
             except Exception as fallback_error:
                st.error(f"Fallback processing also failed: {fallback_error}")
                return {}, [] # Return empty if fallback fails too
        else:
             return {}, [] # Return empty if error occurred after fallback


# Fallback processing function (Basic Implementation)
def process_with_fallback(image):
    """
    Applies basic pre-processing if initial OCR fails.
    Current implementation is very simple (grayscale + thresholding).
    Consider adding more advanced techniques (e.g., OpenCV's adaptiveThreshold,
    noise reduction, deskewing) if needed for difficult documents.
    """
    st.write("Applying basic fallback image processing (Grayscale + Thresholding)...")
    try:
        # Convert to grayscale
        processed_image = image.convert("L")
        # Apply simple binary thresholding
        processed_image = processed_image.point(lambda x: 0 if x < 128 else 255, '1')
        # Convert back to RGB for compatibility with downstream functions
        return processed_image.convert("RGB")
    except Exception as e:
        st.error(f"Error during fallback processing: {e}")
        return image # Return original image if fallback fails


# Function to draw boxes on image with color coding and transparency
def draw_segmented_boxes(image, segments, fill=False):
    """Draws bounding boxes on the image based on segmented entities."""
    if not isinstance(image, Image.Image):
         st.error("Invalid image type passed to draw_segmented_boxes.")
         return None # Or return a blank image

    # Ensure image is in RGBA mode for transparency
    vis_image = image.copy()
    if vis_image.mode != "RGBA":
        vis_image = vis_image.convert("RGBA")

    draw = ImageDraw.Draw(vis_image, "RGBA")
    try:
        # Use a default font if possible, consider adding a specific font file
        font = ImageFont.load_default()
    except IOError:
        st.warning("Default font not found. Text labels on boxes might be missing.")
        font = None # Set font to None if it fails to load

    color_map = {
        "Address": (0, 0, 255, 100),    # Blue with transparency
        "Date": (0, 128, 0, 100),      # Dark Green with transparency
        "Total": (255, 0, 0, 100),      # Red with transparency
        "Item": (255, 165, 0, 100),    # Orange with transparency
        "Price": (0, 255, 255, 100),    # Cyan with transparency
        "Name": (255, 0, 255, 100),      # Magenta with transparency
        "Other": (128, 128, 128, 100)   # Gray with transparency
    }

    for segment, entities in segments.items():
        color = color_map.get(segment, (0, 0, 0, 100)) # Default to black with transparency
        outline_color = color[:-1] + (255,) # Solid outline color

        for entity in entities:
            try:
                 box_coords = entity['box']
                 # Ensure box coordinates are valid integers
                 x, y, w, h = map(int, [box_coords[0], box_coords[1], box_coords[2], box_coords[3]])
                 box = (x, y, x + w, y + h)

                 if fill:
                     draw.rectangle(box, fill=color, outline=outline_color, width=2)
                     # Draw text inside the box if font is available
                     if font:
                          text_position = (x + 2, y + 2)
                          # Draw black text for better visibility on colored background
                          draw.text(text_position, entity['word'], fill="black", font=font)
                 else:
                     draw.rectangle(box, outline=outline_color, width=2)
                     # Draw the original text label above the bounding box if font is available
                     if font:
                          # Calculate text size to position it above the box
                          try:
                               # Use textbbox for more accurate sizing if available, else textlength/size
                               if hasattr(draw, 'textbbox'):
                                    text_bbox = draw.textbbox((0,0), entity['word'], font=font)
                                    text_width = text_bbox[2] - text_bbox[0]
                                    text_height = text_bbox[3] - text_bbox[1]
                               else: # Fallback for older Pillow versions
                                    text_width = draw.textlength(entity['word'], font=font)
                                    # Estimate height based on font size (approximation)
                                    text_height = font.size if hasattr(font, 'size') else 10
                          except AttributeError: # Handle cases where font methods fail
                                text_width = len(entity['word']) * 6 # Rough estimate
                                text_height = 10 # Rough estimate

                          text_position = (x, y - text_height - 2)
                          # Add a small background rectangle for the text label for better readability
                          label_bg_box = (text_position[0] - 1, text_position[1] - 1,
                                          text_position[0] + text_width + 1, text_position[1] + text_height + 1)
                          draw.rectangle(label_bg_box, fill=(255, 255, 255, 180)) # Semi-transparent white bg
                          draw.text(text_position, entity['word'], fill=outline_color, font=font)

            except (IndexError, ValueError, TypeError) as box_err:
                 st.warning(f"Skipping invalid box data for word '{entity.get('word', 'N/A')}': {box_err}")
                 continue # Skip this entity if box data is problematic

    return vis_image


# Function to load images from PDF
def load_images_from_pdf(uploaded_pdf_file):
    """Converts an uploaded PDF file object into a list of PIL Images."""
    images = []
    try:
        # Use a temporary file to store the uploaded PDF content
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_pdf_file.read())
            temp_pdf_path = temp_pdf.name

        # Convert PDF to images
        images = convert_from_path(temp_pdf_path)
        os.remove(temp_pdf_path) # Clean up the temporary file
        return images
    except Exception as e:
        st.error(f"Failed to load or convert PDF: {e}")
        # Clean up temp file even if conversion fails
        if 'temp_pdf_path' in locals() and os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return []


# Function to convert text to PDF using ReportLab
def convert_text_to_pdf(text):
    """Generates a PDF file containing the extracted text."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Set up text object
    text_object = c.beginText()
    text_object.setTextOrigin(30, height - 40) # Position near top-left
    text_object.setFont("Helvetica", 10) # Use a standard font

    # Add title
    text_object.textLine("Extracted OCR Text:")
    text_object.moveCursor(0, 14) # Move down for spacing

    # Add the extracted text, handling line breaks
    lines = text.split('\n')
    for line in lines:
        text_object.textLine(line)

    # Draw the text object and save
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
        return img # Display the frame back in the UI

# --- End Helper Functions ---


# --- Streamlit App ---
st.set_page_config(layout="wide") # Use wider layout
st.title("📄 Document Segmentation and Information Extraction")

# Load and display the header image (with error handling)
try:
    if os.path.exists(HEADER_IMAGE_PATH):
        header_image = Image.open(HEADER_IMAGE_PATH)
        st.image(header_image, use_column_width=False, width=200) # Control width if needed
    # else: # Optionally notify if image is expected but not found
    #    st.sidebar.warning(f"Header image '{HEADER_IMAGE_PATH}' not found.")
except Exception as e:
    st.sidebar.warning(f"Could not load header image: {e}")


# Sidebar for Input Options
st.sidebar.title("Input Options")
option = st.sidebar.selectbox(
    "Choose how to provide documents:",
    ("Upload Documents", "Scan with Internal Camera", "Scan with External Camera (Phone)")
    # "Upload Images from Phone" is functionally same as "Upload Documents"
)

# Initialize session state to store images across runs if needed
if 'doc_images' not in st.session_state:
    st.session_state.doc_images = []

captured_image = None # To store image from camera capture

if option == "Upload Documents":
    uploaded_files = st.sidebar.file_uploader(
        "Upload PNG, JPG, JPEG, or PDF files",
        type=["png", "jpg", "jpeg", "pdf"],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.doc_images = [] # Clear previous images when new files are uploaded
        with st.spinner("Processing uploaded files..."):
            for uploaded_file in uploaded_files:
                st.sidebar.write(f"Processing: {uploaded_file.name}")
                if uploaded_file.type == "application/pdf":
                    pdf_images = load_images_from_pdf(uploaded_file)
                    if pdf_images:
                        st.session_state.doc_images.extend(pdf_images)
                else:
                    try:
                        image = Image.open(uploaded_file).convert("RGB") # Ensure RGB
                        st.session_state.doc_images.append(image)
                    except Exception as e:
                        st.error(f"Failed to load image file {uploaded_file.name}: {e}")

elif option == "Scan with Internal Camera":
    st.sidebar.write("Use the internal camera to scan.")
    webrtc_ctx = webrtc_streamer(
        key="internal-camera",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
        #rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} # Optional STUN server
    )

    if webrtc_ctx.video_transformer:
        if st.sidebar.button("📸 Capture Document"):
            if webrtc_ctx.video_transformer.frame is not None:
                img_array = webrtc_ctx.video_transformer.frame
                captured_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                st.session_state.doc_images = [captured_image] # Replace list with the single capture
                st.sidebar.success("Image captured!")
            else:
                st.sidebar.warning("No frame captured from camera.")

elif option == "Scan with External Camera (Phone)":
    st.sidebar.write("Scan the QR code with your phone's camera app or QR reader:")

    # Generate QR code using the configured URL
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=6, border=4)
    qr.add_data(APP_PUBLIC_URL)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")

    # Display QR code in sidebar
    buf = io.BytesIO()
    img_qr.save(buf, format="PNG")
    st.sidebar.image(buf.getvalue(), caption="Scan to open app on phone", use_column_width=True)

    st.sidebar.info(f"Ensure app is accessible at: {APP_PUBLIC_URL}")
    st.sidebar.warning("Requires phone on same network or app publicly hosted (e.g., using ngrok for local testing).")
    st.sidebar.write("Upload the photo taken from your phone using the 'Upload Documents' option above.")
    # Note: Direct capture from external phone to Streamlit requires more complex setup (e.g., websockets, dedicated server)
    # This approach guides the user to upload manually after scanning QR to access the app.


# --- Main Area for Displaying Results ---

if st.session_state.doc_images:
    st.header("Processed Documents")

    # Use tabs for multiple documents
    if len(st.session_state.doc_images) > 1:
         tab_titles = [f"Document {i+1}" for i in range(len(st.session_state.doc_images))]
         tabs = st.tabs(tab_titles)
    else:
         tabs = [st] # Use the main st object if only one image

    for i, (tab, image) in enumerate(zip(tabs, st.session_state.doc_images)):
        with tab: # Process and display each image in its tab (or main area if single image)
            st.write(f"### Analysis Result for Document {i + 1}")
            with st.spinner(f"Analyzing Document {i+1}..."):
                 segments, extracted_words = extract_and_segment_entities(image)

            if not segments and not extracted_words:
                st.error(f"Could not extract text from Document {i + 1}. Please check the image quality or try another document.")
                st.image(image, caption=f'Document {i + 1} - Analysis Failed', use_column_width=True)
                continue # Skip to the next document/tab if analysis failed

            # --- Display Results in Columns ---
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Original Document")
                st.image(image, caption=f'Document {i + 1}', use_column_width=True)

            with col2:
                st.subheader("Segmented (Filled)")
                if segments:
                     filled_annotated_image = draw_segmented_boxes(image, segments, fill=True)
                     if filled_annotated_image:
                          st.image(filled_annotated_image, caption=f"Segmented Document {i + 1}", use_column_width=True)
                     else:
                          st.warning("Could not draw filled segmentation.")
                else:
                     st.info("No segments found to display.")

            with col3:
                st.subheader("Segmented (Outlined)")
                if segments:
                     annotated_image = draw_segmented_boxes(image, segments, fill=False)
                     if annotated_image:
                          st.image(annotated_image, caption=f"Segmented Details {i + 1}", use_column_width=True)
                     else:
                          st.warning("Could not draw outlined segmentation.")
                else:
                     st.info("No segments found to display.")


            # --- Display Extracted Text ---
            st.subheader("Extracted OCR Text")
            if extracted_words:
                 formatted_text = " ".join(extracted_words) # Join words for display/download
                 # Displaying the extracted text with a scrollable container
                 st.markdown(
                     f"""
                     <div style='max-height: 300px; overflow-y: scroll; padding: 10px; background-color: #f0f2f6; border: 1px solid #cccccc; border-radius: 5px; font-family: monospace; white-space: pre-wrap;'>
                         {formatted_text}
                     </div>
                     """,
                     unsafe_allow_html=True
                 )

                 # --- Download Options ---
                 st.subheader("Download Extracted Text")
                 col_dl_1, col_dl_2, col_dl_3 = st.columns(3)

                 # Text Download
                 with col_dl_1:
                     text_bytes = formatted_text.encode('utf-8')
                     st.download_button(
                         label="📥 Download as Text (.txt)",
                         data=text_bytes,
                         file_name=f"extracted_text_doc_{i + 1}.txt",
                         mime="text/plain",
                         key=f"txt_dl_{i}"
                     )

                 # PDF Download
                 with col_dl_2:
                     try:
                          pdf_buffer = convert_text_to_pdf(formatted_text)
                          st.download_button(
                              label="📄 Download as PDF",
                              data=pdf_buffer,
                              file_name=f"extracted_text_doc_{i + 1}.pdf",
                              mime="application/pdf",
                              key=f"pdf_dl_{i}"
                          )
                     except Exception as pdf_err:
                          st.error(f"PDF Generation Error: {pdf_err}")

                 # Excel Download
                 with col_dl_3:
                     try:
                          # Create a simple DataFrame for Excel export
                          df = pd.DataFrame({"Extracted Text": [formatted_text]}) # Single cell with all text
                          # Or DataFrame with words per row: df = pd.DataFrame({"Word": extracted_words})
                          excel_buffer = io.BytesIO()
                          with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                              df.to_excel(writer, index=False, sheet_name=f'Doc_{i+1}_Text')
                          excel_buffer.seek(0)
                          st.download_button(
                              label="📊 Download as Excel (.xlsx)",
                              data=excel_buffer,
                              file_name=f"extracted_text_doc_{i + 1}.xlsx",
                              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              key=f"excel_dl_{i}"
                          )
                     except Exception as excel_err:
                          st.error(f"Excel Generation Error: {excel_err}")

            else:
                 st.info("No text was extracted to display or download.")

            st.markdown("---") # Separator between documents if multiple

elif not option.startswith("Scan"): # Show only if no images loaded and not actively scanning
    st.info("Upload documents or use the camera options in the sidebar to get started.")
# --- End Streamlit App ---