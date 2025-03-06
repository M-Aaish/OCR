import streamlit as st
import cv2
import numpy as np
import time

# Import the OCR libraries
from paddleocr import PaddleOCR
import easyocr

# Initialize OCR engines (doing this outside the function avoids re-loading on every run)
ocr_paddle = PaddleOCR(use_angle_cls=True, lang='en')
ocr_easy = easyocr.Reader(['en'])

def process_paddleocr(image_bytes):
    # Convert uploaded bytes to a NumPy array and then decode into an image using OpenCV
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Error: Unable to read the image!")
        return None, None
    start_time = time.time()
    # Run PaddleOCR on the image
    result = ocr_paddle.ocr(img, det=True, rec=True, cls=True)
    time_taken = time.time() - start_time

    extracted_text = ""
    # The result is a list of text lines; each line contains bounding box info and text details.
    for line in result[0]:
        extracted_text += line[1][0] + "\n"
    return extracted_text, time_taken

def process_easyocr(image_bytes):
    # Convert uploaded bytes to an image that EasyOCR can process
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Error: Unable to read the image!")
        return None, None
    start_time = time.time()
    # Run EasyOCR on the image
    result = ocr_easy.readtext(img)
    time_taken = time.time() - start_time

    extracted_text = ""
    # Each result is a tuple: (bounding_box, text, confidence)
    for (_, text, _) in result:
        extracted_text += text + "\n"
    return extracted_text, time_taken

# Streamlit UI
st.title("OCR Streamlit App")

# Sidebar radio button to choose between OCR modes
mode = st.sidebar.radio("Choose OCR Mode", ["EasyOCR", "PaddleOCR"])

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image data in bytes and display the image
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
    
    if mode == "PaddleOCR":
        st.write("Running PaddleOCR...")
        text, time_taken = process_paddleocr(image_bytes)
    else:
        st.write("Running EasyOCR...")
        text, time_taken = process_easyocr(image_bytes)
    
    if text is not None:
        st.text_area("Extracted Text", text, height=300)
        st.write(f"Time taken: {time_taken:.2f} seconds")
