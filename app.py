import streamlit as st
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import time

# Sidebar selector for OCR engine
ocr_choice = st.sidebar.radio("Select OCR Engine", ["Easy OCR", "Pytesseract OCR", "Paddle OCR"])

# File uploader for the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded file as bytes, then convert to a NumPy array for OpenCV
    file_bytes = uploaded_file.getvalue()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Also create a PIL image (for pytesseract)
    img_pil = Image.open(BytesIO(file_bytes))
    
    # Display the uploaded image
    st.image(img_cv2, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    if ocr_choice == "Pytesseract OCR":
        # --- Pytesseract OCR ---
        import pytesseract
        # Set the tesseract_cmd path if needed, e.g.:
        # pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
        
        # Perform OCR on the PIL image
        text = pytesseract.image_to_string(img_pil)
        st.subheader("Extracted Text")
        st.text(text)
        
    elif ocr_choice == "Easy OCR":
        # --- Easy OCR ---
        import easyocr
        # Initialize the EasyOCR reader; change languages if necessary.
        reader = easyocr.Reader(['en'])
        # EasyOCR accepts a numpy array; convert to RGB (if desired)
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        results = reader.readtext(img_rgb)
        
        # Format the results for display
        extracted_text = ""
        for (bbox, text, prob) in results:
            extracted_text += f"{text} (Confidence: {prob:.2f})\n"
            
        st.subheader("Extracted Text")
        st.text_area("OCR Output", extracted_text, height=200)
        
    elif ocr_choice == "Paddle OCR":
        # --- Paddle OCR ---
        from paddleocr import PaddleOCR, draw_ocr
        # Initialize PaddleOCR with angle classification enabled
        ocr = PaddleOCR(use_angle_cls=True)
        
        # Run OCR and measure processing time
        start_time = time.time()
        result = ocr.ocr(img_cv2)
        elapsed_time = time.time() - start_time
        st.write(f"Time taken for OCR: {elapsed_time:.4f} seconds")
        
        # Check if OCR results exist
        if result and len(result) > 0:
            # Extract bounding boxes, text, and confidence scores from the first result set.
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]
            
            # Path to the font file (make sure simfang.ttf is in the same folder)
            font_path = './simfang.ttf'
            # Draw the OCR results on the image
            im_show = draw_ocr(img_cv2, boxes, txts, scores, font_path=font_path)
            im_show_rgb = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
            
            # Display the annotated image and extracted text
            st.image(im_show_rgb, caption="OCR Result", use_column_width=True)
            extracted_text = "\n".join(txts)
            st.subheader("Extracted Text")
            st.text_area("OCR Output", extracted_text, height=200)
        else:
            st.write("No text found.")
