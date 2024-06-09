import streamlit as st
from PIL import Image
import numpy as np
import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the fine-tuned model
model = VisionEncoderDecoderModel.from_pretrained("./results3/final_model")

# Load the processor from the base TrOCR model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Define the preprocessing function
def preprocess_image(image):
    # Convert PIL Image to numpy array for processing
    image_np = np.array(image.convert("RGB"))

    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Noise reduction using Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Threshold to get binary image
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the image
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    binary_image = cv2.dilate(binary_image, kernel, iterations=1)
    binary_image = cv2.erode(binary_image, kernel, iterations=1)
    
    # Convert back to RGB PIL Image
    processed_image = Image.fromarray(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB))
    return processor(processed_image, return_tensors="pt").pixel_values

# Define the function to extract text
def extract_text_from_image(pixel_values):
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Streamlit UI
st.title("CAPTCHA Text Extractor")

uploaded_file = st.file_uploader("Choose a CAPTCHA image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded CAPTCHA Image', use_column_width=True)
    
    st.write("Processing...")
    pixel_values = preprocess_image(image)
    extracted_text = extract_text_from_image(pixel_values)
    st.write(f"Extracted Text: {extracted_text}")
