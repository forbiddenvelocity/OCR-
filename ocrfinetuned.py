import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Load the fine-tuned model
model = VisionEncoderDecoderModel.from_pretrained("./results/final_model")

# Load the processor from the base model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Define the preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")
    return processor(image, return_tensors="pt").pixel_values

# Define the function to extract text
def extract_text_from_image(image):
    pixel_values = preprocess_image(image)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def post_process_text(text):
    corrections = {
        '8': 'B',
        'B': '8',
        'M': 'n',
        'n': 'M',
        'W': 'v',
        'v': 'W',
        'O': 'g',
        # Add other common corrections here
    }
    corrected_text = ''.join(corrections.get(c, c) for c in text)
    return corrected_text

# Streamlit UI
st.title("CAPTCHA Text Extractor")

uploaded_file = st.file_uploader("Choose a CAPTCHA image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded CAPTCHA Image', use_column_width=True)
    
    st.write("Processing...")
    extracted_text = extract_text_from_image(image)
    st.write(f"Extracted Text: {extracted_text}")
